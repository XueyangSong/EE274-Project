from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
import functools

SIZE_BYTE = 32

class LZSSEncoder(DataEncoder):
    def __init__(self, type, binary_type):
        self.type = type
        self.binary_type = binary_type
    """
    Construct one possible encoding table
    Args:
        s: string to encode
        type: "shortest" -- Find shortest possible unmatched literals
              "merged" -- Find longest possible unmatched literal more space efficient than several rows
    Returns:
        table: list of lists representing rows
        Format of output table:
        Unmatched literals | Match length | Match offset
    """
    def encode_literal(self, s) -> list:
        table = []
        search_idx, match_idx = 0, 1
        unmatched = s[0]
        while match_idx < len(s):
            match_length, match_offset = self.find_match(s, search_idx, match_idx)
            if match_length == 0:
                unmatched += s[match_idx]
                match_idx += 1
            else:
                table.append([unmatched, match_length, match_offset])
                unmatched = ""
                match_idx += match_length
        if self.type == "merged":
            merged_table = []
            merged_table.append(table[0])
            for i in range(1, len(table)):
                if (merged_table[-1][0] != "") and (merged_table[-1][1] + len(table[i][0]) < self.row_size(table[i])):
                    cur_unmatched, cur_length, cur_offset = merged_table[-1]
                    if cur_length < cur_offset:
                        to_repeat = cur_unmatched[cur_offset*(-1):cur_offset*(-1)+cur_length]
                    else:
                        to_repeat = cur_unmatched[cur_offset*(-1):]
                    while cur_length > 0:
                        cur_unmatched += to_repeat
                        cur_length -= cur_offset
                    cur_unmatched += table[i][0]
                    merged_table[-1] = [cur_unmatched, table[i][1], table[i][2]]
                else:
                    merged_table.append(table[i])
            return merged_table
        return table

    """
    Get size of a row in bytes
    Assume row size (bytes) = 4 + len(pattern) + 4 + 4
    Args:
        r: row
    Returns:
        row_size: int
    """
    def row_size(self, r) -> int:
        return 4 + len(r[0]) + 4 + 4

    """
    Greedy search that matches the maximum possible string in the look_ahead_buffer to a substring in search_buffer
    Args:
        s: string to process
        search_idx: starting search index in the string to process
        match_idx: starting match index in the string to process
    Returns:
        match length: int
        match offset: int
    """
    def find_match(self, s, search_idx, match_idx) -> (int, int):
        assert match_idx > search_idx, f"Match index at {match_idx} starts before search index at {search_idx}"
        max_match_length, offset = 0, -1
        search_start, search_end, look_ahead_start, look_ahead_end = search_idx, search_idx, match_idx, match_idx
        while search_start < look_ahead_start and search_end < len(s) and look_ahead_end < len(s):
            while look_ahead_end < len(s) and search_end < len(s) and (s[look_ahead_end] == s[search_end]):
                look_ahead_end += 1
                search_end += 1
                if look_ahead_end - look_ahead_start >= max_match_length:
                    max_match_length = look_ahead_end - look_ahead_start
                    offset = look_ahead_start - search_start
            search_start += 1
            search_end = search_start
            look_ahead_end = look_ahead_start
        # print(s[match_idx:(match_idx + max_match_length)])
        return max_match_length, offset

    def encode_block(self, data_block: DataBlock):
        table = self.encode_literal(''.join(data_block.data_list))
        return table

    def encoding(self, data_block: DataBlock):
        return table_to_binary_baseline(self.encode_block(data_block))


"""
Format of table:
    pattern(assume ascii)    match length    match offset
Format of binary:
    concatenation of:
        - 4-byte int: number of rows
        - rows being concatenation of:
            - 4-byte int: number of ascii chars in pattern
            - concatenation of 1-byte ascii encodings (pattern)
            - 4-byte int (match length)
            - 4-byte int (match offset)
"""
def table_to_binary_baseline(table):
    ret = uint_to_bitarray(len(table), SIZE_BYTE)
    for pattern, match_len, match_offset in table:
        # pattern
        ret += uint_to_bitarray(len(pattern), SIZE_BYTE)
        ret += (functools.reduce(lambda b1, b2: b1 + b2, [uint_to_bitarray(ord(c), 8) for c in pattern], BitArray('')))
        # match lenth
        ret += uint_to_bitarray(match_len, SIZE_BYTE)
        # match offset
        ret += uint_to_bitarray(match_offset, SIZE_BYTE)
    return ret

def binary_to_table_baseline(input_bitarray):
    ret, ptr, num_rows = [], SIZE_BYTE, bitarray_to_uint(input_bitarray[:SIZE_BYTE]),
    for _ in range(num_rows):
        # pattern
        num_chars, ptr, pattern =  bitarray_to_uint(input_bitarray[ptr:(ptr+SIZE_BYTE)]), ptr + SIZE_BYTE, ""
        for _ in range(num_chars):
            pattern += chr(bitarray_to_uint(input_bitarray[ptr:(ptr + 8)]))
            ptr += 8
        # match lenth
        match_len, ptr = bitarray_to_uint(input_bitarray[ptr:(ptr+SIZE_BYTE)]), ptr + SIZE_BYTE
        # match offset
        match_offset, ptr = bitarray_to_uint(input_bitarray[ptr:(ptr+SIZE_BYTE)]), ptr + SIZE_BYTE
        ret.append([pattern, match_len, match_offset])
    return ret

class LZSSDecoder(DataDecoder):
    def __init__(self): pass

    def decode_block(self, table):
        ret, ptr, row = ['' for _ in range(sum([len(r[0]) + r[1] for r in table]))], 0, 0
        while ptr < len(ret):
            pattern, match_len, match_offset = table[row]
            for c in pattern:
                ret[ptr] = c
                ptr += 1
            for _ in range(match_len):
                ret[ptr] = ret[ptr - match_offset]
                ptr += 1
            row += 1
        return ''.join(ret)

    def decoding(self, binary):
        table = binary_to_table_baseline(binary)
        return self.decode_block(table)

if __name__ == "__main__":
    encoder_s = LZSSEncoder("shortest", "baseline")
    encoder_m = LZSSEncoder("merged", "baseline")
    decoder = LZSSDecoder()
    # # [['ab', 1, 1], ['', 6, 3], ['c', 2, 4]]
    s = "abbabbabbcab"
    e1 = encoder_s.encode_block(DataBlock(s))
    assert e1 == [['ab', 1, 1], ['', 6, 3], ['c', 2, 4]]
    assert s == decoder.decode_block(e1)
    assert s == decoder.decoding(encoder_s.encoding(DataBlock(s)))

    # shortest: [['A', 1, 1], ['B', 6, 1], ['', 5, 9], ['CD', 4, 2]]
    # other: [['AAB', 6, 1], ['', 5, 9], ['CD', 4, 2]]
    # other: [['AABBBBBBB', 5, 9], ['CD', 4, 2]]
    # merged: [['AABBBBBBBAABBBCD', 4, 2]]
    # print(encode("A"*2 + "B"*7 + "A"*2 + "B"*3 + "CD"*3, "shortest"))
    # print(encode("A"*2 + "B"*7 + "A"*2 + "B"*3 + "CD"*3, "merged"))
    s = "A"*2 + "B"*7 + "A"*2 + "B"*3 + "CD"*3
    e1, e2 = encoder_s.encode_block(DataBlock(s)), encoder_m.encode_block(DataBlock(s))
    assert e1 == [['A', 1, 1], ['B', 6, 1], ['', 5, 9], ['CD', 4, 2]]
    assert e2 == [['AABBBBBBBAABBBCD', 4, 2]]
    assert s == decoder.decode_block(e1)
    assert s == decoder.decode_block(e2)
    assert s == decoder.decoding(encoder_s.encoding(DataBlock(s)))
    assert s == decoder.decoding(encoder_m.encoding(DataBlock(s)))

    # shortest: [['A', 1, 1], ['B', 17, 1], ['C', 1, 1], ['D', 1, 1]]
    # merged: [['AAB', 17, 1], ['CCD', 1, 1]]
    # print(encode("A"*2 + "B"*18 + "C"*2 + "D"*2, "shortest"))
    # print(encode("A"*2 + "B"*18 + "C"*2 + "D"*2, "merged"))
    s = "A"*2 + "B"*18 + "C"*2 + "D"*2
    e1, e2 = encoder_s.encode_block(DataBlock(s)), encoder_m.encode_block(DataBlock(s))
    assert e1 == [['A', 1, 1], ['B', 17, 1], ['C', 1, 1], ['D', 1, 1]]
    assert e2 == [['AAB', 17, 1], ['CCD', 1, 1]]
    assert s == decoder.decode_block(e1)
    assert s == decoder.decode_block(e2)
    assert s == decoder.decoding(encoder_s.encoding(DataBlock(s)))
    assert s == decoder.decoding(encoder_m.encoding(DataBlock(s)))

    # shortest: [['A', 1, 1], ['B', 17, 1], ['', 3, 20] ['C', 1, 1], ['D', 1, 1]]
    # merged: [['AAB', 17, 1], ['', 3, 20], ['CCD', 1, 1]]
    # print(encode("A"*2 + "B"*18 + "AAB" + "C"*2 + "D"*2, "shortest"))
    # print(encode("A"*2 + "B"*18 + "AAB" + "C"*2 + "D"*2, "merged"))
    s = "A"*2 + "B"*18 + "AAB" + "C"*2 + "D"*2
    e1, e2 = encoder_s.encode_block(DataBlock(s)), encoder_m.encode_block(DataBlock(s))
    assert e1 == [['A', 1, 1], ['B', 17, 1], ['', 3, 20], ['C', 1, 1], ['D', 1, 1]]
    assert e2 == [['AAB', 17, 1], ['', 3, 20], ['CCD', 1, 1]]
    assert s == decoder.decode_block(e1)
    assert s == decoder.decode_block(e2)
    assert s == decoder.decoding(encoder_s.encoding(DataBlock(s)))
    assert s == decoder.decoding(encoder_m.encoding(DataBlock(s)))

    t1 = [['ab', 1, 1], ['', 6, 3], ['c', 2, 4]]
    t2 = []
    t3 = [['abcd', 1, 1], ['', 1, 3], ['', 5, 11]]
    assert t1 == binary_to_table_baseline(table_to_binary_baseline(t1))
    assert t2 == binary_to_table_baseline(table_to_binary_baseline(t2))
    assert t3 == binary_to_table_baseline(table_to_binary_baseline(t3))
