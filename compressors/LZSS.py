from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
from compressors.elias_delta_uint_coder import EliasDeltaUintDecoder, EliasDeltaUintEncoder
from compressors.huffman_coder import HuffmanDecoder, HuffmanEncoder
from core.prob_dist import ProbabilityDist
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
import functools
import math
import os
import hashlib
from collections import defaultdict

SIZE_BYTE = 32
MAX_WINDOW_SIZE = 1024
HASH_NUM_BYTES = 10
NUM_HASH_TO_SEARCH = 16
MAX_MATCH_SIZE = 1024

"""
TODO:
1. Evaluate correctness/optimality of optimal parsing.
2. Implement block_encoding/decoding. Currently takes in entire input.
3. Implement search from fixed-size window instead of search from entire history.
4. Add min_match_length.
5. Remove "merged" table type, as the idea is already in `min_match_length` and `optimal_parsing`.
6. Generalize to bytearray inputs instead of strings.
7. Dump output to dataframe instead of print to terminal. Easier for graphing.
8. Test on larger files.
"""

class LZSSEncoder(DataEncoder):
    """
    window_size: int -- how far we allow the compressor to look backwards to find matches
                 Defaults to be MAX_WINDOW_SIZE
    table_type: "shortest" -- Find shortest possible unmatched literals
                "merged" -- Find longest possible unmatched literal more space efficient than several rows
    find_match_method: "basic" -- sequential search for the longest match from the start of the lookahead buffer
                       "hashchain" -- use chained hashes of fixed number bytes to find the longest match
                                     from the start of the lookahead buffer
    greedy_optimal: "greedy" -- greedily search for the longest possible match
                    "optimal" -- implements optimal parsing that may not return the longest possible match for all positions
    binary_type: "baseline" -- Using baseline compression algorithm to convert the table to binary
                 "optimized" -- Using optimized algorithm to convert table to binary
    """
    def __init__(self, table_type, find_match_method, binary_type, greedy_optimal, window_size=MAX_WINDOW_SIZE):
        assert window_size > 0, f"window size is {window_size}, should be larger than 0."
        self.window_size = window_size
        self.window_mask = self.window_size - 1
        self.table_type = table_type
        self.binary_type = binary_type
        self.find_match_method = find_match_method
        self.greedy_optimal = greedy_optimal
        if self.find_match_method == "hashchain":
            self.hash_table = dict() # key -> closest index of same key
            self.chained_prev = [0] * self.window_size # storing the last index of same key
            # self.hash_collision_check = defaultdict(set)

    """
    Parse a datablock by the selected method into a table
    Args:
        data_block: DataBlock
    Returns:
        table: list of lists representing rows
        Format of output table:
        Unmatched literals | Match length | Match offset
    """
    def block_to_table(self, data_block: DataBlock) -> list:
        if self.greedy_optimal == "greedy":
            table = self.greedy_parsing(''.join(data_block.data_list))
        elif self.greedy_optimal == "optimal":
            table = self.optimal_parsing(''.join(data_block.data_list))
        # print(table)
        return table

    """
    Greedily construct the encoding table
    Args:
        s: string to encode
    Returns:
        table: list of lists representing rows
        Format of output table:
        Unmatched literals | Match length | Match offset
    """
    def greedy_parsing(self, s) -> list:
        table = []
        search_idx, match_idx = 0, 0
        unmatched = ""
        while match_idx < len(s):
            search_idx = match_idx - self.window_size if match_idx - self.window_size >= 0 else 0
            if self.find_match_method == "basic":
                match_count, match_lengths, match_offsets = self.find_match_basic(s, search_idx, match_idx, "greedy")
                # print(match_lengths)
            elif self.find_match_method == "hashchain":
                match_count, match_lengths, match_offsets = self.find_match_hashchain(s, search_idx, match_idx, "greedy")
            if match_count == 0:
                unmatched += s[match_idx]
                match_idx += 1
            else:
                table.append([unmatched, match_lengths[0], match_offsets[0]])
                unmatched = ""
                match_idx += match_lengths[0]
        if unmatched != "":
            table.append([unmatched, 0, 0])
        return table

    """
    Implements optimal parsing that may not return the longest possible match for all positions
    Args:
        s: string to encode
    Returns:
        table: list of lists representing rows
        Format of output table:
        Unmatched literals | Match length | Match offset
    """
    def optimal_parsing(self, s) -> list:
        # match_size = min(MAX_MATCH_SIZE, len(s) - match_idx)
        match_size = len(s)
        prices = [99999999] * (match_size + 1)
        lengths = [0] * (match_size + 1)
        dists = [0] * (match_size + 1)
        prices[0] = 0

        # forward pass
        for i in range(match_size):
            lit_cost = prices[i] + self.literal_price(s[i])
            if lit_cost < prices[i + 1]:
                prices[i + 1] = lit_cost
                lengths[i + 1] = 1
                dists[i + 1] = 0
            # Don't try matches close to end of buffer
            # if i + 4 >= length:
            #     continue
            if self.find_match_method == "basic":
                num_matches, match_len, match_dist = self.find_match_basic(s, 0, i, "optimal")
            elif self.find_match_method == "hashchain":
                num_matches, match_len, match_dist = self.find_match_hashchain(s, 0, i, "optimal")
            for j in range(num_matches):
                match_cost = prices[i] + self.match_price(match_len[j], match_dist[j])
                if match_cost < prices[i + match_len[j]]:
                    prices[i + match_len[j]] = match_cost
                    lengths[i + match_len[j]] = match_len[j]
                    dists[i + match_len[j]] = match_dist[j]

        # backward pass
        # Process from the end of current block
        table = []
        unmatched_literal = ""
        cur = match_size
        while cur > 0 and lengths[cur] == 1:
            unmatched_literal = s[cur - 1] + unmatched_literal
            cur -= 1
        table.insert(0, [unmatched_literal, 0, 0])
        while cur > 0:
            if lengths[cur] > 1:
                if len(table) != 0:
                    table[0][0] = unmatched_literal
                    unmatched_literal = ""
                table.insert(0, [unmatched_literal, lengths[cur], dists[cur]])
                cur -= lengths[cur]
            else:
                unmatched_literal = s[cur - 1] + unmatched_literal
                cur -= 1
        if unmatched_literal != "":
            table[0][0] = unmatched_literal
        return table

    """
    Empirical cost heuristics for each literal
    Args:
        c: the current literal to be matched
    Returns:
        Empirical cost for each literal (Credit: https://glinscott.github.io/lz/index.html#toc4.2.2)
    """
    def literal_price(self, c):
        return 6

    """
    Empirical cost heuristics for a match
    Args:
        length: match length
        dist: match offset
    Returns:
        Empirical cost for each match (Credit: https://glinscott.github.io/lz/index.html#toc4.2.2)
    """
    def match_price(self, length, dist):
        len_cost = 6 + math.log2(length)
        dist_cost = max(0, math.log2(dist) - 3)
        return len_cost + dist_cost

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
        greedy_optimal: string to indicate whether we greedily return longest match or return all matches
    Returns:
        match count: int. Number of matches found. Always equals 1 if `greedy_optimal` is "greedy".
        match length: [int]. Lengths of matches found. Always size 1 if `greedy_optimal` is "greedy".
        match offset: [int]. Offsets of matches found. Always size 1 if `greedy_optimal` is "greedy".
    """
    def find_match_basic(self, s, search_idx, match_idx, greedy_optimal) -> (int, [int], [int]):
        assert match_idx >= search_idx, f"Match index at {match_idx} starts before search index at {search_idx}"
        match_count = 0
        max_match_length = 0
        lengths, offsets = [], []
        search_start, search_end, look_ahead_start, look_ahead_end = search_idx, search_idx, match_idx, match_idx
        while search_start < look_ahead_start and search_end < len(s) and look_ahead_end < len(s):
            while look_ahead_end < len(s) and search_end < len(s) and (s[look_ahead_end] == s[search_end]):
                look_ahead_end += 1
                search_end += 1
                if greedy_optimal == "greedy":
                    if look_ahead_end - look_ahead_start >= max_match_length:
                        lengths, offsets = [look_ahead_end - look_ahead_start], [look_ahead_start - search_start]
                        max_match_length = lengths[0]
                        match_count = 1
                elif greedy_optimal == "optimal":
                    if look_ahead_end - look_ahead_start > 0:
                        lengths.append(look_ahead_end - look_ahead_start)
                        offsets.append(look_ahead_start - search_start)
                        match_count += 1
            search_start += 1
            search_end = search_start
            look_ahead_end = look_ahead_start
        return match_count, lengths, offsets

    """
    Use a hashtable keyed by HASH_NUM_BYTES-long prefix hash and storing lists of indices
    To find the longest match starting from match index in the search buffer
    Args:
        s: string to process
        search_idx: starting search index in the string to process
        match_idx: starting match index in the string to process
        greedy_optimal: string to indicate whether we greedily return longest match or return all matches
    Returns:
        match count: int. Number of matches found. Always equals 1 if `greedy_optimal` is "greedy".
        match length: [int]. Lengths of matches found. Always size 1 if `greedy_optimal` is "greedy".
        match offset: [int]. Offsets of matches found. Always size 1 if `greedy_optimal` is "greedy".
    """
    def find_match_hashchain(self, s, search_idx, match_idx, greedy_optimal) -> (int, [int], [int]):
        assert match_idx >= search_idx, f"Match index at {match_idx} starts before search index at {search_idx}"
        match_count = 0
        max_match_length = 0
        lengths, offsets = [], []
        prefix_to_hash = s[match_idx : match_idx + HASH_NUM_BYTES]
        # hash_key = hashlib.sha256(bytes(prefix_to_hash, 'utf-8'))
        # print(hash_key)
        hash_key = hash(prefix_to_hash)

        # self.hash_collision_check[hash_key] .add(prefix_to_hash)

        # Only search since search_idx
        # If current `hash_key` doesn't exist in `hash_table`, no same prefix has been seen
        if hash_key not in self.hash_table.keys():
            self.chained_prev[match_idx & self.window_mask] = -1
            self.hash_table[hash_key] = match_idx
        else:
            # print("here")
            cur_search_idx = self.hash_table[hash_key]
            num_hash_searched = 0
            while cur_search_idx >= search_idx and num_hash_searched < NUM_HASH_TO_SEARCH:
                assert match_idx >= cur_search_idx, f"Match index at {match_idx} starts before current search index at {cur_search_idx}"
                cur_match_length = self.match_length(s, cur_search_idx, match_idx)
                if greedy_optimal == "greedy":
                    if cur_match_length > max_match_length:
                        lengths, offsets = [cur_match_length], [match_idx - cur_search_idx]
                        max_match_length = cur_match_length
                        match_count = 1
                elif greedy_optimal == "optimal":
                    if cur_match_length > 0:
                        lengths.append(cur_match_length)
                        offsets.append(match_idx - cur_search_idx)
                        match_count += 1
                num_hash_searched += 1
                cur_search_idx = self.chained_prev[cur_search_idx & self.window_mask]
            self.chained_prev[match_idx & self.window_mask] = self.hash_table[hash_key]
            self.hash_table[hash_key] = match_idx
        return match_count, lengths, offsets

    """
    Find longest matching substring starting from match_idx and search_idx
    Args:
        s: string to process
        search_idx: starting search index in the string to process
        match_idx: starting match index in the string to process
    Returns:
        length: int
    """
    def match_length(self, s, search_idx, match_idx) -> (int):
        cur_search, cur_match = search_idx + HASH_NUM_BYTES, match_idx + HASH_NUM_BYTES
        length = HASH_NUM_BYTES
        while cur_search < len(s) and cur_match < len(s) and s[cur_search] == s[cur_match]:
            length += 1
            cur_search += 1
            cur_match += 1
        return length

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
    def table_to_binary_baseline(self, table) -> BitArray:
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

    '''
    Format of table:
        pattern(assume ascii)    match length    match offset
    Format of binary:
        concatenation of:
            - number of bits used for huffman encoding
            - Huffman encoding of all patterns
            - min match length
            - rows being concatenation of:
                - pattern == '': 0
                  pattern != '': 1 + number of chars in pattern + Huffman encoding of pattern
                - Elias Delta encoded int (match length - min match length)
                - Elias Delta encoded int (match offset)
    '''
    def table_to_binary_optimized(self, table) -> BitArray:
        if len(table) == 0: return BitArray('')
        text = ''.join([x for x, _, _ in table])
        text = [ord(c) for c in text]
        counts = DataBlock(text).get_counts()
        prob_dict = ProbabilityDist.normalize_prob_dict(counts).prob_dict
        prob_dist_sorted = ProbabilityDist({i: prob_dict[i] for i in sorted(prob_dict)})
        literal_encoding = HuffmanEncoder(prob_dist_sorted).encode_block(DataBlock(text))
        for i in range(256):
            if i not in counts:
                counts[i] = 0
        counts_list = [counts[i] for i in range(256)]
        counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
        ed_int_encoder = EliasDeltaUintEncoder()
        ret = ed_int_encoder.encode_symbol(len(literal_encoding)) + literal_encoding + ed_int_encoder.encode_symbol(len(counts_encoding)) + counts_encoding
        min_match_len = min([x for _, x, _ in table])
        ret += ed_int_encoder.encode_symbol(min_match_len)
        for pattern, match_len, match_offset in table:
            # pattern
            if pattern == '': ret += BitArray('0')
            else:
                ret += BitArray('1')
                ret += ed_int_encoder.encode_symbol(len(pattern))
            # match lenth
            ret += ed_int_encoder.encode_symbol(match_len - min_match_len)
            # match offset
            ret += ed_int_encoder.encode_symbol(match_offset)
        return ret

    def encoding(self, data_block: DataBlock) -> BitArray:
        if self.binary_type == "baseline":
            return self.table_to_binary_baseline(self.block_to_table(data_block))
        return self.table_to_binary_optimized(self.block_to_table(data_block))

class LZSSDecoder(DataDecoder):
    def __init__(self, binary_type):
        self.binary_type = binary_type

    def table_to_block(self, table):
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

    def binary_to_table_baseline(self, input_bitarray):
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

    def binary_to_table_optimized(self, input_bitarray):
        if len(input_bitarray) == 0: return []
        ed_int_decoder = EliasDeltaUintDecoder()
        # number of bits for Huffman
        huffman_len, num_bits_consumed = ed_int_decoder.decode_symbol(input_bitarray)
        huffman_text = input_bitarray[num_bits_consumed : (huffman_len + num_bits_consumed)]
        rest_table = input_bitarray[(huffman_len + num_bits_consumed) : ]
        count_len, num_bits_consumed = ed_int_decoder.decode_symbol(rest_table)
        count = rest_table[num_bits_consumed : (count_len + num_bits_consumed)]
        rest_table = rest_table[(count_len + num_bits_consumed) : ]
        # min match length
        min_match_len, num_bits_consumed = ed_int_decoder.decode_symbol(rest_table)
        ret = []
        while num_bits_consumed < len(rest_table):
            # pattern
            num_bits_consumed += 1
            if rest_table[num_bits_consumed - 1] == 0:
                pattern_len = 0
            else:
                pattern_len, n = ed_int_decoder.decode_symbol(rest_table[num_bits_consumed : ])
                num_bits_consumed += n
            # match length
            match_len, n = ed_int_decoder.decode_symbol(rest_table[num_bits_consumed : ])
            num_bits_consumed += n
            match_len += min_match_len
            # match offset
            match_offset, n = ed_int_decoder.decode_symbol(rest_table[num_bits_consumed : ])
            num_bits_consumed += n
            ret.append([pattern_len, match_len, match_offset])
        literal_counts, _ = ed_int_decoder.decode_block(count)
        literal_counts = literal_counts.data_list
        literal_counts = {i: literal_counts[i] for i in range(256) if literal_counts[i] > 0}
        prob_dist = ProbabilityDist.normalize_prob_dict(literal_counts)
        decoded_literals, _ = HuffmanDecoder(prob_dist).decode_block(huffman_text)
        decoded_literals = [chr(c) for c in decoded_literals.data_list]
        cnt = 0
        for i in range(len(ret)):
            l = ret[i][0]
            ret[i][0] = ''.join(decoded_literals[cnt : (cnt + l)])
            cnt += l
        return ret

    def decoding(self, binary):
        if self.binary_type == "baseline":
            return self.table_to_block(self.binary_to_table_baseline(binary))
        return self.table_to_block(self.binary_to_table_optimized(binary))


if __name__ == "__main__":

    def read_as_test_str(path: str):
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), path)) as f:
            contents = f.read()
        return contents

    def enc_dec_equality(s: str, table_type: str, find_match_method: str, binary_type: str, greedy_optimal: str):
        encoder = LZSSEncoder(table_type, find_match_method, binary_type, greedy_optimal)
        decoder = LZSSDecoder(binary_type)
        encoded = encoder.encoding(DataBlock(s))
        # print(encoder.hash_collision_check)
        # output_list = [li for li in difflib.ndiff(s, decoder.decoding(encoded)) if li[0] != ' ']
        # print(output_list)
        print(decoder.decoding(encoded))
        assert s == decoder.decoding(encoded)
        print("{} encoded with {} using table type {} and {}/{} has output length: {} and compression rate: {}".format("", binary_type, table_type, find_match_method, greedy_optimal, len(encoded)/8, len(encoded)/8/len(s)))

    TABLE_TYPE_ARGS = ["shortest"]
    FIND_MATCH_METHOD_ARGS = ["hashchain"]
    BINARY_TYPE_ARGS = ["baseline"]
    GREEDY_OPTIMAL = ["optimal"]
    TEST_PATHS = [
                    "../test/sof_cleaned.txt"
                    ]
    TEST_STRS = [
                 # "abb"*3 + "cab",
                 # "A"*2 + "B"*7 + "A"*2 + "B"*3 + "CD"*3,
                 # "A"*2 + "B"*18 + "C"*2 + "D"*2,
                 # "A"*2 + "B"*18 + "AAB" + "C"*2 + "D"*2,
                 # "ABCABC"
                ]
    TEST_STRS.extend([read_as_test_str(path) for path in TEST_PATHS])
    # print(len(TEST_STRS[0]))

    for s in TEST_STRS:
        for table_type in TABLE_TYPE_ARGS:
            for find_match_method in FIND_MATCH_METHOD_ARGS:
                for binary_type in BINARY_TYPE_ARGS:
                    for greedy_optimal in GREEDY_OPTIMAL:
                        enc_dec_equality(s, table_type, find_match_method, binary_type, greedy_optimal)




