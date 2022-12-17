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
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import argparse

SIZE_BYTE = 32
MAX_WINDOW_SIZE = 1024 * 100
HASH_NUM_BYTES = 7
NUM_HASH_TO_SEARCH = 16


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

    def __init__(
        self,
        table_type,
        find_match_method,
        binary_type,
        greedy_optimal,
        window_size=MAX_WINDOW_SIZE,
        hash_num_bytes=HASH_NUM_BYTES,
        num_hash_to_search=HASH_NUM_BYTES,
    ):
        # assert window_size > 0, f"window size is {window_size}, should be larger than 0."
        # self.window_size = window_size
        # self.window_mask = self.window_size - 1
        self.table_type = table_type
        self.binary_type = binary_type
        self.find_match_method = find_match_method
        self.greedy_optimal = greedy_optimal
        if self.find_match_method == "hashchain":
            self.hash_table = dict()  # key -> closest index of same key
            self.chained_prev = [0] * MAX_WINDOW_SIZE  # storing the last index of same key
            self.hash_num_bytes = hash_num_bytes
            self.num_hash_to_search = num_hash_to_search

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
            table = self.greedy_parsing("".join(data_block.data_list))
        elif self.greedy_optimal == "optimal":
            table = self.optimal_parsing("".join(data_block.data_list))
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
            # search_idx = match_idx - self.window_size if match_idx - self.window_size >= 0 else 0
            if self.find_match_method == "basic":
                match_count, match_lengths, match_offsets = self.find_match_basic(
                    s, search_idx, match_idx, "greedy"
                )
            elif self.find_match_method == "hashchain":
                match_count, match_lengths, match_offsets = self.find_match_hashchain(
                    s, search_idx, match_idx, "greedy"
                )
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
        assert (
            match_idx >= search_idx
        ), f"Match index at {match_idx} starts before search index at {search_idx}"
        match_count = 0
        max_match_length = 0
        lengths, offsets = [], []
        search_start, search_end, look_ahead_start, look_ahead_end = (
            search_idx,
            search_idx,
            match_idx,
            match_idx,
        )
        while search_start < look_ahead_start and search_end < len(s) and look_ahead_end < len(s):
            while (
                look_ahead_end < len(s)
                and search_end < len(s)
                and (s[look_ahead_end] == s[search_end])
            ):
                look_ahead_end += 1
                search_end += 1
                if greedy_optimal == "greedy":
                    if look_ahead_end - look_ahead_start >= max_match_length:
                        lengths, offsets = [look_ahead_end - look_ahead_start], [
                            look_ahead_start - search_start
                        ]
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
        assert (
            match_idx >= search_idx
        ), f"Match index at {match_idx} starts before search index at {search_idx}"
        match_count = 0
        max_match_length = 0
        lengths, offsets = [], []
        prefix_to_hash = s[match_idx : match_idx + self.hash_num_bytes]
        hash_key = hash(prefix_to_hash)

        # Only search since search_idx
        # If current `hash_key` doesn't exist in `hash_table`, no same prefix has been seen
        if hash_key not in self.hash_table.keys():
            # self.chained_prev[match_idx & self.window_mask] = -1
            self.chained_prev[match_idx] = -1
            self.hash_table[hash_key] = match_idx
        else:
            cur_search_idx = self.hash_table[hash_key]
            num_hash_searched = 0
            # seach_idx = max(search_idx, match_idx - self.window_size)
            while cur_search_idx >= search_idx and num_hash_searched < self.num_hash_to_search:
                assert (
                    match_idx >= cur_search_idx
                ), f"Match index at {match_idx} starts before current search index at {cur_search_idx}"
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
                # cur_search_idx = self.chained_prev[cur_search_idx & self.window_mask]
                cur_search_idx = self.chained_prev[cur_search_idx]
            # self.chained_prev[match_idx & self.window_mask] = self.hash_table[hash_key]
            self.chained_prev[match_idx] = self.hash_table[hash_key]
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
        cur_search, cur_match = search_idx + self.hash_num_bytes, match_idx + self.hash_num_bytes
        length = self.hash_num_bytes
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
            ret += functools.reduce(
                lambda b1, b2: b1 + b2, [uint_to_bitarray(ord(c), 8) for c in pattern], BitArray("")
            )
            # match lenth
            ret += uint_to_bitarray(match_len, SIZE_BYTE)
            # match offset
            ret += uint_to_bitarray(match_offset, SIZE_BYTE)
        return ret

    """
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
    """

    def table_to_binary_optimized(self, table) -> BitArray:
        if len(table) == 0:
            return BitArray("")
        # prepare for huffman
        # concatenating all texts to one long string
        text = "".join([x for x, _, _ in table])
        text = [ord(c) for c in text]
        counts = DataBlock(text).get_counts()
        prob_dict = ProbabilityDist.normalize_prob_dict(counts).prob_dict
        prob_dist_sorted = ProbabilityDist({i: prob_dict[i] for i in sorted(prob_dict)})
        literal_encoding = HuffmanEncoder(prob_dist_sorted).encode_block(DataBlock(text))
        for i in range(256):
            if i not in counts:
                counts[i] = 0
        counts_list, ed_int_encoder = [counts[i] for i in range(256)], EliasDeltaUintEncoder()
        counts_encoding = ed_int_encoder.encode_block(DataBlock(counts_list))
        ret = (
            ed_int_encoder.encode_symbol(len(literal_encoding))
            + literal_encoding
            + ed_int_encoder.encode_symbol(len(counts_encoding))
            + counts_encoding
        )
        # if only one row in table
        if len(table) == 1:
            return ret
        # if last row end with 0 -> not encoding last row
        if table[-1][1] == 0:
            table = table[:-1]
        min_match_len = min([x for _, x, _ in table])
        ret += ed_int_encoder.encode_symbol(min_match_len)
        for pattern, match_len, match_offset in table:
            # pattern
            ret += ed_int_encoder.encode_symbol(len(pattern))
            # match lenth
            ret += ed_int_encoder.encode_symbol(match_len - min_match_len)
            # match offset
            ret += ed_int_encoder.encode_symbol(match_offset)
        return ret

    """
    concatenation of:
    - fse encoded patterns
        - EliasDelta encoded int: len of encoded distribution
        - encoded distribution with encoding_ascii_distribution
        - EliasDelta encoded int: len of fse encoded patterns
        - fse encoded pattern text with encoding_process
    - fse encoded pattern_len
        - EliasDelta encoded int: len of encoded distribution
        - encoded distribution with encoding_num_distribution
        - EliasDelta encoded int: len of encoded pattern_len
        - fse encoded pattern_len text with encoding_process
    - EliasDelta encoded int: min_match_len
    - fse encoded match_len list
        - EliasDelta encoded int: len of encoded distribution
        - encoded distribution with encoding_num_distribution
        - EliasDelta encoded int: len of encoded match_len
        - fse encoded match_len text with encoding_process
    - fse encoded match_offset list
        - EliasDelta encoded int: len of encoded distribution
        - encoded distribution with encoding_num_distribution
        - EliasDelta encoded int: len of encoded match_offset
        - fse encoded match_offset text with encoding_process
    """

    def table_to_binary_fse(self, table) -> BitArray:
        """
        l is the list of ints to be encoded

        - int : number of pairs to be looked up
        - (int, int) : value and counts
        """

        def encoding_num_distribution(d):
            encoder = EliasDeltaUintEncoder()
            ret = encoder.encode_symbol(len(d))
            for k, v in d.items():
                ret += encoder.encode_symbol(k) + encoder.encode_symbol(v)
            return ret

        """
        l is the list of bytes to be encoded

        - int (256 times): counts of each ascii value
        """

        def encoding_ascii_distribution(l):
            encoder = EliasDeltaUintEncoder()
            d = normalizeFrequencies(dict(Counter(l)))
            ret = BitArray()
            for i in range(255):
                if chr(i) not in d.keys():
                    ret += encoder.encode_symbol(0)
                else:
                    ret += encoder.encode_symbol(d[chr(i)])
            return ret

        """
        l is the uniformly distributed table generated
        text is the text to be encoded (might be int or char)

        concatenation of:
        - final state
        - offsets following decoding process
        """

        def encoding_process(l, text):
            denominator = len(l)
            counter = Counter(l)
            ret = BitArray()
            prev_symbol = text[0]
            prev_state = max(
                index for index, item in enumerate(l) if item == prev_symbol
            )  # l.index(prev_symbol)
            for curr_symbol in text[1:]:
                l1, c1, l2, c2 = formSubrange(counter[curr_symbol], denominator)
                # find subrange lie in
                # lie in larger subrange
                if l1 * c1 <= prev_state:
                    len_to_write = int(math.log2(l2))
                    index_subrange = (prev_state - l1 * c1) % l2
                    offset_subrange = uint_to_bitarray(index_subrange, len_to_write)
                    ret += offset_subrange
                    # update prev_state
                    cnt = 0
                    for i, s in enumerate(l):
                        if (cnt == (prev_state - c1 * l1) // l2 + c1) and (s == curr_symbol):
                            prev_state = i
                            break
                        elif s == curr_symbol:
                            cnt += 1
                    prev_symbol = curr_symbol
                # lie in smaller subrange
                else:
                    len_to_write = int(math.log2(l1))
                    index_subrange = prev_state % l1
                    if len_to_write != 0:
                        ret += uint_to_bitarray(index_subrange, len_to_write)
                    # update prev_state
                    cnt = 0
                    for i, s in enumerate(l):
                        if (cnt == prev_state // l1) and (s == curr_symbol):
                            prev_state = i
                            break
                        elif s == curr_symbol:
                            cnt += 1
                    prev_symbol = curr_symbol
            return (
                EliasDeltaUintEncoder().encode_symbol(prev_state)
                + EliasDeltaUintEncoder().encode_symbol(len(text))
                + ret
            )

        encoder = EliasDeltaUintEncoder()
        # encoding patterns
        patterns = "".join([t for t, _, _ in table])
        d = normalizeFrequencies(dict(Counter(patterns)))
        encoded_pattern_distribution = encoding_ascii_distribution(d)
        
        l = formUniformList(d)
        encoded_pattern = encoding_process(l, patterns)
        ret = (
            encoder.encode_symbol(len(encoded_pattern_distribution))
            + encoded_pattern_distribution
            + encoder.encode_symbol(len(encoded_pattern))
            + encoded_pattern
        )
        # pattern_len
        pattern_list = [len(x) for x, _, _ in table]
        d = normalizeFrequencies(dict(Counter(pattern_list)))
        encoded_pattern_len_distribution = encoding_num_distribution(d)
        l = formUniformList(d)
        encoded_pattern_len = encoding_process(l, pattern_list)
        ret += (
            encoder.encode_symbol(len(encoded_pattern_len_distribution))
            + encoded_pattern_len_distribution
        )
        ret += encoder.encode_symbol(len(encoded_pattern_len)) + encoded_pattern_len
        if (table[-1][1] == 0) and (table[-1][2] == 0):
            table = table[:-1]
        # min_match_len
        min_match_len = min([x for _, x, _ in table])
        ret += encoder.encode_symbol(min_match_len)
        # match_len_list
        match_len_list = [x - min_match_len for _, x, _ in table]
        d = normalizeFrequencies(dict(Counter(match_len_list)))
        encoded_match_len_distribution = encoding_num_distribution(d)
        l = formUniformList(d)
        encoded_match_len = encoding_process(l, match_len_list)
        ret += (
            encoder.encode_symbol(len(encoded_match_len_distribution))
            + encoded_match_len_distribution
        )
        ret += encoder.encode_symbol(len(encoded_match_len)) + encoded_match_len
        # match_offset_list
        match_offset_list = [x for _, _, x in table]
        d = normalizeFrequencies(dict(Counter(match_offset_list)))
        encoded_match_offset_distribution = encoding_num_distribution(d)
        l = formUniformList(d)
        encoded_match_offset = encoding_process(l, match_offset_list)
        ret += (
            encoder.encode_symbol(len(encoded_match_offset_distribution))
            + encoded_match_offset_distribution
        )
        ret += encoder.encode_symbol(len(encoded_match_offset)) + encoded_match_offset

        return ret

    def encoding(self, data_block: DataBlock) -> BitArray:
        if self.binary_type == "baseline":
            return self.table_to_binary_baseline(self.block_to_table(data_block))
        if self.binary_type == "optimized":
            return self.table_to_binary_optimized(self.block_to_table(data_block))
        return self.table_to_binary_fse(self.block_to_table(data_block))


class LZSSDecoder(DataDecoder):
    def __init__(self, binary_type):
        self.binary_type = binary_type

    def table_to_block(self, table):
        ret, ptr, row = ["" for _ in range(sum([len(r[0]) + r[1] for r in table]))], 0, 0
        while ptr < len(ret):
            pattern, match_len, match_offset = table[row]
            for c in pattern:
                ret[ptr] = c
                ptr += 1
            for _ in range(match_len):
                ret[ptr] = ret[ptr - match_offset]
                ptr += 1
            row += 1
        return "".join(ret)

    def binary_to_table_baseline(self, input_bitarray):
        ret, ptr, num_rows = (
            [],
            SIZE_BYTE,
            bitarray_to_uint(input_bitarray[:SIZE_BYTE]),
        )
        for _ in range(num_rows):
            # pattern
            num_chars, ptr, pattern = (
                bitarray_to_uint(input_bitarray[ptr : (ptr + SIZE_BYTE)]),
                ptr + SIZE_BYTE,
                "",
            )
            for _ in range(num_chars):
                pattern += chr(bitarray_to_uint(input_bitarray[ptr : (ptr + 8)]))
                ptr += 8
            # match lenth
            match_len, ptr = (
                bitarray_to_uint(input_bitarray[ptr : (ptr + SIZE_BYTE)]),
                ptr + SIZE_BYTE,
            )
            # match offset
            match_offset, ptr = (
                bitarray_to_uint(input_bitarray[ptr : (ptr + SIZE_BYTE)]),
                ptr + SIZE_BYTE,
            )
            ret.append([pattern, match_len, match_offset])
        return ret

    def binary_to_table_optimized(self, input_bitarray):
        if len(input_bitarray) == 0:
            return []
        ed_int_decoder = EliasDeltaUintDecoder()
        # number of bits for Huffman
        huffman_len, num_bits_consumed = ed_int_decoder.decode_symbol(input_bitarray)
        huffman_text, huffman_text_len = (
            input_bitarray[num_bits_consumed : (huffman_len + num_bits_consumed)],
            huffman_len + num_bits_consumed,
        )
        count_len, n = ed_int_decoder.decode_symbol(input_bitarray[huffman_text_len:])
        count = input_bitarray[(huffman_text_len + n) : (huffman_text_len + count_len + n)]
        rest_table = input_bitarray[(huffman_text_len + count_len + n) :]

        # decoding texts
        literal_counts, _ = ed_int_decoder.decode_block(count)
        literal_counts = literal_counts.data_list
        prob_dist = ProbabilityDist.normalize_prob_dict(
            {i: literal_counts[i] for i in range(256) if literal_counts[i] > 0}
        )
        decoded_literals, _ = HuffmanDecoder(prob_dist).decode_block(huffman_text)
        decoded_literals = [chr(c) for c in decoded_literals.data_list]

        # if reaching the end -> only one row in table
        if not len(rest_table):
            return [["".join(decoded_literals), 0, 0]]
        # min match length
        (min_match_len, num_bits_consumed), ret = ed_int_decoder.decode_symbol(rest_table), []
        while num_bits_consumed < len(rest_table):
            # pattern
            # print(num_bits_consumed, len(rest_table), num_bits_consumed < len(rest_table), rest_table[num_bits_consumed:])
            pattern_len, n = ed_int_decoder.decode_symbol(rest_table[num_bits_consumed:])
            num_bits_consumed += n
            # match length
            match_len, n = ed_int_decoder.decode_symbol(rest_table[num_bits_consumed:])
            num_bits_consumed += n
            # match offset
            match_offset, n = ed_int_decoder.decode_symbol(rest_table[num_bits_consumed:])
            num_bits_consumed += n
            # append to table
            ret.append([pattern_len, match_len + min_match_len, match_offset])
        cnt = 0
        # add to returning table
        for i in range(len(ret)):
            l = ret[i][0]
            ret[i][0] = "".join(decoded_literals[cnt : (cnt + l)])
            cnt += l
        if cnt < len(decoded_literals):
            ret.append(["".join(decoded_literals[cnt:]), 0, 0])
        return ret

    def binary_to_table_fse(self, input_bitarray):
        def decoding_num_distribution(input_bitarray):
            decoder = EliasDeltaUintDecoder()
            num, num_bits_consumed = decoder.decode_symbol(input_bitarray)
            ret = dict()
            for i in range(num):
                k, n = decoder.decode_symbol(input_bitarray[num_bits_consumed:])
                v, n1 = decoder.decode_symbol(input_bitarray[num_bits_consumed + n :])
                num_bits_consumed += n + n1
                ret[k] = v
            return ret

        def decoding_ascii_distribution(input_bitarray):
            decoder, num_bits_consumed, ret = EliasDeltaUintDecoder(), 0, dict()
            for i in range(255):
                c, n = decoder.decode_symbol(input_bitarray[num_bits_consumed:])
                num_bits_consumed += n
                if c != 0:
                    ret[chr(i)] = c
            return ret

        def decoding_process(l, input_bitarray):
            last_state, num_bits_consumed = EliasDeltaUintDecoder().decode_symbol(input_bitarray)
            ret = [l[last_state]]
            input_bitarray = input_bitarray[num_bits_consumed:]
            len_text, num_bits_consumed = EliasDeltaUintDecoder().decode_symbol(input_bitarray)
            input_bitarray = input_bitarray[num_bits_consumed:]
            ptr = len(input_bitarray)
            last_symbol = ret[0]
            denominator = len(l)
            counter = Counter(l)
            for _ in range(len_text - 1):
                # while ptr > 0 or max(index for index, item in enumerate(l) if item == last_symbol) == last_state:
                l1, c1, l2, c2 = formSubrange(counter[last_symbol], denominator)
                # find which subrange lie in
                cnt = 1
                for c in l[:last_state]:
                    if c == last_symbol:
                        cnt += 1
                # lie in larger subrange
                if cnt > c1:
                    len_to_read = int(math.log2(l2))
                    offset = input_bitarray[(ptr - len_to_read) : ptr]
                    ptr -= len_to_read
                    offset = bitarray_to_uint(offset)

                    new_state = offset + c1 * l1 + (cnt - c1 - 1) * l2
                    # print(l, new_state, len(l))
                    ret.insert(0, l[new_state])
                    last_state = new_state
                    last_symbol = ret[0]
                # lie in smaller subrange
                else:
                    len_to_read = int(math.log2(l1))
                    offset = input_bitarray[ptr - len_to_read : ptr]
                    ptr -= len_to_read
                    if len_to_read == 0 or len(offset) == 0:
                        offset = 0
                    else:
                        offset = bitarray_to_uint(offset)

                    new_state = offset + (cnt - 1) * l1
                    ret.insert(0, l[new_state])
                    last_state = new_state
                    last_symbol = ret[0]
            return ret

        decoder = EliasDeltaUintDecoder()
        # patterns
        len_pattern_distritbution, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_pattern_distribution = input_bitarray[:len_pattern_distritbution]
        input_bitarray = input_bitarray[len_pattern_distritbution:]
        pattern_distritbution = decoding_ascii_distribution(encoded_pattern_distribution)

        len_pattern, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_pattern = input_bitarray[:len_pattern]
        input_bitarray = input_bitarray[len_pattern:]
        l = formUniformList(pattern_distritbution)
        pattern = decoding_process(l, encoded_pattern)

        # pattern_len_list
        len_pattern_len_distritbution, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_pattern_len_distribution = input_bitarray[:len_pattern_len_distritbution]
        input_bitarray = input_bitarray[len_pattern_len_distritbution:]
        pattern_len_distritbution = decoding_num_distribution(encoded_pattern_len_distribution)

        len_pattern_len, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_pattern_len = input_bitarray[:len_pattern_len]
        input_bitarray = input_bitarray[len_pattern_len:]
        l = formUniformList(pattern_len_distritbution)
        pattern_len_list = decoding_process(l, encoded_pattern_len)

        # min_match_len
        min_match_len, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]

        # match_len_list
        len_match_len_distritbution, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_match_len_distribution = input_bitarray[:len_match_len_distritbution]
        input_bitarray = input_bitarray[len_match_len_distritbution:]
        match_len_distritbution = decoding_num_distribution(encoded_match_len_distribution)

        len_match_len, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_match_len = input_bitarray[:len_match_len]
        input_bitarray = input_bitarray[len_match_len:]
        l = formUniformList(match_len_distritbution)
        match_len_list = decoding_process(l, encoded_match_len)

        # match_offset_list
        len_match_offset_distritbution, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_match_offset_distribution = input_bitarray[:len_match_offset_distritbution]
        input_bitarray = input_bitarray[len_match_offset_distritbution:]
        match_offset_distritbution = decoding_num_distribution(encoded_match_offset_distribution)

        len_match_offset, num_bits_consumed = decoder.decode_symbol(input_bitarray)
        input_bitarray = input_bitarray[num_bits_consumed:]
        encoded_match_offset = input_bitarray[:len_match_offset]
        input_bitarray = input_bitarray[len_match_offset:]
        l = formUniformList(match_offset_distritbution)
        match_offset_list = decoding_process(l, encoded_match_offset)

        temp = list(
            zip(pattern_len_list, [x + min_match_len for x in match_len_list], match_offset_list)
        )
        ptr = 0
        ret = []
        for i in range(len(match_len_list)):
            ret.append(
                [
                    "".join(pattern[ptr : ptr + pattern_len_list[i]]),
                    min_match_len + match_len_list[i],
                    match_offset_list[i],
                ]
            )
            ptr += pattern_len_list[i]

        if len(pattern[ptr:]) != 0 : ret.append(["".join(pattern[ptr:]), 0, 0])
        return ret

    def decoding(self, binary):
        if self.binary_type == "baseline":
            return self.table_to_block(self.binary_to_table_baseline(binary))
        if self.binary_type == "optimized":
            return self.table_to_block(self.binary_to_table_optimized(binary))
        return self.table_to_block(self.binary_to_table_fse(binary))


"""
Form subrange of symbol with frequency x occured in 2^n symbols.
Smaller subranges lie on the left side.
denominator needs to be a power of 2
"""


def formSubrange(numerator, denominator):
    l1 = pow(2, (math.floor(math.log2(denominator // numerator))))
    l2 = l1 * 2
    x = (denominator - l2 * numerator) // (l1 - l2)
    return l1, x, l2, numerator - x


def formUniformList(d):
    l = functools.reduce(lambda l, x: l + [x[0]] * x[1], d.items(), [])
    c, ret = Counter(l), []
    while +c:
        # sort ll for determinism
        ll, times = sorted(list(+c)), min([v for _, v in (+c).items()])
        ret.extend(ll * times)
        for k, v in c.items():
            c[k] = v - times if v > 0 else 0
    return ret


# make sure sum of total occurences is a power of 2 while maintaining approximately
# original distribution
def normalizeFrequencies(d):
    n = sum(d.values())
    if math.ceil(math.log2(n)) == math.floor(math.log2(n)):
        return d
    n_prime = pow(2, (math.ceil(math.log2(n))))
    d = dict([(k, (c * n_prime // n)) for (k, c) in d.items()])
    if sum(d.values()) == n_prime:
        return d
    # sort d in descending order
    d = dict(sorted(d.items(), key=lambda item: -item[1]))
    times = n_prime - sum(d.values())
    cnt = 0
    for c in d.keys():
        if cnt < times:
            d[c] += 1
            cnt += 1
        else:
            break
    return d


# =============================== Functions for Testing/Evaluation =====================================
def read_as_test_str(file_name: str):
    path = "../test/{}".format(file_name)
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), path)) as f:
        contents = f.read()
    return contents


def enc_dec_equality(
    file_name: str,
    s: str,
    table_type: str,
    find_match_method: str,
    binary_type: str,
    greedy_optimal: str,
    window_size: int,
    hash_num_bytes=HASH_NUM_BYTES,
    num_hash_to_search=NUM_HASH_TO_SEARCH,
):
    encoder = LZSSEncoder(
        table_type,
        find_match_method,
        binary_type,
        greedy_optimal,
        window_size,
        hash_num_bytes,
        num_hash_to_search,
    )
    decoder = LZSSDecoder(binary_type)
    encoded = encoder.encoding(DataBlock(s))
    assert s == decoder.decoding(encoded)
    # print("{} encoded with {} using table type {} and {}/{} has output length: {} and compression rate: {}".format("", binary_type, table_type, find_match_method, greedy_optimal, len(encoded)/8, len(encoded)/8/len(s)))
    return (
        file_name,
        find_match_method,
        greedy_optimal,
        binary_type,
        hash_num_bytes,
        num_hash_to_search,
        len(encoded) / 8 / len(s),
    )


def eval_as_df(
    test_files_w_window,
    table_type_args,
    binary_type_args,
    greedy_optimal_args,
    find_match_method_args,
    output_path,
):
    l = []
    for file_name, window_size in list(test_files_w_window.items()):
        s = read_as_test_str(file_name)
        for table_type in table_type_args:
            for find_match_method in find_match_method_args:
                for greedy_optimal in greedy_optimal_args:
                    for binary_type in binary_type_args:
                        result = list(
                            enc_dec_equality(
                                file_name,
                                s,
                                table_type,
                                find_match_method,
                                binary_type,
                                greedy_optimal,
                                window_size,
                            )
                        )
                        del result[-2]
                        del result[-2]
                        l.append(result)
    D = np.array(l)
    df = pd.DataFrame(
        data=D, columns=["File Name", "Matching", "Parsing", "Encoding", "Compression Rate"]
    )
    print(df)

    abs_output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_path)
    df.to_csv(abs_output_path)
    print("Finished writing to output path: {}".format(abs_output_path))
    return df


def tune_hash_num_bytes(
    test_files_w_window: dict, greedy_optimal_args, binary_type_args, output_path
):
    l = []
    hash_num_bytes_list = range(3, 10)
    for file_name, window_size in list(test_files_w_window.items()):
        s = read_as_test_str(file_name)
        for greedy_optimal in greedy_optimal_args:
            for binary_type in binary_type_args:
                for hash_num_bytes in hash_num_bytes_list:
                    result = list(
                        enc_dec_equality(
                            file_name,
                            s,
                            "shortest",
                            "hashchain",
                            binary_type,
                            greedy_optimal,
                            window_size,
                            hash_num_bytes,
                        )
                    )
                    del result[-2]
                    l.append(result)
    D = np.array(l)
    df = pd.DataFrame(
        data=D,
        columns=[
            "File Name",
            "Matching",
            "Parsing",
            "Encoding",
            "Hash Prefix Length",
            "Compression Rate",
        ],
    )
    print(df)

    abs_output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_path)
    df.to_csv(abs_output_path)
    print("Finished writing to output path: {}".format(abs_output_path))
    return df


def tune_hash_search_range(
    test_files_w_window: dict, greedy_optimal_args, binary_type_args, output_path
):
    l = []
    hash_num_bytes_list = range(4, 8)
    num_hash_to_search_list = [8, 16, 32]
    for file_name, window_size in list(test_files_w_window.items()):
        s = read_as_test_str(file_name)
        for greedy_optimal in greedy_optimal_args:
            for binary_type in binary_type_args:
                for hash_num_bytes in hash_num_bytes_list:
                    for num_hash_to_search in num_hash_to_search_list:
                        l.append(
                            enc_dec_equality(
                                file_name,
                                s,
                                "shortest",
                                "hashchain",
                                binary_type,
                                greedy_optimal,
                                window_size,
                                hash_num_bytes,
                                num_hash_to_search,
                            )
                        )
    D = np.array(l)
    df = pd.DataFrame(
        data=D,
        columns=[
            "File Name",
            "Matching",
            "Parsing",
            "Encoding",
            "Hash Prefix Length",
            "Prefix Search Range",
            "Compression Rate",
        ],
    )
    print(df)

    abs_output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_path)
    df.to_csv(abs_output_path)
    print("Finished writing to output path: {}".format(abs_output_path))
    return df


# =============================== Functions for Testing/Evaluation =====================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--decompress", help="decompress", action="store_true")
    parser.add_argument("-c", "--compress", help="compress", action="store_true")
    parser.add_argument("-i", "--input", help="input file", type=str)
    parser.add_argument("-o", "--output", help="output file", type=str)
    parser.add_argument("-t", "--table_type", help="table type", type=str)
    parser.add_argument("-m", "--find_match_method", help="find match", type=str)
    parser.add_argument("-b", "--binary_type", help="binary type", type=str)
    parser.add_argument("-g", "--greedy_optimal", help="greedy optimal", type=str)

    args = parser.parse_args()
    if args.decompress and args.compress:
        print("Invalid. Cannot compress and decompress at the same time!")
        quit()

    # s = read_as_test_str("Crooked Man.txt")
    table_type = "shortest"
    find_match_method = "hashchain"
    binary_type = "optimized"
    greedy_optimal = "greedy"
    window_size = MAX_WINDOW_SIZE
    hash_num_bytes = HASH_NUM_BYTES
    num_hash_to_search = NUM_HASH_TO_SEARCH

    if args.decompress:
        if (args.input == None) or (args.output == None):
            print("Invalid. Need input and output files!")
            quit()
        if args.binary_type != None: binary_type = args.binary_type
        s = BitArray()
        with open(args.input, 'rb') as fh:
                s.fromfile(fh)
        s = s[:-2]
        decoded = LZSSDecoder(binary_type).decoding(s)
        f = open(args.output, "a")
        f.write(decoded)
        f.close()

    elif args.compress:
        if (args.input == None) or (args.output == None):
            print("Invalid. Need input and output files!")
            quit()
        s = read_as_test_str(args.input)
        if args.table_type != None: table_type = args.table_type
        if args.find_match_method != None: find_match_method = args.find_match_method
        if args.binary_type != None: binary_type = args.binary_type
        if args.greedy_optimal != None: greedy_optimal = args.greedy_optimal
        encoder = LZSSEncoder(
            table_type,
            find_match_method,
            binary_type,
            greedy_optimal,
            window_size,
            hash_num_bytes,
            num_hash_to_search,
        )
        encoded = encoder.encoding(DataBlock(s))
        with open(args.output, 'wb') as fh:
            encoded.tofile(fh)
    else:
        UNIT_TESTS = [
            "abb" * 3 + "cab",
            "A" * 2 + "B" * 7 + "A" * 2 + "B" * 3 + "CD" * 3,
            "A" * 2 + "B" * 18 + "C" * 2 + "D" * 2,
            "A" * 2 + "B" * 18 + "AAB" + "C" * 2 + "D" * 2,
            "ABCABC",
            # "A" * 100 + "B" * 99 + "ACCC" * 100 + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100,
        ]
        LARGE_TEST_FILES_W_WINDOW = {
            "Sign of Four.txt": MAX_WINDOW_SIZE,
            "Crooked Man.txt": MAX_WINDOW_SIZE,
            "Sign of Four Spaced.txt": MAX_WINDOW_SIZE,
            "Verwandlung.txt": MAX_WINDOW_SIZE,
        }

        TABLE_TYPE_ARGS = ["shortest"]
        FIND_MATCH_METHOD_ARGS = ["basic", "hashchain"]
        BINARY_TYPE_ARGS = ["baseline", "optimized", "fse"]
        GREEDY_OPTIMAL_ARGS = ["greedy", "optimal"]

        # Algorithm combination comparisons
        output_path = "../test/result/algorithms_comparisons_long.csv"
        eval_as_df(
            LARGE_TEST_FILES_W_WINDOW,
            TABLE_TYPE_ARGS,
            BINARY_TYPE_ARGS,
            GREEDY_OPTIMAL_ARGS,
            FIND_MATCH_METHOD_ARGS,
            output_path,
        )

        # Length of hash prefix tuning
        output_path = "../test/result/prefix_length_tuning_crooked.csv"
        # tune_hash_num_bytes(LARGE_TEST_FILES_W_WINDOW, GREEDY_OPTIMAL_ARGS, BINARY_TYPE_ARGS, output_path)

        # Number of prefix searches tuning
        output_path = "../test/result/prefix_search_range_tuning_sof.csv"
        # tune_hash_search_range(LARGE_TEST_FILES_W_WINDOW, GREEDY_OPTIMAL_ARGS, BINARY_TYPE_ARGS, output_path)
