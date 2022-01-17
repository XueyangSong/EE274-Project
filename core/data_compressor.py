import numpy as np
from core.data_block import BitsDataBlock, DataBlock
from core.data_transformer import DataTransformer


class DataCompressor:
    """
    Base Data Compressor
    """

    def __init__(
        self, encoder_transform: DataTransformer = None, decoder_transform: DataTransformer = None
    ):
        self.encoder_transform = encoder_transform
        self.decoder_transform = decoder_transform

    def set_encoder_decoder_params(self, data_block: DataBlock):
        """
        Usually we will set the self.encoder_transform and self.decoder_transform in this function
        as most of time the parameters of the data_block are necessary to create the transformers
        FIXME: This is a bit ugly
        """
        pass

    def encode(self, data_block: DataBlock):
        """
        The core encode function of the compressor
        """

        # set the parameters of the encoder/decoder using the data_block
        self.set_encoder_decoder_params(data_block)

        # perform the encoding
        output_bits_block = self.encoder_transform.transform(data_block)

        # the final output of encoder needs to be a block of bits
        assert isinstance(output_bits_block, BitsDataBlock)
        return output_bits_block

    def decode(self, data_block: BitsDataBlock):

        # input block to the decoder needs to be a block of bits
        assert isinstance(data_block, BitsDataBlock)
        return self.decoder_transform.transform(data_block)