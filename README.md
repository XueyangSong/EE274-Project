
# Stanford Compression Library -- Optimized LZ77
The goal of the repository is the class project for [EE274: Data Compression course, Fall 22](https://stanforddatacompressionclass.github.io/Fall22/) at Stanford University. The goal for the project is to implement optimized LZ77, LZSS, using different optimization algorithms.

## Compression algorithms
Here is a list of algorithms used.
- [Huffman codes](compressors/huffman_coder.py)
- [Elias Delta code](compressors/elias_delta_uint_coder.py)

Here is a list of algorithms implemented.
- [LZSS](compressors/LZSS.py)
- [Finite State Entropy code](compressors/LZSS.py)

## Getting started
- Create conda environment and install required packages:
    ```
    conda create --name myenv python=3.8.2
    conda activate myenv
    python -m pip install -r requirements.txt
    ```
- Add path to the repo to `PYTHONPATH`:
    ```
    export PYTHONPATH=$PYTHONPATH:<path_to_repo>
    ```

## Running the code
- Run unit test codes: (Note it takes a long time)
    ```
    python3 compressors/LZSS.py
    ```
- Compress file test:
    ```
    python3 compressors/LZSS.py -i <*input_file*> [-t <*table_type*>] [-m <*find_match_method*>] [-b <*binary_type*>] [-g <*greedy_optimal*>]
    ```
The default table_type is "shortest". Default find_match_method is "hashchain". Default binary_type is "optimized". Default greedy_optimal is "greedy". Also note that the input file need to be in directory test/.
The command makes sure that the decoding of the encoded file is the same as the original file.

## Evaluation
- Codes used for evaluation:
[eval_results-checkpoint.ipynb](test/.ipynb_checkpoints)
- Evaluation results:
[result](test/result)

## Report and Presentation links
[Report](https://github.com/chendiw/stanford_compression_library_lz77/blob/main/project_report.pdf)

[Presentation Slides](https://docs.google.com/presentation/d/1IvNpNxeBvL9jRCT4w7LkT2Osie4XTMWMNh6xIjWzPNQ/edit?usp=sharing) (Need Stanford account access)

## Contact
The best way to contact the maintainers is to file an issue with your question.
If not please use the following email:
- Chendi Wu: chendiw@stanford.edu
- Xueyang Song: jamiesxy@stanford.edu
