
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
- Run unit test codes:
    ```
    python3 compressors/LZSS.py
    ```

## Using the codes
- Run evaluation:
    ```
    TODO
    ```
- Compress file:
    ```
    python3 compressors/LZSS.py TODO
    ```

## Report and Presentation links
[Report](https://google.com)
[Presentation Slides](https://docs.google.com/presentation/d/1IvNpNxeBvL9jRCT4w7LkT2Osie4XTMWMNh6xIjWzPNQ/edit?usp=sharing) (Need Stanford account access)

## Contact
The best way to contact the maintainers is to file an issue with your question.
If not please use the following email:
- Chendi Wu: chendiw@stanford.edu
- Xueyang Song: jamiesxy@stanford.edu
