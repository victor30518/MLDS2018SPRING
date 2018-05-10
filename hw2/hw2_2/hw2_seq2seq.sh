#!/bin/bash
wget -O 120000_64.tar https://www.dropbox.com/s/kttbyi66zblee14/120000_64.tar?dl=1
python3 hw2_seq2seq.py --EVAL --input_file=$1 --output_file=$2 --loadFilename=120000_64.tar
