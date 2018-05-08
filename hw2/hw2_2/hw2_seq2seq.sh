wget 'https://www.dropbox.com/s/kttbyi66zblee14/120000_64.tar?dl=1' -O 120000_64.tar
python3 hw2_seq2seq.py --EVAL --input_file=$1 --output_file=$2
