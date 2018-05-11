# Implement a chatbot
We want to use seq2seq model to implement a chatbot.<br>
## Requirements & their versions：
Python >=3.5<br>
Pytorch 0.3.0<br>
Numpy
## Evaluate our seq2seq model(with attention)：
Run the bash script.
```Bash
bash hw2_seq2seq.sh $1 $2
# $1:path_of_input_file, e.g., ./test_input.txt
# $2:path_of_output_file, e.g., ./output.txt
```
## Train a seq2seq model using original encoder-decoder：
It will save the model to the path: ./model_naive/model_new/2-2_512/
```Bash
python3 hw2_seq2seq_naive.py --training_file=path_of_training_file --loadFilename=path_of_checkpoint_file --n_iterations=a_number
# e.g., python3 hw2_seq2seq_naive.py --training_file=training_input.txt --loadFilename=50000_64.tar --n_iterations=60000
# if you doensn't want to adjust the training iterations or using checkpoint, please feel free to ignore these arguments
```
## Train a seq2seq model using attetional decoder：
It will save the model to the path: ./model_save/model_new/2-2_512/
```Bash
python3 hw2_seq2seq.py --training_file=path_of_training_file --loadFilename=path_of_checkpoint_file --n_iterations=a_number
# e.g., python3 hw2_seq2seq.py --training_file=training_input.txt --loadFilename=120000_64.tar --n_iterations=130000
# if you doensn't want to adjust the training iterations or using checkpoint, please feel free to ignore these arguments
```
## Evaluate our seq2seq model(without attention)：
```Bash
wget 'https://www.dropbox.com/s/z79d5am3kss9wec/50000_64.tar?dl=1' -O 50000_64.tar
python3 hw2_seq2seq_naive.py --EVAL --input_file=path_of_input_file --output_file=path_of_output_file --loadFilename=./50000_64.tar
```
## Reference
We refer to the language translation model, [seq2seq-translation-batched](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb), built by spro to do our work.<br>
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) Luong et al., 2015



