# Implement a seq2seq model
## Evaluate our seq2seq model(with attention)：
Run the bash script.
```Bash
bash hw2_seq2seq.sh $1 $2
# $1:input_file, e.g., test_input.txt
# $2:output_file, e.g., output.txt
```
## Train a seq2seq model using original encoder-decoder：
```Bash
python3 hw2_seq2seq_naive.py --training_file=path_of_training_file --loadFilename=path_of_checkpoint_file --n_iterations=a_number
# e.g., python3 hw2_seq2seq_naive.py --training_file='training_input.txt' --loadFilename='model.tar' --n_iterations=50000
# if you doensn't want to adjust the training iterations or using checkpoint, please feel free to ignore these arguments
```
## Train a seq2seq model using attetional decoder：
```Bash
python3 hw2_seq2seq.py --training_file=path_of_training_file --loadFilename=path_of_checkpoint_file --n_iterations=a_number
# e.g., python3 hw2_seq2seq.py --training_file='training_input.txt' --loadFilename='model.tar' --n_iterations=50000
# if you doensn't want to adjust the training iterations or using checkpoint, please feel free to ignore these arguments
```
## Evaluate our seq2seq model(without attention)：
Run the bash script.
```Bash
bash hw2_seq2seq.sh $1 $2
# $1:input_file, e.g., test_input.txt
# $2:output_file, e.g., output.txt
```



