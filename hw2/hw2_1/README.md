# S2VT
We want to use seq2seq model to solve S2VT problem.<br>

## Requirements & their versions：
Python >=3.5<br>
Pytorch 0.3.0<br>
h5py (only for training)<br>
Numpy

## Evaluate the model：
Run the bash script.
```Bash
bash hw2_seq2seq.sh $1 $2
# $1:path_of_input_folder, e.g., ./testing
# $2:path_of_output_file, e.g., ./output.txt
```
## Train a model：
It will save the checkpoint to the path: ./ckpt
```Bash
# Before training, you have to give it training data folder path and testing data folder in line13/line14
python3 model_seq2seq.py 

# You can set dimension of GRU hidden layer in line114
# You can set dimension of word embedding in line115
# You can set training epochs in line116
# You can set teacher forcing ratio in line84
```
After training, you can give generateCaption.py a new checkpoint in line70 and run the evaluation script to get result

## Reference：
S. Venugopalan, M. Rohrbach, J. Donahue, R. J. Mooney,T. Darrell, and K. Saenko. Sequence to sequence - video to text<br>
http://cs231n.stanford.edu/reports/2017/pdfs/31.pdf<br>
https://github.com/Blues5/video-caption-pytorch


