#!/bin/bash 
wget -O good_model.pth "https://www.dropbox.com/s/5e9u9tt6gmecq5n/good_model.pth?dl=1"
python generateCaption.py $1 $2
