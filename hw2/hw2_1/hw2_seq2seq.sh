#!/bin/bash 
wget -O model_68.pth "https://www.dropbox.com/s/ccsmqgex6n5nclv/model_68.pth?dl=1"
python generateCaption.py $1 $2
