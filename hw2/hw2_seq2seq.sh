#!/bin/bash
wget -O Model.zip https://www.dropbox.com/sh/2nwaafwj28nowz3/AAAtib0YZYa9gh4m58KmqU9Ka?dl=1
unzip Model.zip
python3 final_test.py $1 $2 $3