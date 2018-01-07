#!/bin/bash
wget -O Model.zip https://www.dropbox.com/sh/m83ia42k8lfi2da/AADBIx_3xQzhXP9wy8jLaoeoa?dl=1
unzip Model.zip
python3 train.py $1
 