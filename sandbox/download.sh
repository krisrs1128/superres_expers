#!/usr/bin/env bash

cd /home/
wget https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip
mkdir data
mv probav_data.zip data
unzip data/probav_data.zip
rm data/probav_data.zip
