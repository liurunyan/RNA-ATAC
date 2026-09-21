#!/bin/bash

source /data/run01/scxl504/miniconda3/etc/profile.d/conda.sh
conda activate R4

echo "=============================="
echo "Hostname:"
hostname

echo "=============================="
echo "Start:"
date

Rscript preprocess.R

echo "=============================="
echo "End:"
date