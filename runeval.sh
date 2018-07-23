#!/bin/sh

SOURCE=wakati/filtered-200-summarize-source.txt
TARGET=wakati/filtered-200-summarize-target.txt
MODEL=result-sum-f200/best_model.npz

PYTHONIOENCODING=utf-8
export PYTHONIOENCODING

python -u eval.py -g -1 -i '' \
       -s $SOURCE -t $TARGET --model $MODEL
