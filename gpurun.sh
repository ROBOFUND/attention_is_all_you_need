#!/bin/sh

SOURCE=wakati/summarize-source.txt
TARGET=wakati/summarize-target.txt

PYTHONIOENCODING=utf-8
export PYTHONIOENCODING

python -u train.py -g 0 -o result-sum -i '' \
       -s $SOURCE -t $TARGET --source-valid $SOURCE --target-valid $TARGET \
       2>&1 | tee log-sum
