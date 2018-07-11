#!/bin/sh

SOURCE=wakati/filtered-200-summarize-source.txt
TARGET=wakati/filtered-200-summarize-target.txt
EVAL_SOURCE=wakati/filtered-200-eval-summarize-source.txt
EVAL_TARGET=wakati/filtered-200-eval-summarize-target.txt

PYTHONIOENCODING=utf-8
export PYTHONIOENCODING

python -u train.py -g 0 -o result-sum-f200 -i '' \
       -s $SOURCE -t $TARGET -b 24 \
       --source-valid $EVAL_SOURCE --target-valid $EVAL_TARGET \
       2>&1 | tee log-sum-f200.txt
