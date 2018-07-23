# -*- coding: utf-8 -*-
#
# ref: https://qiita.com/ynakayama/items/2cc0b1d3cf1a2da612e4

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import argparse
import chainer

import preprocess
import net

from natto import MeCab
import neologdn

def get_args():
    parser = argparse.ArgumentParser(
        description='transformer web app')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--head', type=int, default=8,
                        help='Number of heads in attention mechanism')
    parser.add_argument('--dropout', '-d', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--source', '-s', type=str,
                        default='wakati/filtered-200-summarize-source.txt',
                        help='Filename of train data for source language')
    parser.add_argument('--target', '-t', type=str,
                        default='wakati/filtered-200-summarize-target.txt',
                        help='Filename of train data for target language')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--use-label-smoothing', action='store_true',
                        help='Use label smoothing for cross entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')
    parser.add_argument('--model', type=str,
                        default='result-sum-f200/best_model.npz',
                        help='Vocabulary size of target language')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    app = Flask(__name__)

    source_vocab = ['<eos>', '<unk>', '<bos>'] + \
        preprocess.count_words(args.source, args.source_vocab)
    source_data = preprocess.make_dataset(args.source, source_vocab)
    target_vocab = ['<eos>', '<unk>', '<bos>'] + \
        preprocess.count_words(args.target, args.target_vocab)
    target_data = preprocess.make_dataset(args.target, target_vocab)
    
    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}
    
    model = net.Transformer(
        args.layer,
        min(len(source_ids), len(source_words)),
        min(len(target_ids), len(target_words)),
        args.unit,
        h=args.head,
        dropout=args.dropout,
        max_length=500,
        use_label_smoothing=args.use_label_smoothing,
        embed_position=args.embed_position)

    chainer.serializers.load_npz(args.model, model)

    m = MeCab('-Owakati')

    def picked_up():
        messages = [
            'こんにちは、あなたの名前を入力してください',
            'やあ、お名前はなんですか?',
            'あなたの名前を教えてね'
        ]
        return np.random.choice(messages)

    @app.route('/', methods=['GET', 'POST'])
    def post():
        title = 'こんにちは'
        if request.method == 'GET':
            title = 'ようこそ'
            message = picked_up()
            return render_template('index.html',
                                   message=message, title=title)
        elif request.method == 'POST':
            body = request.form['body']
            body = m.parse(neologdn.normalize(body))
            return render_template('index.html',
                                   body=body, title=title)
        else:
            return redirect(url_for('index'))
    
    app.debug = True
    app.run(host='localhost')

if __name__ == '__main__':
    main()
