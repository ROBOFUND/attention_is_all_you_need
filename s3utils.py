# -*- coding: utf-8 -*-
#

import boto3
import io

s3 = boto3.resource('s3')
bucket = s3.Bucket('enc-report-data')

source_fname = "wakati/summarize-source.txt"
target_fname = "wakati/summarize-target.txt"

def get_s3file(fname):
    obj = bucket.Object(fname)
    resp = obj.get()
    body = resp['Body'].read()
    ret = io.StringIO(body.decode('utf-8'))
    return ret

def put_s3file(fname, buf):
    obj = bucket.Object(fname)
    resp = obj.put(
        Body=buf,
        ContentEncoding='utf-8',
        ContentType='text/plain')
    
def open_buf():
    buf = io.StringIO()
    return buf

def write_obj(fname, buf):
    obj = bucket.Object(fname)
    resp = obj.put(
        Body=buf.getvalue(),
        ContentEncoding='utf-8',
        ContentType='text/plain')
    return

def read_obj(fname):
    obj = bucket.Object(fname)
    resp = obj.get()
    body = resp['Body'].read()
    ret = io.StringIO(body.decode('utf-8'))
    return ret
