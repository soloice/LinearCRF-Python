#!/bin/sh
echo 'Begin training'
echo `date`
/home/soloice/Downloads/CRF++-0.58/crf_learn -f 3 -c 4.0 ../data/template ../data/6.train.data ../data/6.model > ../data/6.train.rst
echo 'Finish training'
echo `date`
/home/soloice/Downloads/CRF++-0.58/crf_test -m ../data/6.model ../data/6.test.data > ../data/6.test.rst
echo 'Finish testing'
echo `date`
python crf_eval.py ../data/6.test.rst
