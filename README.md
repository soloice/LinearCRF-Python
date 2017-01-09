# LinearCRF-Python
Implemented a Linear Chain CRF to perform Chinese Word Segmentation.

In this project, Chinese Word Segmentation is treated as a sequence-labelling problem, and a 6-label system (`B/E/S/M/M1/M2` system) is used: `S` for single-character words, `BE` for two-character words, `BME` for three-character words, `BM1M2E` for four-character words, and `BM1M2M...ME` for even longer words.

In the `/data` folder, training and testing data are provided. The file `template` specifies feature template used by CRF. The format is the same with a popular crf toolkit: `crf++`.

In the `/src` folder, `test_inference.py` compared several different inference algorithms. To be precise, I tested Brute Force algorithm, Viterbi algorithm, and Viterbi algorithm on the log domain. `crf.py` implemented a full CRF algorithm, and `crf_eval.py` trains and tests it.  The main source file is `main.py`, it is almost the same as `crf_eval.py`, but exploited sparsity to accelerate the algorithm. `greedy.py` is a simple baseline: it assigns to each character the most frequent label of it in the corpus. `exec_crfpp.sh` is used to run the popular crf toolkit `crf++`, and compare the results of my implementation and `crf++`.
