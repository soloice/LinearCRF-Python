import codecs
import numpy as np
from scipy.optimize import minimize
import time
from sklearn.svm import SVC
import random
import os


# count the most probable tag for each character
class Greedy:
    def __init__(self):
        self.model = SVC()
        self.n_unique_char = -1
        self.char_count = {}
        self.char_index, self.index_char = {}, {}
        self.train_size, self.test_size = -1, -1
        self.n_tags = 6
        self.label_index = {'S': 0, 'B': 1, 'E': 2, 'M': 3, 'M1': 4, 'M2': 5}
        self.index_label = {0: 'S', 1: 'B', 2: 'E', 3: 'M', 4: 'M1', 5: 'M2'}
        self.sentences, self.labels = [], []
        self.start_symbol, self.end_symbol = '<BOS>', '<EOS>'
        self.start_state, self.end_state = 'S', 'S'
        self.char_count[self.start_symbol] = 0.0
        self.char_count[self.end_symbol] = 0.0
        self.most_count = []

    def load_training_data(self, file_name):
        self.sentences, self.labels = [], []
        fh = codecs.open(file_name, 'r', encoding='utf-8')
        sentence, label = [self.start_symbol], [self.start_state]
        for line in fh.readlines():
            if len(line) < 3:  # an empty line
                sentence.append(self.end_symbol)
                label.append(self.end_state)
                self.char_count[self.start_symbol] += 1.0
                self.char_count[self.end_symbol] += 1.0
                self.sentences.append(sentence)
                self.labels.append(label)
                sentence, label = [self.start_symbol], [self.start_state]
            else:
                char, pos, tag = line.split()
                sentence.append(char)
                label.append(tag)
                if char in self.char_count:
                    self.char_count[char] += 1.0
                else:
                    self.char_count[char] = 1.0
        fh.close()

        self.n_unique_char = len(self.char_count)
        print '# of unique characters:', self.n_unique_char
        print 'max(len):', max(map(len, self.sentences))
        print '# of training examples:', len(self.sentences), len(self.labels)
        self.train_size = len(self.labels)

        idx = 0
        for k in self.char_count.keys():
            self.char_index[k] = idx
            self.index_char[idx] = k
            idx += 1

        print ' '.join(self.sentences[0]), ' '.join(self.labels[0])
        self.sentences = map(lambda x: [self.char_index[c] for c in x], self.sentences)
        self.labels = map(lambda y: [self.label_index[yt] for yt in y], self.labels)
        print self.sentences[0], self.labels[0]

        print 'train_size:', self.train_size

    def load_testing_data(self, file_name):
        fh = codecs.open(file_name, 'r', encoding='utf-8')
        sentence, label = [self.start_symbol], [self.start_state]
        self.sentences, self.labels = [], []
        for line in fh.readlines():
            if len(line) < 3:  # an empty line
                sentence.append(self.end_symbol)
                label.append(self.end_state)
                self.sentences.append(sentence)
                self.labels.append(label)
                sentence, label = [self.start_symbol], [self.start_state]
            else:
                char, pos, tag = line.split()
                sentence.append(char)
                label.append(tag)
        fh.close()

        print 'max(len):', max(map(len, self.sentences))
        print '# of testing examples:', len(self.sentences), len(self.labels)
        self.test_size = len(self.labels)

        print ' '.join(self.sentences[0]), ' '.join(self.labels[0])
        # A more canonical way should be introduce a <UNK> symbol rather than randomly return a word id.
        self.sentences = map(lambda x: [self.char_index.get(c, random.randint(0, self.n_unique_char))
                                        for c in x], self.sentences)
        self.labels = map(lambda y: [self.label_index[yt] for yt in y], self.labels)
        print self.sentences[0], self.labels[0]
        print 'test_size:', self.test_size

    def sequence_to_triplet(self):
        self.most_count = np.zeros((self.n_unique_char, self.n_tags))
        for sentence, label in zip(self.sentences, self.labels):
            len_i = len(sentence) - 2
            for i in xrange(1, len_i+1):
                self.most_count[sentence[i], label[i]] += 1.0
        self.most_count = self.most_count.argmax(axis=1)
        return 0

    def fit(self):
        self.sequence_to_triplet()

    def predict(self):
        res_flat = []
        for sentence, label in zip(self.sentences, self.labels):
            len_i = len(sentence) - 2
            for i in xrange(1, len_i+1):
                res_flat.append(self.most_count[sentence[i]])
        file_name1, file_name2 = '../data/6.test.data', '../data/6.test.rst2'
        fh1 = codecs.open(file_name1, 'r', encoding='utf-8')
        fh2 = codecs.open(file_name2, 'w', encoding='utf-8')
        cnt = 0
        for line in fh1.readlines():
            if len(line) < 3:  # an empty line
                fh2.write('\n')
            else:
                fh2.write(line.strip() + ' ' + self.index_label[res_flat[cnt]] + '\n')
                cnt += 1
        fh1.close()
        fh2.close()
        return res_flat


clf = Greedy()
t1 = time.time()
clf.load_training_data('../data/6.train.data')
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'

t1 = time.time()
clf.fit()
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'

t1 = time.time()
clf.load_testing_data('../data/6.test.data')
pred = clf.predict()
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'

os.system('python crf_eval.py ../data/6.test.rst2')
