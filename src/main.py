import codecs
import numpy as np
from scipy.optimize import minimize
import time
import random
import os


class LinearCRF:
    def __init__(self):
        self.template = 1
        self.char_count = {}
        self.char_index, self.index_char = {}, {}
        self.train_size, self.test_size = -1, -1
        self.n_param = 0
        self.n_tags = 6
        self.n_unique_char = -1
        self.sentences, self.labels = [], []
        self.start_symbol, self.end_symbol = '<BOS>', '<EOS>'
        self.label_index = {'S': 0, 'B': 1, 'E': 2, 'M': 3, 'M1': 4, 'M2': 5}
        self.index_label = {0: 'S', 1: 'B', 2: 'E', 3: 'M', 4: 'M1', 5: 'M2'}
        self.start_state, self.end_state = 'S', 'S'
        self.start_state_id = 0
        self.char_count[self.start_symbol] = 0.0
        self.char_count[self.end_symbol] = 0.0
        self.prior_feature_expectation = np.zeros(self.n_param)
        self.theta = np.zeros(self.n_param)
        self.sigma2 = 10.0  #np.inf  # no Gaussian prior
        self.feature_index, self.index_feature = {}, {}
        self.u_rel_pos_list, self.b_rel_pos_list = [-1, 0, 1], [0]

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

        # Unigram feature
        feature_id = 0
        for rel_pos in self.u_rel_pos_list:
            for word in xrange(self.n_unique_char):
                for tag in xrange(self.n_tags):
                    feature = ('U', rel_pos, word, tag)
                    self.index_feature[feature_id] = feature
                    self.feature_index[feature] = feature_id
                    feature_id += 1

        # Bigram feature
        for rel_pos in self.b_rel_pos_list:
            for word in xrange(self.n_unique_char):
                for tag1 in xrange(self.n_tags):
                    for tag2 in xrange(self.n_tags):
                        feature = ('B', rel_pos, word, tag1, tag2)
                        self.index_feature[feature_id] = feature
                        self.feature_index[feature] = feature_id
                        feature_id += 1

        # State transfer feature
        for tag1 in xrange(self.n_tags):
            for tag2 in xrange(self.n_tags):
                feature = ('T', tag1, tag2)
                self.index_feature[feature_id] = feature
                self.feature_index[feature] = feature_id
                feature_id += 1

        self.n_param = len(self.index_feature)
        self.theta = np.ones(self.n_param) / self.n_param
        print 'self.n_param:', self.n_param

        print ' '.join(self.sentences[0]), ' '.join(self.labels[0])
        self.sentences = map(lambda x: [self.char_index[c] for c in x], self.sentences)
        self.labels = map(lambda y: [self.label_index[yt] for yt in y], self.labels)
        print self.sentences[0], self.labels[0]

        print 'train_size:', self.train_size
        self.get_prior_feature_expectation()

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

    def get_prior_feature_expectation(self):
        self.prior_feature_expectation = np.zeros(self.n_param)
        for i in xrange(self.train_size):
            x, y = self.sentences[i], self.labels[i]
            len_i = len(y) - 2  # actual length of the sentence
            # Exploit sparsity! Many features are zero!
            for t in xrange(1, len_i + 1):
                word, tag1, tag2 = x[t], y[t - 1], y[t]
                u_feature = [self.feature_index[('U', rel_pos, word, tag2)] for rel_pos in self.u_rel_pos_list]
                b_feature = [] if t == 1 else [self.feature_index[('B', rel_pos, word, tag1, tag2)]
                                               for rel_pos in self.b_rel_pos_list]
                t_feature = [] if t == 1 else [self.feature_index[('T', tag1, tag2)]]
                active_features = u_feature + b_feature + t_feature
                self.prior_feature_expectation[active_features] += 1.0
        return self.prior_feature_expectation

    def feature_at(self, k, x, t, yt, yt1):
        # return feature k for an example x at position t,
        # supposing the labels for position t and position t-1 are yt and yt1 respectively.
        # Note: This function is obsoleted when exploiting sparsity structure.
        feature = self.index_feature[k]
        if feature[0] == 'U':
            # unigram feature
            _, rel_pos, word, tag = feature
            if x[t + rel_pos] == word and yt == tag:
                return 1.0
            else:
                return 0.0
        elif feature[0] == 'B':
            # bigram feature
            _, rel_pos, word, tag1, tag2 = feature
            if t > 1 and x[t + rel_pos] == word and yt1 == tag1 and yt == tag2:
                return 1.0
            else:
                return 0.0
        else:
            # state transfer feature
            assert feature[0] == 'T'
            _, tag1, tag2 = feature
            if t > 1 and yt1 == tag1 and yt == tag2:
                return 1.0
            else:
                return 0.0

    def log_potential_at(self, x, t, yt, yt1):
        # return the potential function for an example x,
        # supposing the labels for position t and position t-1 are yt and yt1 respectively.

        # Exploit sparsity: many features are zero.
        word, tag1, tag2 = x[t], yt1, yt
        u_feature = [self.feature_index[('U', rel_pos, word, tag2)] for rel_pos in self.u_rel_pos_list]
        b_feature = [] if t == 1 else [self.feature_index[('B', rel_pos, word, tag1, tag2)]
                                       for rel_pos in self.b_rel_pos_list]
        t_feature = [] if t == 1 else [self.feature_index[('T', tag1, tag2)]]
        active_features = u_feature + b_feature + t_feature
        return self.theta[active_features].sum()

    def calc_log_potential(self, x, y):
        res = 0.0
        len_y = len(y) - 2
        for t in xrange(1, len_y + 1):
            res += self.log_potential_at(x, t, y[t], y[t - 1])
        return res

    def infer_viterbi_log_domain(self, x, y, log_potential=None, return_alpha=False, return_y_star=False):
        """
        Perform inference on example (x, y) using forward-backward algorithm.
        :param x: input sentence (word ids)
        :param y: labels of the corresponding input sentence
        :param log_potential: if potentials are calculated in advance, you can use a cached version.
        :param return_alpha: if you want to return beta array of this input sentence.
                                When calculating cll, this is not necessary.
        :param return_y_star: if you want to calculate the most possible labels of this input sentence.
        :return: log_alpha, log_beta, log_z, and y_star
        """

        def log_sum_exp(arr):
            # For numerically stability
            max_value = np.max(arr)
            return max_value + np.log(np.sum(np.exp(arr - max_value)))

        # len_i: actual length of the sentence
        len_i = len(y) - 2

        if log_potential is None:
            log_potential = np.zeros((len_i + 1, self.n_tags, self.n_tags))
            for t in xrange(len_i + 1):
                for yt in xrange(self.n_tags):
                    for yt1 in xrange(self.n_tags):
                        log_potential[t, yt, yt1] = self.log_potential_at(x, t, yt, yt1)

        if return_y_star:
            log_delta = np.zeros((len_i + 1, self.n_tags))
            pre = np.zeros((len_i + 1, self.n_tags), dtype='int')

            log_delta[1] = log_potential[1, :, self.start_state_id]
            pre[1] = self.start_state_id  # should be redundant
            for t in xrange(2, len_i + 1):
                for j in xrange(self.n_tags):
                    best = np.argmax(log_potential[t, j, :] + log_delta[t - 1])
                    pre[t, j] = best
                    log_delta[t, j] = log_potential[t, j, best] + log_delta[t - 1, best]

            y_star = [self.start_state_id] * (len_i + 1)
            y_star[len_i] = np.argmax(log_delta[len_i])
            for t in xrange(len_i - 1, 0, -1):
                y_star[t] = pre[t + 1, y_star[t + 1]]
            y_star = y_star[1:]
        else:
            y_star = None

        if return_alpha:
            # Forward propagation
            log_alpha = np.zeros((len_i + 1, self.n_tags))
            log_alpha[1] = log_potential[1, :, self.start_state_id]
            for t in xrange(2, len_i + 1):
                for j in xrange(self.n_tags):
                    log_alpha[t, j] = log_sum_exp(log_potential[t, j, :] + log_alpha[t - 1])
        else:
            log_alpha = None

        # Backward propagation
        log_beta = np.zeros((len_i + 1, self.n_tags))
        log_beta[len_i] = 0.0
        for t in xrange(len_i - 1, 0, -1):
            for i in xrange(self.n_tags):
                log_beta[t, i] = log_sum_exp(log_potential[t + 1, :, i] + log_beta[t + 1])
        log_beta0 = log_sum_exp(log_potential[1, :, self.start_state_id] + log_beta[1])
        log_z = log_beta0

        log_pot = self.calc_log_potential(x, y)
        cll = log_pot - log_z
        # print 'Log Partition function:', log_z, np.exp(log_z)
        # print 'CLL(y):', cll
        # print 'y*:', y_star
        return log_alpha, log_beta, log_z, cll, y_star

    def cll(self):
        # compute conditional log likelihood over corpus
        res = 0.0
        for i in xrange(self.train_size):
            # if i % 5000 == 0:
            #     print 'In cll, # of processed examples:', i
            x, y = self.sentences[i], self.labels[i]
            _, _, _, cll, _ = self.infer_viterbi_log_domain(x, y)
            res += cll
        res -= np.dot(self.theta, self.theta) / (2.0 * self.sigma2)
        return res

    def model_expectation_on_example(self, i):
        x, y = self.sentences[i], self.labels[i]
        len_i = len(y) - 2
        log_potential = np.zeros((len_i + 1, self.n_tags, self.n_tags))
        for t in xrange(len_i + 1):
            for yt in xrange(self.n_tags):
                for yt1 in xrange(self.n_tags):
                    log_potential[t, yt, yt1] = self.log_potential_at(x, t, yt, yt1)

        log_alpha, log_beta, log_z, _, _ = self.infer_viterbi_log_domain(x, y, log_potential, return_alpha=True)

        # get p(y_t, y_{t-1} | x_t^({i}))
        p = np.zeros((len_i + 1, self.n_tags, self.n_tags))
        for t in xrange(1, len_i + 1):
            for yt in xrange(self.n_tags):
                for yt1 in xrange(self.n_tags):
                    if t == 1 and yt1 != self.start_state_id:
                        continue
                    p[t, yt, yt1] = log_alpha[t - 1, yt1] + log_potential[t, yt, yt1] + log_beta[t, yt] - log_z
        # assert (p < 1.0).all()
        p = np.exp(p)
        for yt1 in xrange(self.n_tags):
            if yt1 == self.start_state_id:
                continue
            for yt in xrange(self.n_tags):
                p[1, yt, yt1] = 0.0
        del log_alpha, log_beta

        # Exploit sparsity again!
        model_expectation_for_example_i = np.zeros(self.theta.shape)
        for t in xrange(1, len_i + 1):
            word = x[t]
            for tag2 in xrange(self.n_tags):
                for tag1 in xrange(self.n_tags):
                    u_feature = [self.feature_index[('U', rel_pos, word, tag2)] for rel_pos in self.u_rel_pos_list]
                    b_feature = [] if t == 1 else [self.feature_index[('B', rel_pos, word, tag1, tag2)]
                                                   for rel_pos in self.b_rel_pos_list]
                    t_feature = [] if t == 1 else [self.feature_index[('T', tag1, tag2)]]
                    active_features = u_feature + b_feature + t_feature
                    model_expectation_for_example_i[active_features] += p[t, tag2, tag1]
        return model_expectation_for_example_i

    def cll_prime(self):
        # gradient of conditional log likelihood over corpus
        gradient = self.prior_feature_expectation - self.theta / self.sigma2
        for i in xrange(self.train_size):
            # if i % 5000 == 0:
            #     print 'In cll_prime, # of processed examples:', i
            gradient -= self.model_expectation_on_example(i)
        return gradient

    def ncll(self, theta):
        self.theta = theta
        return -self.cll()

    def ncll_prime(self, theta):
        self.theta = theta
        return -self.cll_prime()

    def fit(self):
        res = minimize(self.ncll, self.theta, method='L-BFGS-B',
                       jac=self.ncll_prime, options={'disp': True, 'maxiter': 100})
        if res.success:
            self.theta = res.x
        else:
            print 'Failed to optimize CLL'

    def truncate(self, truncate_size=50):
        # use the first truncate_size examples only
        self.train_size = truncate_size
        self.sentences = crf.sentences[:crf.train_size]
        self.labels = crf.labels[:crf.train_size]

    def predict(self):
        res = []
        for i in xrange(len(self.sentences)):
            x = self.sentences[i]
            y = [self.start_state_id] * len(x)  # dummy y
            _, _, _, _, y_star = self.infer_viterbi_log_domain(x, y, return_y_star=True)
            res.append(map(lambda idx: self.index_label[idx], y_star))
        # print 'res:', res

        # flatten res
        res_flat = reduce(lambda a, b: a + b, res)
        # print res_flat
        file_name1, file_name2 = '../data/6.test.data', '../data/6.test.rst2'
        fh1 = codecs.open(file_name1, 'r', encoding='utf-8')
        fh2 = codecs.open(file_name2, 'w', encoding='utf-8')
        cnt = 0
        for line in fh1.readlines():
            if len(line) < 3:  # an empty line
                fh2.write('\n')
            else:
                fh2.write(line.strip() + ' ' + res_flat[cnt] + '\n')
                cnt += 1
        fh1.close()
        fh2.close()
        return res


crf = LinearCRF()
t1 = time.time()
crf.load_training_data('../data/6.train.data')
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'

crf.truncate(20000)

t1 = time.time()
crf.cll()
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'
# ~133s on full training set

t1 = time.time()
crf.cll_prime()
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'
# ~179s on the first 2w training data

t1 = time.time()
crf.fit()
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'

t1 = time.time()
crf.load_testing_data('../data/6.test.data')
pred = crf.predict()
t2 = time.time()
print 'Time eclipsed:', t2 - t1, ' seconds\n\n'

os.system('python crf_eval.py ../data/6.test.rst2')
