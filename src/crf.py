import codecs
import numpy as np
from scipy.optimize import minimize


class LinearCRF:
    def __init__(self):
        self.template = 1
        self.char_count = {}
        self.char_index = {}
        self.train_size = -1
        self.n_param = 0
        self.n_tags = 6
        self.n_unique_char = -1
        self.n_feature_template = 3
        # Currently, using the 3 following features:
        # U00:%x[-1,0]
        # U01:%x[0,0]
        # U02:%x[1,0]
        self.sentences, self.labels = [], []
        self.start_symbol, self.end_symbol = '<BOS>', '<EOS>'
        self.label_index = {'S': 0, 'B': 1, 'E': 2, 'M': 3, 'M1': 4, 'M2': 5}
        self.index_label = {0: 'S', 1: 'B', 2: 'E', 3: 'M', 4: 'M1', 5: 'M2'}
        self.start_state, self.end_state = 'S', 'S'
        self.char_count[self.start_symbol] = 0.0
        self.char_count[self.end_symbol] = 0.0
        self.prior_feature_expectation = np.zeros(self.n_param)
        self.theta = np.zeros(self.n_param)
        self.sigma2 = 100.0
        self.feature_index, self.index_feature = {}, {}

    def load_training_data(self, file_name):
        fh = codecs.open(file_name, 'r', encoding='utf-8')
        sentence, label = [self.start_symbol], [self.start_state]
        for line in fh.readlines():
            if len(line) < 3:   # an empty line
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
            idx += 1

        # 4 kinds of hand-coded feature
        feature_id = 0
        for rel_pos in [-1, 0, 1]:
            for word in xrange(self.n_unique_char):
                for tag in xrange(self.n_tags):
                    feature = ('U', rel_pos, word, tag)
                    self.index_feature[feature_id] = feature
                    self.feature_index[feature] = feature_id
                    feature_id += 1

        for rel_pos in [-1, 0, 1]:
            for word in xrange(self.n_unique_char):
                for tag1 in xrange(self.n_tags):
                    for tag2 in xrange(self.n_tags):
                        feature = ('B', rel_pos, word, tag1, tag2)
                        self.index_feature[feature_id] = feature
                        self.feature_index[feature] = feature_id
                        feature_id += 1

        self.n_param = len(self.index_feature)
        self.theta = np.zeros(self.n_param)
        tmp = self.n_feature_template * self.n_unique_char * self.n_tags
        print 'self.n_param:', self.n_param, '(SHOULD =', tmp + tmp * self.n_tags

        print ' '.join(self.sentences[0]), ' '.join(self.labels[0])
        self.sentences = map(lambda x: [self.char_index[c] for c in x], self.sentences)
        self.labels = map(lambda y: [self.label_index[yt] for yt in y], self.labels)
        print self.sentences[0], self.labels[0]

        self.get_prior_feature_expectation()

    def get_prior_feature_expectation(self):
        self.prior_feature_expectation = np.zeros(self.n_param)
        for i in xrange(self.train_size):
            print i
            x, y = self.sentences[i], self.labels[i]
            len_i = len(y) - 2  # actual length of the sentence
            for k in xrange(self.n_param):
                for t in xrange(1, len_i+1):
                    self.prior_feature_expectation[k] += self.feature_at(k, x, t, y[t], y[t - 1])
        return 0

    def feature_at(self, k, x, t, yt, yt1):
        # return feature k for an example x at position t,
        # supposing the labels for position t and position t-1 are yt and yt1 respectively.
        feature = self.index_feature[k]
        if feature[0] == 'U':
            # unigram feature
            _, rel_pos, word, tag = feature
            if x[t + rel_pos] == word and yt == tag:
                return 1.0
            else:
                return 0.0
        else:
            # bigram feature
            _, rel_pos, word, tag1, tag2 = feature
            if x[t + rel_pos] == word and yt1 == tag1 and yt == tag2:
                return 1.0
            else:
                return 0.0

    def log_potential_at(self, x, t, yt, yt1):
        # return the potential function for an example x,
        # supposing the labels for position t and position t-1 are yt and yt1 respectively.

        # Exploit sparsity: many features are zero.
        # Get active features only.
        word, tag1, tag2 = x[t], yt1, yt
        u_feature = [self.feature_index[('U', rel_pos, word, tag2)] for rel_pos in [-1, 0, 1]]
        b_feature = [self.feature_index[('B', rel_pos, word, tag1, tag2)] for rel_pos in [-1, 0, 1]]
        active_features = u_feature + b_feature
        return self.theta[active_features].sum()

    # def potential_at(self, x, t, yt, yt1):
    #     # return the potential function for an example x,
    #     # supposing the labels for position t and position t-1 are yt and yt1 respectively.
    #     return np.exp(self.log_potential_at(x, t, yt, yt1))
    #     pass

    def calc_log_potential(self, x, y):
        res = 0.0
        len_y = len(y) - 2
        for t in xrange(1, len_y+1):
            res += self.log_potential_at(x, t, y[t], y[t-1])
        return res

    def infer_viterbi_log_domain(self, x, y, log_potential=None, return_y_star=False):
        """
        Perform inference on example (x, y)
        Using forward-backward algorithm
        :param x: input sentence (word ids)
        :param y: labels of the corresponding input sentence
        :param log_potential: if potentials are calculated in advance, you can use a cached version.
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

            log_delta[1] = log_potential[1, :, self.start_state]
            pre[1] = self.start_state  # should be redundant
            for t in xrange(2, len_i+1):
                for j in xrange(self.n_tags):
                    best = np.argmax(log_potential[t, j, :] + log_delta[t-1])
                    pre[t, j] = best
                    log_delta[t, j] = log_potential[t, j, best] + log_delta[t-1, best]

            y_star = [self.start_state] * (len_i + 1)
            y_star[len_i] = np.argmax(log_delta[len_i])
            for t in xrange(len_i - 1, 0, -1):
                y_star[t] = pre[t+1, y_star[t+1]]
            y_star = y_star[1:]
        else:
            y_star = None

        # Forward propagation
        log_alpha = np.zeros((len_i + 1, self.n_tags))
        log_alpha[1] = log_potential[1, :, self.start_state]
        for t in xrange(2, len_i+1):
            for j in xrange(self.n_tags):
                log_alpha[t, j] = log_sum_exp(log_potential[t, j, :] + log_alpha[t-1])
        # log_z = log_sum_exp(log_alpha[len_i])
        # print 'log(z) = log_sum_exp(log_alpha[T]) =', log_z
        # print 'For validation, z =', np.exp(log_z)

        # Backward propagation
        log_beta = np.zeros((len_i + 1, self.n_tags))
        log_beta[len_i] = 0.0
        for t in xrange(len_i-1, 0, -1):
            for i in xrange(self.n_tags):
                log_beta[t, i] = log_sum_exp(log_potential[t+1, :, i] + log_beta[t+1])
        log_beta0 = log_sum_exp(log_potential[1, :, self.start_state] + log_beta[1])
        log_z = log_beta0
        # print 'log(z) = log(beta0) =', log_z
        # print 'For validation, z =', np.exp(log_z)

        log_pot = self.calc_log_potential(x, y)
        cll = log_pot - log_z
        # print 'Log Potential(y):', log_pot, np.exp(log_pot)
        # print 'Log Partition function:', log_z, np.exp(log_z)
        # print 'CLL(y):', cll
        # print 'y*:', y_star

        # t = marginal_pos
        # mar_p = log_alpha[t-1, y[t-1]] + log_potential[t, y[t], y[t-1]] + log_beta[t, y[t]] - log_z
        # print 'log marginal p:', mar_p, np.exp(mar_p)
        return log_alpha, log_beta, log_z, cll, y_star

    def cll(self):
        # compute conditional log likelihood over corpus
        res = 0.0
        for i in xrange(self.train_size):
            x, y = self.sentences[i], self.labels[i]
            _, _, _, cll, _ = self.infer_viterbi_log_domain(x, y)
            res += cll
        res -= np.dot(self.theta, self.theta)/(2.0 * self.sigma2)
        return -res  # return -CLL rather than CLL itself.

    def model_expectation_on_example(self, i):
        x, y = self.sentences[i], self.labels[i]
        len_i = len(y) - 2
        log_potential = np.zeros((len_i + 1, self.n_tags, self.n_tags))
        for t in xrange(len_i + 1):
            for yt in xrange(self.n_tags):
                for yt1 in xrange(self.n_tags):
                    log_potential[t, yt, yt1] = self.log_potential_at(x, t, yt, yt1)

        log_alpha, log_beta, log_z, _, _ = self.infer_viterbi_log_domain(x, y, log_potential)

        # get p(y=y1, y'=y2 | x_t^({i}))
        log_p = np.zeros((len_i+1, self.n_tags, self.n_tags))
        for t in xrange(1, len_i+1):
            for y1 in xrange(self.n_tags):
                for y2 in xrange(self.n_tags):
                    log_p[t, y1, y2] = log_alpha[t-1, y1] + log_potential[t, y2, y1] + log_beta[t, y2] - log_z
        del log_alpha, log_beta

        model_expectation_for_example_i = np.zeros(self.theta.shape)
        for k in xrange(self.n_param):
            # For each feature k, get an array fk of shape (len_i+1, self.n_tags, self.n_tags)
            # By convention, don't use index 0.
            fk = np.zeros((len_i+1, self.n_tags, self.n_tags))
            for t in xrange(1, len_i+1):
                for y1 in xrange(self.n_tags):
                    for y2 in xrange(self.n_tags):
                        fk[t, y1, y2] = self.feature_at(k, x, t, y1, y2)
            # update gradient for theta_k
            model_expectation_for_example_i[k] = np.sum(fk[1:, :, :] * np.exp(log_p[1:, :, :]))
        return model_expectation_for_example_i
    
    def cll_prime(self):
        # gradient of conditional log likelihood over corpus
        gradient = self.prior_feature_expectation - self.theta / self.sigma2
        for i in xrange(self.train_size):
            gradient -= self.model_expectation_on_example(i)
        return -gradient  # because we want to maximize CLL, i.e. minimize -CLL

    def fit(self):
        res = minimize(self.cll, self.theta, method='BFGS', jac=self.cll_prime)
        if res.success:
            self.theta = res.x
        else:
            print 'Failed to optimize CLL'

    def predict(self):
        res = []
        for i in xrange(len(self.sentences)):
            x = self.sentences[i]
            y = [self.start_state] * len(x)  # dummy y
            _, _, _, _, y_star = self.infer_viterbi_log_domain(x, y, return_y_star=True)
            res.append(map(lambda idx: self.index_label[idx], y_star))
        return res

crf = LinearCRF()
crf.load_training_data('../data/6.train.data')
