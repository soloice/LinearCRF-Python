import numpy as np
from itertools import product


class Inference:
    def __init__(self, n_time_step=4, n_tags=3, random_state=1234):
        np.random.seed(random_state)
        self.n_time_step = n_time_step
        self.n_tags = n_tags
        self.start_state_id = 0
        # self.potential[t, i, j] = \psi_t(y_t=i, y_{t-1}=j, x)
        self.potential = np.random.random((self.n_time_step+1, self.n_tags, self.n_tags))
        # print self.potential

    def calc_potential(self, y):
        res = 1.0
        for t in xrange(1, self.n_time_step+1):
            res *= self.potential[t, y[t], y[t-1]]
        return res

    def calc_log_potential(self, y):
        res = 0.0
        for t in xrange(1, self.n_time_step+1):
            res += np.log(self.potential[t, y[t], y[t-1]])
        return res

    def calc_cll(self, y, method='brute_force', marginal_pos=2):
        assert len(y) == self.n_time_step + 1
        if method == 'brute_force':
            self.infer_brute_force(y, marginal_pos)
        elif method == 'viterbi':
            self.infer_viterbi(y, marginal_pos)
        elif method == 'viterbi_log':
            self.infer_viterbi_log_domain(y, marginal_pos)
        return 0

    def infer_brute_force(self, y, marginal_pos=2):
        z = 0.0
        margin_sum = 0.0
        y_star, y_star_potential = [self.start_state_id] * (self.n_time_step + 1), 0.0
        for y_ in product(range(self.n_tags), repeat=self.n_time_step):
            # print y_
            # Assume sequences always start with state 0.
            seq_y = [self.start_state_id] + list(y_)
            score_y_ = self.calc_potential(seq_y)
            if seq_y[marginal_pos] == y[marginal_pos] and seq_y[marginal_pos-1] == y[marginal_pos-1]:
                margin_sum += score_y_
            if score_y_ > y_star_potential:
                y_star_potential = score_y_
                y_star = seq_y
            z += score_y_
        pot = self.calc_potential(y)
        cll = np.log(pot / z)
        print 'Potential(y):', pot
        print 'Partition function:', z
        print 'CLL(y):', cll
        print 'y*:', y_star
        print 'marginal p:', margin_sum / z
        return cll, y_star

    def infer_viterbi(self, y, marginal_pos=2):
        alpha = np.zeros((self.n_time_step + 1, self.n_tags))
        beta = np.zeros((self.n_time_step + 1, self.n_tags))
        delta = np.zeros((self.n_time_step + 1, self.n_tags))
        pre = np.zeros((self.n_time_step + 1, self.n_tags), dtype='int')

        delta[1] = self.potential[1, :, self.start_state_id]
        pre[1] = self.start_state_id  # should be redundant
        for t in xrange(2, self.n_time_step+1):
            for j in xrange(self.n_tags):
                best = np.argmax(self.potential[t, j, :] * delta[t-1])
                pre[t, j] = best
                delta[t, j] = self.potential[t, j, best] * delta[t-1, best]

        # print 'pre:', pre
        # print 'delta:', delta
        y_star = [self.start_state_id] * (self.n_time_step + 1)
        y_star[self.n_time_step] = np.argmax(delta[self.n_time_step])
        for t in xrange(self.n_time_step - 1, 0, -1):
            y_star[t] = pre[t+1, y_star[t+1]]

        alpha[1] = self.potential[1, :, self.start_state_id]
        for t in xrange(2, self.n_time_step+1):
            for j in xrange(self.n_tags):
                alpha[t, j] = np.dot(self.potential[t, j, :], alpha[t-1])
        z = alpha[self.n_time_step].sum()
        print 'z = sum(alpha[T]) =', z

        beta[self.n_time_step] = 1.0
        for t in xrange(self.n_time_step-1, 0, -1):
            for i in xrange(self.n_tags):
                beta[t, i] = np.dot(self.potential[t+1, :, i], beta[t+1])
        beta0 = np.dot(self.potential[1, :, self.start_state_id], beta[1])
        z = beta0
        print 'z = beta0 =', beta0

        pot = self.calc_potential(y)
        cll = np.log(pot / z)
        print 'Potential(y):', pot
        print 'Partition function:', z
        print 'CLL(y):', cll
        print 'y*:', y_star
        t = marginal_pos
        print 'marginal p:', alpha[t-1, y[t-1]] * self.potential[t, y[t], y[t-1]] * beta[t, y[t]] / z
        return cll, y_star

    def infer_viterbi_log_domain(self, y, marginal_pos=2):
        def log_sum_exp(arr):
            # For numerically stability
            max_value = np.max(arr)
            return max_value + np.log(np.sum(np.exp(arr - max_value)))

        log_potential = np.log(self.potential)
        log_alpha = np.zeros((self.n_time_step + 1, self.n_tags))
        log_beta = np.zeros((self.n_time_step + 1, self.n_tags))

        log_delta = np.zeros((self.n_time_step + 1, self.n_tags))
        pre = np.zeros((self.n_time_step + 1, self.n_tags), dtype='int')

        log_delta[1] = log_potential[1, :, self.start_state_id]
        pre[1] = self.start_state_id  # should be redundant
        for t in xrange(2, self.n_time_step+1):
            for j in xrange(self.n_tags):
                best = np.argmax(log_potential[t, j, :] + log_delta[t-1])
                pre[t, j] = best
                log_delta[t, j] = log_potential[t, j, best] + log_delta[t-1, best]

        y_star = [self.start_state_id] * (self.n_time_step + 1)
        y_star[self.n_time_step] = np.argmax(log_delta[self.n_time_step])
        for t in xrange(self.n_time_step - 1, 0, -1):
            y_star[t] = pre[t+1, y_star[t+1]]

        log_alpha[1] = log_potential[1, :, self.start_state_id]
        for t in xrange(2, self.n_time_step+1):
            for j in xrange(self.n_tags):
                log_alpha[t, j] = log_sum_exp(log_potential[t, j, :] + log_alpha[t-1])
        log_z = log_sum_exp(log_alpha[self.n_time_step])
        print 'log(z) = log_sum_exp(log_alpha[T]) =', log_z
        print 'For validation, z =', np.exp(log_z)

        log_beta[self.n_time_step] = 0.0
        for t in xrange(self.n_time_step-1, 0, -1):
            for i in xrange(self.n_tags):
                log_beta[t, i] = log_sum_exp(log_potential[t+1, :, i] + log_beta[t+1])
        log_beta0 = log_sum_exp(log_potential[1, :, self.start_state_id] + log_beta[1])
        log_z = log_beta0
        print 'log(z) = log(beta0) =', log_z
        print 'For validation, z =', np.exp(log_z)

        log_pot = self.calc_log_potential(y)
        cll = log_pot - log_z
        print 'Log Potential(y):', log_pot, np.exp(log_pot)
        print 'Log Partition function:', log_z, np.exp(log_z)
        print 'CLL(y):', cll
        print 'y*:', y_star
        t = marginal_pos
        mar_p = log_alpha[t-1, y[t-1]] + log_potential[t, y[t], y[t-1]] + log_beta[t, y[t]] - log_z
        print 'log marginal p:', mar_p, np.exp(mar_p)
        return cll, y_star


inf = Inference(n_time_step=7, random_state=172)
# Padding: always starts with state 0.
y = [inf.start_state_id, 1, 2, 2, 0, 1, 1, 0]
# print inf.calc_potential(y)
print 'brute-force:'
inf.calc_cll(y, method='brute_force')
print '\n\nViterbi:'
inf.calc_cll(y, method='viterbi')
print '\n\nViterbi in log domain:'
inf.calc_cll(y, method='viterbi_log')
