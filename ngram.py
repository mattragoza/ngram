'usage: python ngram.py <1|2|2s|3|3s> <train_file> <dev_file> <test_file>'

import sys
import nltk
import numpy as np


class UsageError(Exception):
    def __init__(self, msg):
        self.msg = msg


class NGramModel:

    def __init__(self, N, corpus=None, interp=False, smooth=False):

        if N <= 0:
            raise ValueError('N must be a positive integer')
        self.N = N

        # can interpolate between models from N to 1
        if interp:
            self.tries = [NGramTrie(N-i) for i in range(N)]
            self.lambdas = np.ones(N) / N

        # or use a single model
        else:
            self.tries = [NGramTrie(N)]
            self.lambdas = [1]

        self.smooth = smooth

        if corpus:
            self.add_corpus(corpus)

    def add_corpus(self, corpus):

        # count ngrams in corpus by adding to tries
        for i, trie in enumerate(self.tries):
            for sentence in corpus:
                for ngram in ngramized(sentence, self.N-i):
                    trie.add(ngram)

    def optimize_lambdas(self, corpus, sample_size=100):
        
        if len(self.lambdas) == 1:
            return self.lambdas

        elif len(self.lambdas) == 2:
            param_space = []
            for i in range(10):
                l1 = i/10
                l2 = 1 - l1
                param_space.append(np.array([l1, l2]))

        elif len(self.lambdas) == 3:
            param_space = []
            for i in range(10):
                for j in range(10-i):
                    l1 = i/10
                    l2 = j/10
                    l3 = 1 - (l1 + l2)
                    param_space.append(np.array([l1, l2, l3]))

        else:
            raise NotImplementedError()

        opt_lambdas, opt_pp = self.lambdas, float('inf')
        for lambdas in param_space:
            self.lambdas = lambdas
            pp = 0
            for i in range(sample_size):
                sentence = np.random.choice(corpus)
                pp += self.perplexity(sentence)
            pp /= float(sample_size)
            if pp < opt_pp:
                opt_lambdas = lambdas
                opt_pp = pp

        self.lambdas = opt_lambdas
        return self.lambdas

    def pr(self, ngram):

        if len(ngram) != self.N:
            raise ValueError('expected a ' + str(self.N) + '-gram')
        
        pr = np.zeros(len(self.tries))
        for i, trie in enumerate(self.tries):

            given = ngram[i:-1]
            next_ = ngram[i:]

            if self.smooth:

                try:
                    c_next = trie.get(next_).count
                except KeyError:
                    c_next = 0
                try:
                    c_given = trie.get(given).count
                    c_next_types = len(trie.get(given).next)
                    discount = c_next_types / (c_given + c_next_types)
                    if c_next == 0:
                        wb = discount / c_next_types
                    else:
                        wb = (1 - discount) * c_next/float(c_given)
                except KeyError:
                    wb = 0
                pr[i] = wb
            
            else:
                try:
                    c_next = trie.get(next_).count
                    c_given = trie.get(given).count
                    mle = c_next/float(c_given)
                except KeyError:
                    mle = 0
                pr[i] = mle

        return np.sum(self.lambdas * pr)

    def entropy(self, sentence):

        h_s, len_s = 0, 0
        for ngram in ngramized(sentence, self.N):
            pr = self.pr(ngram)
            if pr > 0:
                h_s += -np.log2(pr)
            else:
                h_s += float('inf')
            len_s += 1
        if len_s != 0:
            return h_s / len_s
        else:
            return float('inf')

    def perplexity(self, sentence):
        return 2**self.entropy(sentence)

    def __str__(self):
        return '\n'.join(str(trie) for trie in self.tries)


class NGramTrie:

    def __init__(self, N):
        if N < 0:
            raise ValueError('N must be a non-negative integer')
        self.N = N
        self.count = 0
        self.next = {}

    def add(self, ngram):
        if len(ngram) != self.N:
            raise ValueError('expected a ' + str(self.N) + '-gram')
        self.count += 1
        if not ngram:
            return
        if ngram[0] not in self.next:
            self.next[ngram[0]] = NGramTrie(self.N-1)
        self.next[ngram[0]].add(ngram[1:])

    def get(self, ngram):
        if not ngram:
            return self
        elif ngram[0] in self.next:
            return self.next[ngram[0]].get(ngram[1:])
        else:
            raise KeyError('ngram not found')

    def choose(self):
        r = np.random.random() * self.count
        for word in self.next:
            if r <= self.next[word].count:
                return word
            else:
                r -= self.next[word].count

    def generate(self):
        ngram = ['<s>'] * (self.N-1)
        sentence = ['<s>']
        while True:
            word = self.get(ngram).choose()
            sentence.append(word)
            if word == '</s>':
                break
            ngram.append(word)
            ngram.pop(0)

        return sentence

    def __repr__(self):
        return 'NGramTrie()'

    def __str__(self):
        return self._str()

    def _str(self, tabs=0):
        tab = '    '
        s = 'count=' + str(self.count)
        if self.next: s += '\n'
        for word in self.next:
            s += tab*(tabs+1) + '\'' + str(word) + '\': '
            s += self.next[word]._str(tabs+1)
        if self.next: s += tab*tabs
        return s + '\n'


def ngramized(sentence, N):

    if N <= 0 or N != int(N):
        raise ValueError('N must be a positive integer')
    ngram = ['<s>'] * (N-1)
    for word in sentence[1:]:
        ngram.append(word)
        yield ngram
        ngram.pop(0)


def get_vocab_set(corpus, K=1):

    special = {'<s>', '</s>', '<unk>'}
    counts = {}
    for sentence in corpus:
        for word in sentence:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    return {w for w in counts \
        if counts[w] > K or w in special}


def replace_oov_with_unk(corpus, vocab):

    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word not in vocab:
                sentence[i] = '<unk>'


def read_unprocessed_text_file(text_file):

    corpus = []
    punct = ['.', '!', '?']
    with open(text_file, 'r') as f:
        for line in f:
            for sentence in nltk.sent_tokenize(line):
                words = nltk.word_tokenize(sentence)
                words.insert(0, '<s>')
                words.append('</s>')
                words = [w for w in words if w not in punct]
                corpus.append(words)

    return corpus


def read_processed_text_file(text_file):

    corpus = []
    with open(text_file, 'r') as f:
        for line in f:
            corpus.append(line.split())

    return corpus


def parse_args(argv):

    if '-h' in argv:
        raise UsageError(__doc__)
    if len(argv) < 5:
        raise UsageError('error: not enough arguments')
    if argv[1] not in ['1', '2', '2s', '3', '3s']:
        raise UsageError('error: invalid mode argument')
    return argv[1:5]


def main(argv=sys.argv):

    try:
        mode, train_file, dev_file, test_file = parse_args(argv)
    except UsageError as e:
        return e.msg

    train_corpus = read_unprocessed_text_file(train_file)
    dev_corpus   = read_processed_text_file(dev_file)
    test_corpus  = read_processed_text_file(test_file)

    vocab = get_vocab_set(train_corpus, K=1)
    replace_oov_with_unk(train_corpus, vocab)
    replace_oov_with_unk(dev_corpus, vocab)
    replace_oov_with_unk(test_corpus, vocab)

    if mode == '1':
        model = NGramModel(N=1)
        model.add_corpus(train_corpus)
    elif mode == '2':
        model = NGramModel(N=2)
        model.add_corpus(train_corpus)
    elif mode == '2s':
        model = NGramModel(N=2, interp=True)
        model.add_corpus(train_corpus)
        model.optimize_lambdas(dev_corpus)
    elif mode == '3':
        model = NGramModel(N=3)
        model.add_corpus(train_corpus)
    elif mode == '3s':
        model = NGramModel(N=3, interp=True, smooth=True)
        model.add_corpus(train_corpus)
        model.optimize_lambdas(dev_corpus)

    for sentence in test_corpus:
        print(' '.join(sentence) + ' ' + str(model.perplexity(sentence)))


if __name__ == '__main__':
    sys.exit(main())