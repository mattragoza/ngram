'usage: python ngram.py <1|2|2s|3|3s> <train_file> <dev_file> <test_file>'

import sys
import nltk
from numpy import log2


class UsageError(Exception):
    def __init__(self, msg):
        self.msg = msg


class NGramModel:

    def __init__(self, train_corpus, N, interp=False, smooth=False):

        if N <= 0:
            raise ValueError('N must be a positive integer')
        self.N = N

        # can interpolate between models from N to 1
        if interp:
            self.tries = [NGramTrie() for i in range(N)]
            self.lambdas = [1/N for i in range(N)] # TODO learn lambda parameters

        # or use a single model
        else:
            self.tries = [NGramTrie()]
            self.lambdas = [1]

        # count ngrams in train corpus by adding to tries
        for i, trie in enumerate(self.tries):
            for sentence in train_corpus:
                for ngram in ngramized(sentence, self.N-i):
                    trie.add(ngram)

    def train_lambda(self, dev_corpus):
        pass #TODO

    def mle(self, ngram):
        if len(ngram) != self.N:
            raise ValueError('ngram must be of length ' + str(self.N))
        mle = []
        for i, trie in enumerate(self.tries):
            mle.append(self.lambdas[i] * trie.mle(ngram[i:]))
        return sum(mle)

    def entropy(self, sentence):

        h_s, len_s = 0, 0
        for ngram in ngramized(sentence, self.N):
            mle = self.mle(ngram)
            if mle > 0:
                h_s += -log2(mle)
            else:
                h_s = float('inf')
            len_s += 1
        if len_s != 0:
            return h_s/len_s
        else:
            return float('inf')

    def perplexity(self, sentence):
        return 2**self.entropy(sentence)

    def __str__(self):
        return '\n'.join(str(trie) for trie in self.tries)


class NGramTrie:

    def __init__(self):
        self.n = 0 # num instances
        self.c = {} # next words

    def __contains__(self, word):
        return word in self.c

    def __getitem__(self, word):
        return self.c[word]

    def __setitem__(self, word, value):
        self.c[word] = value

    def keys(self):
        return list(self.c.keys())

    def add(self, ngram):
        self.n += 1
        if not ngram: 
            return
        word = ngram[0]
        if word not in self:
            self[word] = NGramTrie()
        self[word].add(ngram[1:])

    def count(self, ngram):
        if not ngram:
            return self.n
        word = ngram[0]
        #if len(ngram) == 1:
        #    return self[word].n
        return self[word].count(ngram[1:]) 

    def mle(self, ngram):
        trie = self
        for word in ngram:
            given_c = trie.n
            if word in trie:
                trie = trie[word]
            elif '<unk>' in trie:
                trie = trie['<unk>']
            else:
                return 0
            curr_c = trie.n
        return curr_c/given_c

    def __str__(self):
        return self._str()

    def __repr__(self):
        return 'NGramTrie()'

    def _str(self, tabs=0):
        tab = '    '
        s = 'n=' + str(self.n) + ', c={'
        if self.c: s += '\n'
        for word in self.c:
            s += tab*(tabs+1) + '\'' + str(word) + '\': '
            s += self.c[word]._str(tabs+1)
        if self.c: s += tab*tabs
        return s + '}\n'


def ngramized(sentence, N):
    ngram = ['<s>'] * (N-1)
    for word in sentence[1:]:
        ngram.append(word)
        yield ngram
        ngram.pop(0)


def combine_unknowns(corpus, K=1):

    special = ['<s>', '</s>', '<unk>']
    counts = {}
    for sentence in corpus:
        for word in sentence:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    unk = {w:c for w,c in counts.items() \
        if c <= K and w not in special}
    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word in unk:
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

    combine_unknowns(train_corpus, K=1)

    if mode == '1':
        model = NGramModel(train_corpus, N=1)
    elif mode == '2':
        model = NGramModel(train_corpus, N=2)
    elif mode == '2s':
        model = NGramModel(train_corpus, N=2, interp=True)
        model.train_lambda(dev_corpus)
    elif mode == '3':
        model = NGramModel(train_corpus, N=3)
    elif mode == '3s':
        model = NGramModel(train_corpus, N=3, interp=True, smooth=True)
        model.train_lambda(dev_corpus)

    PP = [model.perplexity(s) for s in test_corpus]
    print('\n'.join(map(str, PP)))


if __name__ == '__main__':
    sys.exit(main())