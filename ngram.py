'usage: python ngram.py <1|2|2s|3|3s> <train_file> <dev_file> <test_file>'

import sys
import nltk
from numpy import log2


class UsageError(Exception):
    pass


class NGramModel:

    @staticmethod
    def build_ngram_trie(corpus, N):

        if N <= 0:
            raise IndexError('N must be greater than 0')
        trie = WordTrie()
        for sentence in corpus:
            ngram = ['<s>'] * (N-1)
            for word in sentence[1:]:
                ngram.append(word)
                trie.add(ngram)
                ngram.pop(0)
        return trie

    @staticmethod
    def ngram_trie_mle(trie, ngram):

        for word in ngram:
            given_c = trie.n
            if word in trie:
                trie = trie[word]
            elif '<UNK>' in trie:
                trie = trie['<UNK>']
            else:
                return 0
            curr_c = trie.n
        return curr_c/given_c

    def __init__(self, train_corpus, N, interp=False, K=1):
        
        # N is number of words to consider per N-gram
        self.N = N

        # can interpolate between models from N to 1
        self.interp = interp
        if interp:
            self.lambdas = [1/N for i in range(N)]
            self.tries = [self.build_ngram_trie(train_corpus, N-i) for i in range(N)]
            ugram_trie = self.tries[-1]

        # or use a single model
        else:
            self.lambdas = [1]
            self.tries = [self.build_ngram_trie(train_corpus, N)]
            ugram_trie = self.tries[0] if N == 1 else None

        # unigram model can use <UNK> for out-of-vocabulary handling
        if ugram_trie:
            ugram_trie.combine_unknowns(K)

    def mle(self, ngram):

        mle = []
        for i, (lam, trie) in enumerate(zip(self.lambdas, self.tries)):
            mle.append(lam * self.ngram_trie_mle(trie, ngram[i:]))
        return sum(mle)

    def perplexity(self, sentence):

        mle_s, len_s = 1, 0
        ngram = ['<s>'] * (self.N-1)
        for word in sentence[1:]:
            ngram.append(word)
            mle_s *= self.mle(ngram)
            len_s += 1
            ngram.pop(0)
        try:
            return -log2(mle_s)/len_s
        except ZeroDivisionError:
            return float('inf')

    def __str__(self):

        return '\n'.join(str(t) for t in self.tries)


class WordTrie:

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

    def add(self, words):
        self.n += 1
        if not words: 
            return
        w = words[0]
        if w not in self:
            self[w] = WordTrie()
        self[w].add(words[1:])

    def count(self, words):
        if not words:
            return self.n
        w = words[0]
        if len(words) == 1:
            return self[w].n
        return self[w].count(words[1:])

    def combine_unknowns(self, K):
        for word in self.keys():
            if self[word].n <= K and word is not '<UNK>':
                if '<UNK>' not in self:
                    self['<UNK>'] = WordTrie()
                self['<UNK>'].n += self.c.pop(word).n  

    def __str__(self):
        return self._str()

    def __repr__(self):
        return 'WordTrie()'

    def _str(self, tabs=0):
        tab = '    '
        s = 'WordTrie(n=' + str(self.n) + ', c={'
        if self.c: s += '\n'
        for w in self.c:
            s += tab*(tabs+1) + str(w) + ': '
            s += self.c[w]._str(tabs+1)
        if self.c: s += tab*tabs
        return s + '})\n'


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

    if len(argv) < 5:
        raise UsageError()

    if argv[1] not in ['1', '2', '2s', '3', '3s']:
        raise UsageError()

    return argv[1:5]


def main(argv=sys.argv):

    try:
        mode, train_file, dev_file, test_file = parse_args(argv)
    except UsageError:
        return __doc__

    train_corpus = read_unprocessed_text_file(train_file)
    dev_corpus   = read_processed_text_file(dev_file)
    test_corpus  = read_processed_text_file(test_file)

    if mode == '1':
        model = NGramModel(train_corpus, N=1)

    elif mode == '2':
        model = NGramModel(train_corpus, N=2)

    elif mode == '2s':
        model = NGramModel(train_corpus, N=2, interp=True)

    elif mode == '3':
        model = NGramModel(train_corpus, N=3)

    elif mode == '3s':
        return 'TODO'

    print(model, file=sys.stderr)

    PP = [model.perplexity(s) for s in test_corpus]
    for i, sentence in enumerate(test_corpus):
        print(' '.join(sentence) + ' ' + str(PP[i]))


if __name__ == '__main__':
    sys.exit(main())