'usage: python ngram.py <1|2|2s|3|3s> <train_file> <dev_file> <test_file>'

import sys
import nltk


class UsageError(Exception):
    pass


class NGramModel:

    @staticmethod
    def build_ngram_trie(corpus, N):

        trie = WordTrie()
        for sentence in corpus:
            ngram = ['<s>'] * (N-1)
            for word in sentence[1:]:
                ngram.append(word)
                trie.add(ngram)
                ngram.pop(0)

        return trie

    def __init__(self, train_corpus, N, interp=False, K=1):
        
        self.N = N

        # can interpolate between models from 1 to N
        if interp:
            self.lambdas = [1/N for n in range(N)]
            self.tries = [self.build_ngram_trie(train_corpus, n+1) for n in range(N)]
            unigram = self.tries[0]

        # or use a single model
        else:
            self.trie = self.build_ngram_trie(train_corpus, N)
            unigram = self.trie if N == 1 else None

        # unigram can use <UNK> for out-of-vocabulary handling
        if unigram:
            unk = WordTrie()
            for word in unigram.words():
                count = unigram[word].n
                if count <= K:
                    unk.n += count
                    unigram.pop(word)
            unigram['<UNK>'] = unk

    def MLE(self, ngram):

        curr, parent_n = self.trie, 0
        for i in range(self.N):
            word = ngram[i]
            try:
                parent_n = curr.n
                curr = curr[word]
            except KeyError:
                if self.N == 1:
                    curr = curr['<UNK>']
                else:
                    return 0
        return curr.n/parent_n

    def perplexity(self, sentence):

        mle, s = 1, 0
        ngram = ['<s>'] * (self.N-1)
        for word in sentence[1:]:
            ngram.append(word)
            mle *= self.MLE(ngram)
            s += 1
            ngram.pop(0)
        try:
            return mle**(-1/s)
        except ZeroDivisionError:
            return float('inf')


class WordTrie:

    def __init__(self):
        self.n = 0 # num instances
        self.c = {} # next words

    def __contains__(self, key):
        return key in self.c

    def __getitem__(self, key):
        return self.c[key]

    def __setitem__(self, key, value):
        self.c[key] = value

    def add(self, words):
        self.n += 1
        if not words: 
            return
        w = words[0]
        if w not in self:
            self[w] = WordTrie()
        self[w].add(words[1:])

    def pop(self, key, default=None):
        return self.c.pop(key, default)

    def words(self):
        return list(self.c.keys())

    def str(self, tabs=0):
        s = 'n=' + str(self.n) + '\n'
        for w in self.c:
            s += '\t'*(tabs+1) + str(w) + ': '
            s += self.c[w].str(tabs+1)
        return s


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

    print(model.trie.str())

    PP = [model.perplexity(s) for s in test_corpus]
    return '\n'.join(map(str, PP))


if __name__ == '__main__':
    sys.exit(main())