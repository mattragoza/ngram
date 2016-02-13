'''\
usage: python ngram.py MODE TRAIN_FILE DEV_FILE TEST_FILE\
'''

import sys
import nltk


class UsageError(Exception):
    pass


class UnigramModel:

    def __init__(self, train_corpus, k=1):

        # count unigrams in train corpus
        n = 0
        counts = {}
        for line in train_corpus:
            for unigram in line:
                if unigram in counts:
                    counts[unigram] += 1
                else:
                    counts[unigram] = 1
                n += 1

        self.mle = {'<unk>': 0}
        for unigram in counts:
            if counts[unigram] > k:
                self.mle[unigram] = counts[unigram]/n
            else:
                self.mle['<unk>'] += counts[unigram]/n

    def per_word_perplexity(self, line):

        p = 1
        for unigram in line:
            if unigram in self.mle:
                p *= self.mle[unigram]
            else:
                p *= self.mle['<unk>']

        return p**(-1/len(line))


class BigramModel:

    def __init__(self, train_corpus):
        
        # count bigrams in train corpus
        n = 0
        counts = {}
        for line in train_corpus:
            for bigram in zip(line, line[1:]):
                if bigram in counts:
                    counts[bigram] += 1
                else:
                    counts[bigram] = 1
                n += 1

        self.mle = {}
        for bigram in counts:
            self.mle[bigram] = counts[bigram]/n

    def per_word_perplexity(self, line):

        p = 1
        for bigram in zip(line, line[1:]):
            if bigram in self.mle:
                p *= self.mle[bigram]
            else:
                raise NotImplementedError()

        return p**(-1/len(line))


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

    if argv[1] not in ['1', '2', '3']:
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
        unigram_model = UnigramModel(train_corpus)
        for line in test_corpus:
            p = unigram_model.per_word_perplexity(line)
            print(' '.join(line) + ' ' + str(p))

    elif mode == '2':
        bigram_model = BigramModel(train_corpus)
        for line in test_corpus:
            p = bigram_model.per_word_perplexity(line)
            print(' '.join(line) + ' ' + str(p))

    elif mode == '3':
        return 'TODO'


if __name__ == '__main__':
    sys.exit(main())