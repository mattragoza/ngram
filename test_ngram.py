import ngram
import pytest


class TestNGramModel(object):

	def test_build_ngram_trie_empty(self):
		trie = ngram.NGramModel.build_ngram_trie([], 1)
		assert not trie.keys()
		assert trie.n == 0

	def test_build_unigram_trie(self):
		dummy_corpus = ['<s> a b a b a b </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 1)
		assert trie['a'].c == {}

	def test_build_bigram_trie(self):
		dummy_corpus = ['<s> a b a b a b </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 2)
		assert trie['a']['b'].c == {}

	def test_build_trigram_trie(self):
		dummy_corpus = ['<s> a b a b a b </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 3)
		assert trie['a']['b']['a'].c == {}

	def test_build_zerogram_trie(self):
		with pytest.raises(IndexError):
			ngram.NGramModel.build_ngram_trie([], 0)

	def test_ngram_trie_mle_known(self):
		dummy_corpus = ['<s> a b a c a b </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 2)
		test_ngram = ['a', 'b']
		assert ngram.NGramModel.ngram_trie_mle(trie, test_ngram) == 2/3

	def test_ngram_trie_mle_oov(self):
		dummy_corpus = ['<s> a b a c a d </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 1)
		test_ngram = ['e']
		assert ngram.NGramModel.ngram_trie_mle(trie, test_ngram) == 0

	def test_ngram_trie_mle_oov_with_unk(self):
		dummy_corpus = ['<s> a b a c a d </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 1)
		trie.combine_unknowns(K=1)
		test_ngram = ['e']
		assert ngram.NGramModel.ngram_trie_mle(trie, test_ngram) == 4/7


class TestWordTrie(object):

	def test_instantiate_empty(self):
		trie = ngram.WordTrie()
		assert trie.n == 0
		assert trie.c == {}

	def test_set_item(self):
		trie = ngram.WordTrie()
		trie['key'] = 'value'
		assert trie.c['key'] == 'value'

	def test_get_item(self):
		trie = ngram.WordTrie()
		trie.c['key'] = 'value'
		assert trie['key'] == 'value'

	def test_add_word_list(self):
		trie = ngram.WordTrie()
		trie.add(['one', 'two', 'three'])
		assert trie['one']['two']['three']

	def test_add_single_words(self):
		trie = ngram.WordTrie()
		trie.add(['one'])
		trie.add(['two'])
		trie.add(['three'])
		assert trie['one'] and trie['two'] and trie['three']

	def test_count_leaf_words(self):
		trie = ngram.WordTrie()
		trie.add(['one', 'two', 'three'])
		trie.add(['one', 'two', 'three'])
		assert trie.count(['one', 'two', 'three']) == 2

	def test_count_root_words(self):
		trie = ngram.WordTrie()
		trie.add(['one', 'two', 'three'])
		trie.add(['one', 'two', 'four'])
		assert trie.count(['one']) == 2

	def test_contains_true(self):
		trie = ngram.WordTrie()
		trie.add(['word'])
		assert 'word' in trie

	def test_contains_false(self):
		trie = ngram.WordTrie()
		assert 'word' not in trie

	def test_combine_unknowns_present(self):
		trie = ngram.WordTrie()
		trie.add(['one'])
		trie.add(['two'])
		trie.combine_unknowns(1)
		assert trie['<UNK>'].n == 2

	def test_combine_unknowns_absent(self):
		trie = ngram.WordTrie()
		trie.add(['one'])
		trie.add(['one'])
		trie.combine_unknowns(1)
		assert '<UNK>' not in trie


class TestParseArgs(object):

	def test_correct_usage(self):
		argv = ['ngram.py', '1', 'arg2', 'arg3', 'arg4']
		ret = ['1', 'arg2', 'arg3', 'arg4']
		assert ngram.parse_args(argv) == ret

	def test_empty_args(self):
		argv = []
		with pytest.raises(ngram.UsageError):
			ngram.parse_args(argv)

	def test_invalid_mode_arg(self):
		argv = ['ngram.py', 'arg1', 'arg2', 'arg3', 'arg4']
		with pytest.raises(ngram.UsageError):
			ngram.parse_args(argv)

	def test_not_enough_args(self):
		argv = ['ngram.py', '1', 'arg2', 'arg3']
		with pytest.raises(ngram.UsageError):
			ngram.parse_args(argv)

	def test_excess_args(self):
		argv = ['ngram.py', '1', 'arg2', 'arg3', 'arg4', 'arg5']
		ret = ['1', 'arg2', 'arg3', 'arg4']
		assert ngram.parse_args(argv) == ret


class TestUnprocessedFile(object):

	#TODO
	pass


class TestProcessedFile(object):

	#TODO
	pass
