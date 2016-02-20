import ngram
import pytest


class TestNGramModel(object):

	pass


class TestNGramTrie(object):

	def test_instantiate_empty(self):
		trie = ngram.NGramTrie()
		assert trie.n == 0
		assert trie.c == {}

	def test_set_item(self):
		trie = ngram.NGramTrie()
		trie['key'] = 'value'
		assert trie.c['key'] == 'value'

	def test_get_item(self):
		trie = ngram.NGramTrie()
		trie.c['key'] = 'value'
		assert trie['key'] == 'value'

	def test_contains_true(self):
		trie = ngram.NGramTrie()
		trie.c['word'] = None
		assert 'word' in trie

	def test_contains_false(self):
		trie = ngram.NGramTrie()
		assert 'word' not in trie

	def test_add_trigram(self):
		trie = ngram.NGramTrie()
		trie.add(['one', 'two', 'three'])
		assert trie.c['one'].c['two'].c['three']

	def test_add_unigrams(self):
		trie = ngram.NGramTrie()
		trie.add(['one'])
		trie.add(['two'])
		trie.add(['three'])
		assert trie.c['one'] and trie.c['two'] and trie.c['three']

	def test_count_leaf_words(self):
		trie = ngram.NGramTrie()
		trie.add(['one', 'two', 'three'])
		trie.add(['one', 'two', 'three'])
		assert trie.count(['one', 'two', 'three']) == 2

	def test_count_root_words(self):
		trie = ngram.NGramTrie()
		trie.add(['one', 'two', 'three'])
		trie.add(['one', 'two', 'four'])
		assert trie.count(['one']) == 2


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
