import ngram
import unittest


class TestNGramModel(unittest.TestCase):

	def test_build_ngram_trie_empty(self):
		trie = ngram.NGramModel.build_ngram_trie([], 1)
		self.assertFalse(trie.words() or trie.n)

	def test_build_unigram_trie(self):
		dummy_corpus = ['<s> a b a b a b </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 1)
		self.assertEqual(trie['a'].c, {})

	def test_build_bigram_trie(self):
		dummy_corpus = ['<s> a b a b a b </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 2)
		self.assertEqual(trie['a']['b'].c, {})

	def test_build_trigram_trie(self):
		dummy_corpus = ['<s> a b a b a b </s>'.split()]
		trie = ngram.NGramModel.build_ngram_trie(dummy_corpus, 3)
		self.assertEqual(trie['a']['b']['a'].c, {})

	def test_build_zerogram_trie(self):
		with self.assertRaises(IndexError):
			trie = ngram.NGramModel.build_ngram_trie([], 0)

	def test_instantiate_unigram_model(self):
		model = ngram.NGramModel([], 1)
		self.assert



class TestWordTrie(unittest.TestCase):

	def test_instantiate(self):
		trie = ngram.WordTrie()
		self.assertEqual(trie.n, 0)
		self.assertEqual(trie.c, {})

	def test_set_item(self):
		trie = ngram.WordTrie()
		trie['key'] = 'value'
		self.assertEqual(trie.c['key'], 'value')

	def test_get_item(self):
		trie = ngram.WordTrie()
		trie.c['key'] = 'value'
		self.assertEqual(trie['key'], 'value')

	def test_add_word_list(self):
		trie = ngram.WordTrie()
		trie.add(['one', 'two', 'three'])
		self.assertTrue(trie['one']['two']['three'])

	def test_add_single_words(self):
		trie = ngram.WordTrie()
		trie.add(['one'])
		trie.add(['two'])
		trie.add(['three'])
		self.assertTrue(trie['one'] and trie['two'] and trie['three'])

	def test_count_leaf_words(self):
		trie = ngram.WordTrie()
		trie.add(['one', 'two', 'three'])
		trie.add(['one', 'two', 'three'])
		self.assertEquals(trie.count(['one', 'two', 'three']), 2)

	def test_count_root_words(self):
		trie = ngram.WordTrie()
		trie.add(['one', 'two', 'three'])
		trie.add(['one', 'two', 'four'])
		self.assertEquals(trie.count(['one']), 2)

	def test_contains_true(self):
		trie = ngram.WordTrie()
		trie.add(['word'])
		self.assertTrue('word' in trie)

	def test_contains_true(self):
		trie = ngram.WordTrie()
		self.assertFalse('word' in trie)

	def test_make_single_word_unknown(self):
		trie = ngram.WordTrie()
		trie.add(['word'])
		trie.make_unknown('word')
		self.assertTrue('<UNK>' in trie and 'word' not in trie)

	def test_make_multiple_words_unknown(self):
		trie = ngram.WordTrie()
		trie.add(['one'])
		trie.add(['two'])
		trie.make_unknown('one')
		trie.make_unknown('two')
		self.assertEquals(trie.count(['<UNK>']), 2)



class TestParseArgs(unittest.TestCase):

	def test_correct_usage(self):
		argv = ['ngram.py', '1', 'arg2', 'arg3', 'arg4']
		ret = ['1', 'arg2', 'arg3', 'arg4']
		self.assertEqual(ngram.parse_args(argv), ret)

	def test_empty_args(self):
		argv = []
		with self.assertRaises(ngram.UsageError):
			ngram.parse_args(argv)

	def test_invalid_mode_arg(self):
		argv = ['ngram.py', 'arg1', 'arg2', 'arg3', 'arg4']
		with self.assertRaises(ngram.UsageError):
			ngram.parse_args(argv)

	def test_not_enough_args(self):
		argv = ['ngram.py', '1', 'arg2', 'arg3']
		with self.assertRaises(ngram.UsageError):
			ngram.parse_args(argv)

	def test_excess_args(self):
		argv = ['ngram.py', '1', 'arg2', 'arg3', 'arg4', 'arg5']
		ret = ['1', 'arg2', 'arg3', 'arg4']
		self.assertEqual(ngram.parse_args(argv), ret)


class TestUnprocessedFile(unittest.TestCase):

	#TODO
	pass


class TestProcessedFile(unittest.TestCase):

	#TODO
	pass
