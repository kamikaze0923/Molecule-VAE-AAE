from __future__ import print_function
import nltk
import zinc_grammar
import numpy as np
import sys
from nltk.draw.tree import draw_trees





MAX_LEN = 277
NRULES = len(zinc_grammar.GCFG.productions())

prod_map = {}
for ix, prod in enumerate(zinc_grammar.GCFG.productions()):
	prod_map[prod] = ix

def get_zinc_tokenizer(cfg):
	all_tokens = cfg._lexical_index.keys()
	long_tokens = filter(lambda a: len(a) > 1, all_tokens)
	replacements = ['$', '%', '^']  # ,'&']
	assert len(long_tokens) == len(replacements)
	for token in replacements:
		assert not cfg._lexical_index.has_key(token)

	def tokenize(smiles):
		for i, token in enumerate(long_tokens):
			smiles = smiles.replace(token, replacements[i])
		tokens = []
		for token in smiles:
			try:
				ix = replacements.index(token)
				tokens.append(long_tokens[ix])
			except:
				tokens.append(token)
		return tokens

	return tokenize

tokenize = get_zinc_tokenizer(zinc_grammar.GCFG)


def to_one_hot(smiles):
	""" Encode a list of smiles strings to one-hot vectors """
	token = tokenize(smiles)
	parser = nltk.ChartParser(zinc_grammar.GCFG)
	parse_tree = parser.parse(token).next()
	draw_trees(parse_tree)
	print(type(parse_tree))
	exit(0)
	productions_seq = parse_tree.productions()
	print(smiles)
	for i in productions_seq:
		print(i)
	exit(0)
	indices = [prod_map[prod] for prod in productions_seq]
	one_hot = np.zeros(shape=(MAX_LEN, NRULES), dtype=np.float32)
	num_productions = len(indices)
	one_hot[np.arange(num_productions), indices] = 1.
	one_hot[np.arange(num_productions, MAX_LEN), -1] = 1.
	return one_hot

if __name__ == "__main__":
	# f = open('250k_rndm_zinc_drugs_clean.smi', 'r')
	# L = []
	# for line in f:
		# line = line.strip()
		# L.append(line)
	# f.close()

	onehot = to_one_hot('O=C(CCCO)Nc1ccc(F)cc1F')
	OH = np.zeros((len(L), MAX_LEN, NRULES),dtype=np.float16)
	print("Totel smiles strings: %d" % len(L))
	for i in range(len(L)):
		if i%1000 == 0:
			print(i)
			sys.stdout.flush()
		onehot = to_one_hot(L[i])
		OH[i-1, :, :] = onehot
	OH = np.moveaxis(OH, 1, -1)

	np.save("one_hot.npy", OH)



