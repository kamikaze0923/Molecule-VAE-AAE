import nltk
import numpy as np
import six

# the zinc grammar
gram = """smiles -> chain
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
aliphatic_organic -> 'B'
aliphatic_organic -> 'C'
aliphatic_organic -> 'N'
aliphatic_organic -> 'O'
aliphatic_organic -> 'S'
aliphatic_organic -> 'P'
aliphatic_organic -> 'F'
aliphatic_organic -> 'I'
aliphatic_organic -> 'Cl'
aliphatic_organic -> 'Br'
aromatic_organic -> 'c'
aromatic_organic -> 'n'
aromatic_organic -> 'o'
aromatic_organic -> 's'
bracket_atom -> '[' BAI ']'
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> chiral BAH
BAC -> BAH
BAC -> chiral
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge class
BACH -> charge
BACH -> class
symbol -> aliphatic_organic
symbol -> aromatic_organic
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1'
DIGIT -> '2'
DIGIT -> '3'
DIGIT -> '4'
DIGIT -> '5'
DIGIT -> '6'
DIGIT -> '7'
DIGIT -> '8'
chiral -> '@'
chiral -> '@@'
hcount -> 'H'
hcount -> 'H' DIGIT
charge -> '-'
charge -> '-' DIGIT
charge -> '-' DIGIT DIGIT
charge -> '+'
charge -> '+' DIGIT
charge -> '+' DIGIT DIGIT
bond -> '-'
bond -> '='
bond -> '#'
bond -> '/'
bond -> '\\'
ringbond -> DIGIT
ringbond -> bond DIGIT
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
RB -> RB ringbond
RB -> ringbond
BB -> BB branch
BB -> branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
Nothing -> None"""

# form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)
# start_index = GCFG.productions()[0].lhs()

# collect all lhs symbols, and the unique set of them
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]


D = len(GCFG.productions())

# this map tells us the rhs symbol indices for each production rule
# rhs_map = [None]*D
# count = 0
# for prod in GCFG.productions():
    # print(prod)
    # rhs_map[count] = []
    # for rhs in prod.rhs():
        # if not isinstance(rhs,six.string_types):
            # s = rhs.symbol()
            # rhs_map[count].append(list(np.where(np.array(lhs_list) == s)[0]))
    # count = count + 1

masks = np.zeros((len(all_lhs),D))

# this tells us for each lhs symbol which productions rules should be masked
for count, sym in enumerate(all_lhs):
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
	
np.save("mask.npy", masks)

# # this tells us the indices where the masks are equal to 1
# index_array = []
# for i in range(masks.shape[1]):
    # idx = np.where(masks[:,i]==1)[0][0]
    # index_array.append(idx)
# ind_of_ind = np.array(index_array)

# max_rhs = max([len(l) for l in rhs_map])

# # rules 29 and 31 aren't used in the zinc data so we 
# # 0 their masks so they can never be selected
# # masks[:,29] = 0
# # masks[:,31] = 0
