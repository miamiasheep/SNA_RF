import pickle
import util
import networkx as nx
import sys

G = util.buildGraph('train_edges.txt')
A = nx.adjacency_matrix(G)
A = A.todense()
B = A*A

print 'loading ...'
with open('/tmp2/r01922164/old/B.pickle','wb') as f:
    pickle.dump(B,f)
print 'Done!!!'
B = B.todense()
C = A*B

with open('/tmp2/r01922164/old/C.pickle','wb') as f:
    pickle.dump(C,f)
