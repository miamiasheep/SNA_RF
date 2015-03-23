import pickle
import sys

if len(sys.argv) != (1+2):
	print '1:input'
	print '2:output'
	exit(-1)
input = sys.argv[1]
output = sys.argv[2]

with open('mapping.pickle','rb') as f:
	map = pickle.load(f)
with open('data/existing_nodes.pickle','rb') as f:
	existing_nodes = pickle.load(f)

seeds = []
with open('test_nodes.txt','r') as f:
	words = f.readline().split()
	for w in words:
		if int(w) in existing_nodes:
			seeds.append(map[int(w)])

print 'load the data...'
with open(input,'rb') as f:
	target = pickle.load(f)

print 'Done!!!'

print 'write the data...'	
with open(output,'wb') as f:
	pickle.dump(target[seeds,],f)
print 'Done!!!'

	
	
	
	