import sys
import pickle

if len(sys.argv) != (1+1):
	print 'Usage: python reverse_map.py [input]'
	exit(-1)
input = sys.argv[1]

with open(input,'rb') as f:
	map = pickle.load(f)
reverse = {}

for key,value in map.iteritems():
	reverse[value] = key
with open('reverse_' + input,'wb') as f:
	pickle.dump(reverse,f)