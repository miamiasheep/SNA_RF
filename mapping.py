import util
import pickle 

print 'building Graph ...'
G = util.buildGraph('train_edges.txt')
print 'Done!!!'

result = {}
for i in range(len(G.nodes())):
	if(i % 10000) == 0:
		print i
	result[int(G.nodes()[i])] = i
   
with open('data/mapping.pickle','wb') as f:
	pickle.dump(result,f)
    
	
### seeds mapping
with open('data/existing_nodes.pickle','rb') as f:
	existing_nodes = pickle.load(f)
seeds = []
with open('test_nodes.txt','r') as f:
	words = f.readline().split()
	for w in words:
		if int(w) in existing_nodes:
			seeds.append(int(w))	
seeds_map = {}
for i in range(len(seeds)):
	if(i%1000) == 0:
		print i
	seeds_map[seeds[i]] = i
with open('data/seeds_map.pickle','wb') as f:
	pickle.dump(seeds_map,f)


	