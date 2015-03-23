import networkx as nx

def swap(a,b):
	return b,a
### build the graph using TA's format
def buildGraph(file_name):
	with open(file_name,'r') as f:
		### Distinguish which type is the graph
		skip = f.readline() ### skip the type
		skip = f.readline() ### skip user_id 
		G = nx.Graph()	
		### Construct the Graph
		for line in f:
			words = line.strip().split()
			G.add_edge(int(words[0]),int(words[1]))
	return G
if __name__ == '__main__':
	G = buildGraph('train_edges.txt')
