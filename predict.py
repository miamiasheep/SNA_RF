'''
All the code is own by Eric L. Lee
'''


import pickle
import util
from multiprocessing import Process
import networkx as nx
import pandas as pd
import sys
import os
import numpy as np
from sklearn import preprocessing


if len(sys.argv) != (1+3):
	print '1:model_file'
	print '2:filled_file'
	print '3:train_file'
	exit(-1)
model_file = sys.argv[1]
filled_file = sys.argv[2]
train_file = sys.argv[3]
### set the features ...
features = ['cn','en','r_en','adar','age_diff','jaccard','PA','RI','sex','sex_prop','age_norm']
pair = [2,4,5,7,15,30,33,35,54]
pair_features = []
for p in pair:
	pair_features.append('f%d' % p)
	pair_features.append('f%d_ratio' % p)
rec = ['sex_opp','sex_rec','age_opp','age_rec']
sn_features = ['sex_s','sex_n','age_s','age_n']
katz_features = ['katz','adar_C']
degree_features = ['degree_s','degree_n','degree_diff']
features.extend(pair_features)
features.extend(rec)
features.extend(sn_features)
features.extend(katz_features)
features.extend(degree_features)
print features

with open('test_nodes.txt','r') as f:
	seeds = f.readline().split()
with open(filled_file,'r') as f:
	fill = {}
	for line in f:
		w = line.split(':')
		fill[int(w[0])] = w[1]
print 'read the training file ...'
train = pd.read_csv(train_file)	
c49 = pd.read_csv('pre_nodes_profile_sorted.csv',usecols = ['column_49'])['column_49']
print 'Done!!!'
G = util.buildGraph('train_edges.txt')
n_top = 30
print 'read model ...'
model = pickle.load(open(model_file,'rb'))
params = {'n_jobs':1,'oob_score':False,'verbose' : 0}
model.set_params(**params)
print model
print 'Done!!!'
print '='*100
#scaler = preprocessing.StandardScaler().fit(train[features])
def main(num,file_name):
	count = 0
	with open(file_name,'w') as g:
		for s in seeds[500*num : (num+1)*500]:
			s = int(s)
			### be ware of scaling, training and testing must be consistent
			g.write('%d:' %s)
			
			if (s in G.nodes()):
				### predict the people using ind_model
				sub = train[train['s'] == s]
				cand = sub[sub['label'] == 0]
				cand_num = cand['n']
				cand_arr = cand[features].values
				#cand_scale = scaler.transform(cand[features])
				rank = []
				for i in range(30):
					rank.append([-100,-1])
				for c in range(len(cand)):
					score = model.predict_proba(cand_arr[c])[:,1][0] 
					n = cand_num[c + cand_num.index[0]]
					n = int(n)
					if (c49[s] ==0) and (c49[n]==0):
						continue
					
					### insert the score
					if score > rank[n_top-1][0]:
						rank[n_top-1] = [score,n]
						for i in range(n_top-1,0,-1):
							if rank[i][0] > rank[i-1][0]:
								(rank[i],rank[i-1]) = util.swap(rank[i],rank[i-1])
			
				### write the file!!!
				for i in range(len(rank)):
					if(i == (len(rank)-1)):
						g.write('%d' % rank[i][1])
					else:
						g.write('%d,' % rank[i][1])
				g.write('\n')
			else:
				### fill the content of fill file
				g.write(fill[s])
			if(count%10) == 0:
				print count
			count += 1
			
					
dir = 'check_exp'
if os.path.exists(dir) == False:
	os.system('mkdir ' + dir)
p = []
for i in range(20):
	p.append(Process(target=main, args= (i , dir + '/%d.predict' % (i))))
for i in range(20):
	p[i].start()
print 'Done!!!'