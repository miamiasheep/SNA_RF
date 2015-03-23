'''
All the code is own by Eric L. Lee
'''

'''
I am responsible for sampling and generate features
and generate the file to train
'''

### build in library
import random
import pickle
#from sklearn.linear_model import LogisticRegression ###may be useless
import sys
import networkx as nx
import os
from multiprocessing import Process
import pandas as pd
import math
import util
import numpy as np 
from scipy.stats import norm


def fill_mean(attr):
	m = np.mean(attr[attr!=0])
	attr[attr == 0] = m
	attr[pd.isnull(attr)] = m
	return attr

def fill(attr):
	m = np.mean(attr[attr!=0])
	attr[pd.isnull(attr)] = m
	return attr
	
	
### global varibles
data = pd.read_csv('pre_nodes_profile_sorted.csv')
column_4 = pd.read_csv('pre_nodes_profile_sorted.csv',usecols = ['column_4'])['column_4']
age = pd.read_csv('pre_nodes_profile_sorted.csv',usecols = ['column_6'])['column_6']
age = fill_mean(age)
sex = pd.read_csv('pre_nodes_profile_sorted.csv',usecols = ['column_45'])['column_45']
en = pickle.load(open('data/global_en.pickle','rb'))
c49 = pd.read_csv('pre_nodes_profile_sorted.csv',usecols = ['column_49'])['column_49']

with open('/tmp2/r01922164/old/B_reduced.pickle','rb') as f:
	B = pickle.load(f)

print 'read C...'
C = pickle.load(open('/tmp2/r01922164/old/C_reduced.pickle','rb'))
print 'C to dense'
C = C.todense()

with open('data/mapping.pickle','rb') as f:
	map = pickle.load(f)
with open('data/seeds_map.pickle','rb') as f:
	seeds_map = pickle.load(f)
with open('data/reverse_mapping.pickle','rb') as f:
	reverse_map = pickle.load(f)

### build the social network's graph
G = util.buildGraph('train_edges.txt')
features = ['s','n','cn','en','r_en','adar','age_diff','jaccard','PA','RI','sex','sex_prop','age_norm']

### Add ratio features
pair = [2,4,5,7,15,30,33,35,44,54]
pair_features = []
for p in pair:
	pair_features.append('f%d' % p)
	pair_features.append('f%d_ratio' % p)
rec = ['sex_opp','sex_rec','age_opp','age_rec','f16_s','f16_n']
sn_features = ['sex_s','sex_n','age_s','age_n']
degree_features = ['degree_s','degree_n','degree_diff']
katz_features = ['C','katz','adar_C']
features.extend(pair_features)
features.extend(rec)
features.extend(sn_features)
features.extend(katz_features)
features.extend(degree_features)
print features
print '=' * 100

def adar(cn_set, G):
	ascore = 0.0
	for t in cn_set:
		if G.degree(t) == 0:
			continue
		else:
			ascore += 1.0/math.log(float(G.degree(t))+0.01)
	return ascore
	
def resource_index(cn_set, G):
	ascore = 0.0
	for t in cn_set:
		if G.degree(t) == 0:
			continue
		else:
			ascore += 1.0/(float(G.degree(t))+0.01)
	return ascore
	
def cal_en(s,cn_set,en):
	score = 0.0
	for t in cn_set:
		score += (en[s][t] + 0.0001)
	return score
	
### To Do :
### Add reciprocal information for sex and age
def gen_features(f,s,n,s_set,label):
	values = {}
	for p in pair:
		temp = set()
		if pd.isnull(data['column_%d' % p][n]) == False:
			temp = set(data['column_%d' % p][n].split())
		values['f%d' % p] = len(s_set['f%d' % p].intersection(temp))
		total = float(len(temp) + len(s_set['f%d' % p ]))
		values['f%d_ratio' % p] = 0
		if total > 0:
			values['f%d_ratio' % p] = values['f%d' % p]/total
	n_set = set(nx.neighbors(G,int(n)))
	common_neighbor_set = s_set['n'].intersection(n_set)
	values['s'] = s
	values['n'] = n
	values['cn'] = len(common_neighbor_set)
	values['en'] = cal_en(s,common_neighbor_set,en)
	values['r_en'] = cal_en(s,common_neighbor_set,en) * cal_en(n,common_neighbor_set,en)
	values['adar'] = adar(common_neighbor_set,G)
	values['age_diff'] = abs(age[s]-age[n])
	values['jaccard'] = float(len(common_neighbor_set))/len(s_set['n'].union(n_set))
	values['PA'] = G.degree(n) * G.degree(s)
	values['RI'] = resource_index(common_neighbor_set,G)
	values['sex'] = 0
	values['sex_s'] = 0
	values['sex_n'] = 0
	if not (pd.isnull(sex[s]) or pd.isnull(sex[n])):
		values['sex'] = abs(sex[s] - sex[n])
		values['sex_s'] = sex[s]
		values['sex_n'] = sex[n]
	values['sex_prop'] = 0
	if pd.isnull(sex[n]) == False:
		values['sex_prop'] = s_set['sex'][int(sex[n])]
	### add reciprocal information!
	### sex reciprocal info
	count = 0
	for i in nx.neighbors(G,n):
		if not pd.isnull(sex[i]):
			if sex[i] == sex[s]:
				count += 1
	values['sex_opp'] = float(count) / len(nx.neighbors(G,n))
	values['sex_rec'] = values['sex_prop'] * values['sex_opp']	
	
	
	### age norm projects ...
	if s_set['age']['std']!=0:
		values['age_norm'] = norm(s_set['age']['mean'],s_set['age']['std']).pdf(age[n])
	elif s_set['age']['mean'] == age[n]:
		values['age_norm'] = 1
	else:
		values['age_norm'] = 0
	### age rec
	age_list = []
	for nn in n_set:
		age_list.append(age[int(nn)])
	age_mean = np.mean(age_list)
	age_std = np.std(age_list)
	if age_std != 0:
		values['age_opp'] = norm(age_mean,age_std).pdf(age[s])
	elif age_mean == age[s]:
		values['age_opp'] = 1
	else:
		values['age_opp'] = 0
	values['age_rec'] = values['age_opp'] * values['age_norm']
	### add new age features
	values['age_s'] = age[s]
	values['age_n'] = age[n]
	
	### for exp27
	values['f16_s'] = data['column_16'][s]
	values['f16_n'] = data['column_16'][n]	
	### katz features
	values['C'] = C[seeds_map[s],map[n]]
	values['katz'] = values['cn'] + values['C']*0.03
	values['adar_C'] = values['adar'] + values['C']*0.01
	
	### degree features 
	values['degree_s'] = G.degree(s)
	values['degree_n'] = G.degree(n)
	values['degree_diff'] = abs(G.degree(s) - G.degree(n))
	first = True
	for name in features:
		if first:
			f.write('%f' % values[name])
			first = False
		else:	
			f.write(',%f' % values[name])
	f.write(',%d\n' % label)

	
dir = '/tmp2/r01922164/2_level_4'

### make the directory
if os.path.exists(dir) == False:
	os.system('mkdir ' + dir)

with open('test_nodes.txt','r') as f:
	seeds = f.readline().split()

def main(num):
	### generate the features	
	count = 0
	train = []
	with open(dir + '/train_%d.csv' % num, 'w') as f:
		#f.write('cn,en,r_en,adar,f4,age_diff,jaccard,PA,RI,sex,label\n')
		for name in features:
			f.write(name + ',')
		f.write('label\n')
		
		for s in seeds[num*500 : (num+1)*500]:
			s = int(s)
			### only the s in G.nodes will be calculated
			if s in G.nodes():
				### One s calculate once!!!				
				s_set = dict()
				for p in pair:
					s_set['f%d' % p] = set()
					if pd.isnull(data['column_%d' % p][s]) == False:
						s_set['f%d' % p] = set(data['column_%d' % p][s].split())
				
				s_set['n'] = set(nx.neighbors(G,int(s)))
				###implement neighbors sex
				counter = [0,0]
				for ss in s_set['n']:
					if pd.isnull(sex[ss]):
						continue
					counter[int(sex[int(ss)])] += 1
				total = float(counter[0] + counter[1])	
				s_set['sex'] = [counter[0]/total,counter[1]/total]
				s_set['age'] = {}
				s_set['age']['mean'] = np.mean(age[nx.neighbors(G,s)])
				s_set['age']['std'] = np.std(age[nx.neighbors(G,s)])
				
					
				pos = set([p for p in nx.neighbors(G,s) if map[p] in set(B[seeds_map[s],].nonzero()[1])])
				
				neg = set()
				for i in B[seeds_map[s],].nonzero()[1]:
					neg.add(reverse_map[i])
							
				neg = neg.difference(pos)
				neg = neg.difference(set([s]))
				
				
				for n in pos:
					if c49[s]==0 and c49[n]==0:
						continue
					gen_features(f,s,n,s_set,1)
				for n in neg:
					if c49[s]==0 and c49[n]==0:
						continue
					gen_features(f,s,n,s_set,0)

			if (count%10) ==0:
				print count
			count += 1


### multiple process
p = []
for i in range(20):
	p.append(Process(target=main, args=[i]))
for i in range(20):
	p[i].start()		
		
print 'Done!!!'
	