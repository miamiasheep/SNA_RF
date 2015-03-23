'''
All the code is own by Eric L. Lee
'''
'''
Use random forest to learn
'''
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
import pickle
import os 

def gen_importance(features,importance,output):
	temp = dict()
	temp ['importance'] = importance
	temp ['features'] = features
	temp = pd.DataFrame(temp)
	temp = temp.sort('importance',ascending = [0])
	temp.to_csv(output,index = False)

    
if len(sys.argv) != (1+4):
	print '1: input'
	print '2: n_trees'
	print '3: depth'
	print '4: n_jobs'
	exit(-1)
input = sys.argv[1]
n_trees = int(sys.argv[2])
depth = int(sys.argv[3])
n_jobs = int(sys.argv[4])

dir = '/tmp2/r01922164/2_level_4/'
### make the directory
if os.path.exists(dir) == False:
	os.system('mkdir ' + dir) 
model_file = dir + 'rf_%d_%d_%d_exp32' % (n_trees,depth,n_jobs)
output = dir + 'importance_%d_%d_%d_exp32' % (n_trees,depth,n_jobs)
loc_score = dir + 'oob_%d_%d_%d_exp32' % (n_trees,depth,n_jobs)
print 'preprocess...'
train = pd.read_csv(input)
params = {'n_estimators':n_trees, 'max_depth':depth ,'random_state':1,'n_jobs':n_jobs,'oob_score':True,'verbose' : 1}
model = RandomForestClassifier(**params)
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
print 'Done!!!'
print '=' * 100 
model.fit(train[features],train['label'])
gen_importance(features,model.feature_importances_,output)

with open(loc_score,'w') as f:
	print model.oob_score_
	f.write('%f\n' % model.oob_score_)

print 'Write the model...'
with open(model_file,'wb') as f:
	pickle.dump(model,f)

print 'Done!!!'


