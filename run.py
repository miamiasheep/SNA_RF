### run the script
import os

print 'Obtain the information from katz algorithm ...'
dir = '/tmp2/r01922164/old/'  ### You can modify the directory
### test first
os.system('python katz.py')
os.system('reduce.py ' + dir + 'B.pickle ' + dir + 'B_reduced.pickle')
os.system('reduce.py ' + dir + 'C.pickle ' + dir + 'C_reduced.pickle')
