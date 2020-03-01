import glob
import pickle
import pdb
import os

pfile = 'coeff_train.pkl'
if not os.path.exists(pfile):
	tlist = []
	files = sorted(glob.glob('coeff/lrw/*/train/*.npy'))
	for file in files:
		splits = file.split('/')
		tlist.append([splits[-3],splits[-2],splits[-1][:-4]])

	_file = open(pfile,"wb")
	pickle.dump(tlist,_file)
	_file.close()

pfile = 'coeff_test.pkl'
if not os.path.exists(pfile):
	tlist = []
	files = sorted(glob.glob('coeff/lrw/*/test/*.npy'))
	for file in files:
		splits = file.split('/')
		tlist.append([splits[-3],splits[-2],splits[-1][:-4]])

	_file = open(pfile,"wb")
	pickle.dump(tlist,_file)
	_file.close()
