"""
This script adds a parent pointer for all tree nodes.
The original trees generated together with CLEVR images doesn't have this kind of parent pointer,
but the parent pointer can be useful when we implementing PNP-Net 

"""

import _init_paths
import pickle
from lib.tree import Tree
import os.path as osp
import os

def add_parent(tree):
  tree = _add_parent(tree, None)

  return tree

def _add_parent(tree, parent):
  tree.parent = parent
  for i in range(0, tree.num_children):
    tree.children[i] = _add_parent(tree.children[i], tree) 

  return tree

path = 'CLEVR_128_NEW/trees_no_parent'
outpath = 'CLEVR_128_NEW/trees'

split = ['train', 'test']

for s in split:
  treepath = osp.join(path, s)
  files = os.listdir(treepath)
  try:
    os.makedirs(osp.join(outpath, s))
  except:
    pass

  for fi in files:
    if fi.endswith('tree'):
      with open(osp.join(path, s, fi), 'rb') as f:
        treei = pickle.load(f)
      treei = add_parent(treei)
      pickle.dump(treei, open(osp.join(outpath, s, fi), 'wb'))
