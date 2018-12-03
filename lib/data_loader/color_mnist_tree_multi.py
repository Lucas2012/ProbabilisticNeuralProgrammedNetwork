#!/usr/bin/env python
import os.path as osp

import _init_paths
import numpy as np
import PIL.Image
import scipy.misc
import torch
import random
from lib.tree import Tree
import scipy.io


class COLORMNISTTREE():
    def __init__(self, batch_size=16, directory='/cs/vml4/Datasets/COLORMNIST', random_seed=12138, folder='TWO_5000',
                 phase='train', shuffle=True, fileformat='png'):
        # basic
        self.folder = folder
        self.phase = phase
        self.dir = directory
        self.batch_size = batch_size
        self.fileformat = fileformat

        # static tree structure
        self.treefile = osp.join('data', 'COLORMNIST', folder, phase + '_parents.list')
        self.functionfile = osp.join('data', 'COLORMNIST', folder, phase + '_functions.list')
        self.infofile = osp.join(directory, folder, phase + '_text.txt')
        with open(self.infofile, 'r') as f:
            self.info = []
            self.number = []
            for line in f:
              line = line[:-1].split(' ')
              self.number += [int(line[0][0])]
              line[0] = line[0][2:]
              self.info   += [line]

        self.trees = self.read_trees()
        self.functions = self.read_functions()

        # parse info
        imgdir = osp.join(self.dir, folder, phase)
        self.imglist, self.trees, self.dictionary = self.read_info()

        # iterator part
        self.random_generator = random.Random()
        self.random_generator.seed(random_seed)
        self.files = {}
        self.files[phase] = {
            'img': self.imglist,
            'tree': self.trees,
        }

        self.idx_ptr = 0
        self.indexlist = list(xrange(len(self.imglist)))

        self.shuffle = shuffle
        if shuffle:
            self.random_generator.shuffle(self.indexlist)

        # (H, W, C)
        self.im_size = self.test_read()

    def __len__(self):
        return len(self.files[self.phase]['img'])

    def test_read(self):
        data_file = self.files[self.phase]

        # load image
        if self.fileformat == 'mat':
            mat_file = data_file['img'][0] + '.mat'
            img = scipy.io.loadmat(mat_file)['data']
        elif self.fileformat == 'png':
            img_file = data_file['img'][0] + '.png'
            img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.float32)

        return img.shape

    def read_functions(self):
        functionfile = self.functionfile
        f = open(functionfile)

        functions = []
        for line in f:
            functions += [line[:-1].split(" ")]

        return functions

    def next_batch(self):
        data_file = self.files[self.phase]

        imgs, trees, categories = [], [], []
        for i in range(0, min(self.batch_size, self.__len__() - self.idx_ptr)):
            index = self.indexlist[self.idx_ptr]

            # load image
            if self.fileformat == 'mat':
                mat_file = data_file['img'][index] + '.mat'
                img = scipy.io.loadmat(mat_file)['data']
            elif self.fileformat == 'png':
                img_file = data_file['img'][index] + '.png'
                img = PIL.Image.open(img_file)
            img = np.array(img, dtype=np.float32)
            img = (img - 127.5) / 127.5

            imgs.append(img)

            # load label
            tree = data_file['tree'][index]
            trees.append(tree)
            categories.append(self.get_categorical_list(tree))

            self.idx_ptr += 1

        imgs = np.array(imgs, dtype=np.float32).transpose(0, 3, 1, 2)

        refetch = False
        if self.idx_ptr >= self.__len__():
            self.idx_ptr = 0
            refetch = True
            if self.shuffle:
                self.random_generator.shuffle(self.indexlist)

        return torch.from_numpy(imgs), trees, categories, refetch

    def next_batch_multigpu(self):
        data_file = self.files[self.phase]

        imgs, tree_indices = [], []
        for i in range(0, min(self.batch_size, self.__len__() - self.idx_ptr)):
            index = self.indexlist[self.idx_ptr]

            # load image
            if self.fileformat == 'mat':
                mat_file = data_file['img'][index] + '.mat'
                img = scipy.io.loadmat(mat_file)['data']
            elif self.fileformat == 'png':
                img_file = data_file['img'][index] + '.png'
                img = PIL.Image.open(img_file)
            img = np.array(img, dtype=np.float32)
            img = (img - 127.5) / 127.5

            imgs += [img]

            # load label
            tree_indices.append(index)

            self.idx_ptr += 1

        imgs = np.array(imgs, dtype=np.float32).transpose(0, 3, 1, 2)

        refetch = False
        if self.idx_ptr >= self.__len__():
            self.idx_ptr = 0
            refetch = True
            if self.shuffle:
                self.random_generator.shuffle(self.indexlist)

        return torch.from_numpy(imgs), torch.from_numpy(np.array(tree_indices)), refetch

    def get_tree_by_idx(self, index):
        return self.files[self.phase]['tree'][index]

    def get_tree_list_current_epoch(self):
        tree_list = list()
        for idx in self.indexlist:
            tree_list.append(self.files[self.phase]['tree'][idx])
        return tree_list

    def read_info(self):
        imglist = []
        dictionary = []
        functions = self.functions

        count = 0
        for tree in self.trees:
            imglist.append(osp.join(self.dir, self.folder, self.phase, 'image{:05d}'.format(count)))
            words, numbers, bboxes = self._extract_info(self.info[count], self.number[count])
            dictionary = list(set(dictionary + words))
            self.trees[count] = self._read_info(tree, words, functions[count], numbers, bboxes)
            count += 1

        return imglist, self.trees, dictionary

    def _extract_info(self, line, num):
        words = []
        numbers = []
        bboxes = []
        count = 0

        numid = 0
        for ele in line:
            words.append(ele)
            if self._is_number(ele):
                numbers.append(ele)
                numid += 1
                if numid == num:
                    count += 1
                    break
            count += 1

        newid = count
        for i in range(0, newid):
            if self._is_number(line[i]):
                bboxes += [[int(ele) for ele in line[count:count + 4]]]
                count += 4
            else:
                bboxes += [[]]

        return words, numbers, bboxes

    def _is_number(self, n):
        try:
            int(n)
            return True
        except:
            return False

    def _read_info(self, tree, words, functions, numbers, bboxes):
        for i in range(0, tree.num_children):
            tree.children[i] = self._read_info(tree.children[i], words, functions, numbers, bboxes)

        tree.word = words[tree.idx]
        tree.function = functions[tree.idx]
        tree.bbox = np.array(bboxes[tree.idx])

        return tree

    def read_trees(self):
        filename = self.treefile
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in f.readlines()]
        return trees

    def read_tree(self, line):
        parents = map(int, line.split())
        trees = dict()
        root = None
        for i in xrange(1, len(parents) + 1):
            # if not trees[i-1] and parents[i-1]!=-1:
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    # if trees[parent-1] is not None:
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def get_categorical_list(self, tree):
        categorical_list, attr_list = self._get_categorical_list(tree)
        return categorical_list

    def _get_categorical_list(self, tree):
        # must be post-ordering traversal, parent need info from children
        category_list = list()
        attr_list = list()
        for child in tree.children:
            children_category_list, children_attr_list = self._get_categorical_list(child)
            category_list += children_category_list
            attr_list += children_attr_list

        if tree.function == 'describe':
            bbox = tree.bbox
            adapted_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            attr_list.append(tree.word)
            attr_vec = self._get_attr_vec(attr_list)
            obj_category = (adapted_bbox, attr_vec)
            category_list.append(obj_category)

        if tree.function == 'combine':
            attr_list.append(tree.word)  # just pass its word to parent

        return category_list, attr_list

    def _get_attr_vec(self, attr_list):
        vec = np.zeros(len(self.dictionary), dtype=np.float64)
        for attr in attr_list:
            attr_idx = self.dictionary.index(attr)
            vec[attr_idx] = 1.0
        return vec
'''
# Single card
loader = COLORMNISTTREE(directory='data/COLORMNIST', folder='TWO_5000_64_modified')
trees = loader.trees
print(trees[0].word)
print(loader.dictionary)
for i in range(0, 1):
  im, tr, cats, ref = loader.next_batch()
  print(tr[0].function)
  print(tr[0].word)
  print(tr[0].children[0].word)
  print(cats[0])
'''
'''
# Multi-gpu
loader = COLORMNISTTREE(directory='data/COLORMNIST', folder='ONE')
trees = loader.trees
print(trees[0].word)
print(loader.dictionary)
for i in range(0, 1000):
  im, tr_indices, ref = loader.next_batch_multigpu()
  tree = loader.get_tree_by_idx(tr_indices[0])
  print(tree.function)
  print(tree.word)
  print(tree.children[0].word)
'''
