#!/usr/bin/env python

import _init_paths
import os
import os.path as osp
import pickle

import numpy as np
import PIL.Image
import torch
import random


class CLEVRTREE():
    # set the thresholds for the size of objects here
    SMALL_THRESHOLD = 16
    # remove the medium size for simplicity and clarity

    NEW_SIZE_WORDS = ['small', 'large']
    OLD_SIZE_WORDS = ['small', 'large']

    def __init__(self, batch_size=16,
                 base_dir='/cs/vml4/zhiweid/ECCV18/GenerativeNeuralModuleNetwork/data/CLEVR/CLEVR_128',
                 random_seed=12138,
                 phase='train', shuffle=True, file_format='png'):
        self.phase = phase
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.fileformat = file_format

        if phase not in ['train', 'test']:
            raise ValueError('invalid phase name {}, should be train or test'.format(phase))

        self.image_dir = osp.join(base_dir, 'images', phase)
        self.tree_dir = osp.join(base_dir, 'trees', phase)

        # get the file names for images, and corresponding trees, and the dictionary of the dataset
        self.image_files, self.tree_files = self.prepare_file_list()
        self.dictionary_path = osp.join(self.base_dir, 'dictionary_tree.pickle')
        self.dictionary = self.load_dictionary(self.tree_files, self.dictionary_path)

        # update the size words, since we need to adjust the size for all objects
        # according to the actual 2-D bounding box
        for word in self.OLD_SIZE_WORDS:
            self.dictionary.remove(word)
        for word in self.NEW_SIZE_WORDS:
            self.dictionary.append(word)

        # iterator part
        self.random_generator = random.Random()
        self.random_generator.seed(random_seed)

        self.files = dict()
        self.files[phase] = {
            'img': self.image_files,
            'tree': self.tree_files,
        }

        self.index_ptr = 0
        self.index_list = list(range(len(self.image_files)))

        self.shuffle = shuffle
        if shuffle:
            self.random_generator.shuffle(self.index_list)

        self.im_size = self.read_first()
        self.ttdim = len(self.dictionary) + 1

    def __len__(self):
        return len(self.files[self.phase]['img'])

    def read_first(self):
        images, _, _, _ = self.next_batch()
        self.index_ptr = 0

        return images.size()

    def prepare_file_list(self):
        """
        Get the filename list for images and correspondign trees
        :return:
        """
        image_list = []
        tree_list = []
        for image_filename in sorted(os.listdir(self.image_dir)):
            image_path = os.path.join(self.image_dir, image_filename)
            image_list.append(image_path)
            filename, _ = os.path.splitext(image_filename)
            tree_filename = filename + '.tree'
            tree_path = os.path.join(self.tree_dir, tree_filename)
            tree_list.append(tree_path)
        return image_list, tree_list

    def load_dictionary(self, tree_files, dictionary_path):
        if osp.isfile(dictionary_path):  # the dictionary has been created, then just load return
            with open(dictionary_path, 'rb') as f:
                dictionary = pickle.load(f)
        else:
            dictionary_set = set()
            for idx, tree_file_path in enumerate(tree_files):
                with open(tree_file_path, 'rb') as f:
                    tree = pickle.load(f)
                tree_words = self.get_tree_words(tree)
                dictionary_set.update(set(tree_words))

            dictionary = list(dictionary_set)

            # update the size words, since we need to adjust the size for all objects
            # according to the actual 2-D bounding box
            for word in self.OLD_SIZE_WORDS:
                dictionary.remove(word)
            for word in self.NEW_SIZE_WORDS:
                dictionary.append(word)
            with open(dictionary_path, 'wb') as f:
                pickle.dump(dictionary, f)

        return dictionary

    def get_tree_words(self, tree):
        words = [tree.word]
        for child in tree.children:
            words += self.get_tree_words(child)
        return words

    def next_batch(self):
        data_file = self.files[self.phase]

        images, trees, categories = [], [], []
        for i in range(0, min(self.batch_size, len(self) - self.index_ptr)):
            index = self.index_list[self.index_ptr]

            # load image
            if self.fileformat == 'png':
                img_file = data_file['img'][index]
                img = PIL.Image.open(img_file)
            else:
                raise ValueError('wrong file format for images')

            # remove the alpha-channel
            img = np.array(img, dtype=np.float32)[:, :, :-1]
            img = (img - 127.5) / 127.5
            images.append(img)

            # load tree
            with open(data_file['tree'][index], 'rb') as f:
                tree = pickle.load(f)
            tree = self.adapt_tree(tree)
            trees.append(tree)
            categories.append(self.get_categorical_list(tree))

            self.index_ptr += 1

        images = np.array(images, dtype=np.float32).transpose(0, 3, 1, 2)

        refetch = False
        if self.index_ptr >= len(self):
            self.index_ptr = 0
            refetch = True
            if self.shuffle:
                self.random_generator.shuffle(self.index_list)

        return torch.from_numpy(images), trees, categories, refetch

    def get_all(self):
        data_file = self.files[self.phase]

        images, trees, categories = [], [], []
        for i in range(len(self.index_list)):
            index = self.index_list[self.index_ptr]

            # load image
            if self.fileformat == 'png':
                img_file = data_file['img'][index]
                img = PIL.Image.open(img_file)
            else:
                raise ValueError('wrong file format for images')

            # remove the alpha-channel
            img = np.array(img, dtype=np.float32)[:, :, :-1]
            img = (img - 127.5) / 127.5
            images.append(img)

            # load tree
            with open(data_file['tree'][index], 'rb') as f:
                tree = pickle.load(f)
            tree = self.adapt_tree(tree)
            trees.append(tree)

            self.index_ptr += 1

        images = np.array(images, dtype=np.float32).transpose(0, 3, 1, 2)

        return torch.from_numpy(images), trees

    def adapt_tree(self, tree):
        tree = self._adapt_tree(tree, parent_bbox=None)
        return tree

    def _adapt_tree(self, tree, parent_bbox):
        # adjust tree.word for object size according to the bounding box, since the original size is for 3-D world
        if tree.function == 'combine' and tree.word in self.OLD_SIZE_WORDS:
            width = parent_bbox[2]
            height = parent_bbox[3]
            tree.word = self._get_size_word(width, height)

        # set the bbox for passing to children
        if tree.function == 'combine':
            bbox_xywh = parent_bbox
        elif tree.function == 'describe':
            bbox_xywh = tree.bbox
        else:
            bbox_xywh = None

        # then swap the bbox to (y,x,h,w)
        if hasattr(tree, 'bbox'):
            bbox_yxhw = (tree.bbox[1], tree.bbox[0], tree.bbox[3], tree.bbox[2])
            tree.bbox = np.array(bbox_yxhw)

        # pre-order traversal
        for child in tree.children:
            self._adapt_tree(child, parent_bbox=bbox_xywh)
        return tree

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
            adapted_bbox = (bbox[1], bbox[0], bbox[3], bbox[2])
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

    @classmethod
    def _get_size_word(cls, width, height):
        maximum = max(width, height)
        if maximum < cls.SMALL_THRESHOLD:
            return 'small'
        else:
            return 'large'


if __name__ == '__main__':
    loader = CLEVRTREE(phase='test',
                       base_dir='/zhiweid/work/gnmn/GenerativeNeuralModuleNetwork/data/CLEVR/CLEVR_64_MULTI_LARGE')
    for i in range(3):
        im, trees, categories, ref = loader.next_batch()
        import IPython;

        IPython.embed()
        break
        print(im[0].shape)
        print(categories[0])
    print(loader.dictionary)

'''
# For testing the format of PIL.Image and cv2
img2 = PIL.Image.open('/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/CLEVR/clevr-dataset-gen/output/images/train/CLEVR_new_000003.png')
img2 = np.array(img2)
img2 = img2[:,:,:-1]

out = PIL.Image.fromarray(img2)
out.save('test.png')
# img = cv2.imread('/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/CLEVR/clevr-dataset-gen/output/images/train/CLEVR_new_000003.png')
# cv2.imwrite('test.png',img)
print(img2)
#
img = PIL.Image.open('test.png')
img = np.array(img)

print(img2 - img)

'''
