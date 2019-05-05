import os
import numpy as np
import itertools

from matplotlib.image import imread


def infinite_shuffle(iterable):
    while True:
        np.random.shuffle(iterable)
        return itertools.chain(item for item in iterable)

    
def load_img(path):
    return imread(path)


def identity(arg):
    return arg

    
class Batch():
    def __init__(self, sam_dir, lab_dir, test_frac=0.2,
                 batch_size=1, partial_size=1):
        self.sam_dir = sam_dir
        self.lab_dir = lab_dir
        self.test_frac = test_frac
        self.batch_size = batch_size
        self.partial_size = partial_size
        self.split = self.update_split()

    def update_split(self):
        split = np.random.choice([0, 1], size=(self.sam_count),
                                 p=[1 - self.test_frac, self.test_frac])
        return split.astype(bool)

    @property
    def num_samples(self):
        return len(self.img_dirs)

    @property
    def sam_list(self):
        return os.listdir(self.sam_dir)

    @property
    def lab_list(self):
        return os.listdir(self.lab_dir)

    @property
    def sam_paths(self):
        return [os.path.join(self.sam_dir, fname)
                for fname in self.sam_list]

    @property
    def lab_paths(self):
        return [os.path.join(self.lab_dir, fname)
                for fname in self.lab_list]

    @property
    def sam_count(self):
        return len(self.sam_list)
    
    @property
    def lab_count(self):
        return len(self.lab_list)
    
    @property
    def train_sam(self):
        return np.array(self.sam_paths)[~self.split].tolist()

    @property
    def train_lab(self):
        return np.array(self.lab_paths)[~self.split].tolist()

    @property
    def test_sam(self):
        return np.array(self.sam_paths)[self.split].tolist()

    @property
    def test_lab(self):
        return np.array(self.lab_paths)[self.split].tolist()

    def gen_pairs(self, sam, lab):
        return list(zip(sam, lab))

    def gen_partial(self, sam, lab, fun=identity):
        while True:
            shuffle = infinite_shuffle(self.gen_pairs(sam, lab))
            yield itertools.chain([fun(item) for item in next(shuffle)]
                                  for idx in range(self.partial_size))

    def gen_batch(self, sam, lab, fun=identity):
        while True:
            partial = self.gen_partial(sam, lab, fun=fun)
            yield [next(next(partial)) for idx in range(self.batch_size)]
        
        
class ImageBatch(Batch):
    def __init__(self, sam_dir, lab_dir, test_frac=0.2,
                 batch_size=1, partial_size=1):
        Batch.__init__(self, sam_dir, lab_dir, test_frac,
                       batch_size, partial_size)
        
    def img_batch(self, sam, lab, fun=load_img):
        while True:
            yield next(self.gen_batch(sam, lab, fun=fun))
        
    def train_batch(self):
        return self.img_batch(self.train_sam, self.train_lab)

    def test_batch(self):
        return self.img_batch(self.test_sam, self.test_lab)
