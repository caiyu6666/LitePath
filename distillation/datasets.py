import torch
import json
import os
import cv2
from PIL import Image
from torch.utils.data import Sampler
import lmdb
import numpy as np
import hashlib
import zlib


class FeatureExtractDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform_dict):
        super(FeatureExtractDataset, self).__init__()

        self.image_paths = self.get_all_files(root)
        self.transform_dict = transform_dict
        print(self.transform_dict)

        try:
            self.image_patch = cv2.imread(self.try_fast_disk(self.image_paths[0]))[..., ::-1]
        except:
            raise RuntimeError()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        p = self.image_paths[index]
        new_p = self.try_fast_disk(p)
        try:
            img = Image.open(new_p)
            images = {k: transform(img) for k, transform in self.transform_dict.items()}
        except:
            img = Image.fromarray(self.image_patch.copy())
            images = {k: transform(img) for k, transform in self.transform_dict.items()}

        key = hashlib.md5(p.encode('utf-8')).digest()
        return key, images

    @staticmethod
    def get_all_files(root):
        paths = []
        with open(root, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                paths += [os.path.join(k, name) for name in v]
        return paths

    @staticmethod
    def try_fast_disk(p: str):
        new_p = p.replace('/project/vcompath/storage/Pathology', '/scratch/vcompath')
        if os.path.exists(new_p):
            return new_p
        else:
            return p


class FastPathologyDataset(torch.utils.data.Dataset):
    teacher_features_path = {'virchow2': '/scratch/timnhaoprj/ycaibt/pretrain_features/virchow2.lmdb',
                             'h-optimus-1': '/scratch/timnhaoprj/ycaibt/pretrain_features/h-optimus-1.lmdb',
                             'uni2': '/scratch/slcompath/ycaibt/pretrain_features/uni2.lmdb'}

    def __init__(self, root, transform, teachers):
        super(FastPathologyDataset, self).__init__()
        self.lmdb_path = root
        self.teachers = teachers
        self.transform = transform
        # print(self.transform)

        self.path_env = None
        self.path_txn = None
        self.envs = None
        self.txns = None

        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            self.length = int.from_bytes(txn.get(b'__len__'), 'big')
            compressed = txn.get(int(0).to_bytes(4, 'big'))
            self.image_path_0 = zlib.decompress(compressed).decode('utf-8')
        env.close()
        try:
            self.image_patch = cv2.imread(self.try_fast_disk(self.image_path_0))[..., ::-1]
        except:
            raise RuntimeError()

    def _init_envs(self):
        if self.envs is None:
            self.envs = {name: lmdb.open(self.teacher_features_path[name], readonly=True, lock=False, readahead=False)
                         for name in self.teachers}
            self.txns = {name: env.begin(write=False) for name, env in self.envs.items()}
            self.path_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
            self.path_txn = self.path_env.begin(write=False)

    def get_image_path(self, index):
        compressed = self.path_txn.get(index.to_bytes(4, 'big'))
        path = zlib.decompress(compressed).decode('utf-8')
        return path

    def read_lmdb_feature(self, image_path):
        key = hashlib.md5(image_path.encode('utf-8')).digest()
        buf_dict = {name: txn.get(key) for name, txn in self.txns.items()}
        feature_dict = {name: np.frombuffer(buf, dtype=np.float32).copy() for name, buf in buf_dict.items()}
        return feature_dict

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._init_envs()

        p = self.get_image_path(index)
        new_p = self.try_fast_disk(p)
        if not os.path.exists(new_p):
            new_p = p
        try:
            # img = cv2.imread(p)[..., ::-1] #BGR to RGB
            # img = Image.fromarray(img)
            img = Image.open(new_p)
            img = self.transform(img)
        except:
            img = Image.fromarray(self.image_patch.copy())
            img = self.transform(img)
            p = self.image_path_0

        teacher_features = self.read_lmdb_feature(p)

        data = {'image': img, 'teacher_features': teacher_features}
        return data

    @staticmethod
    def try_fast_disk(p: str):
        new_p = p.replace('/project/vcompath/storage/Pathology', '/scratch/vcompath')
        if os.path.exists(new_p):
            return new_p
        else:
            return p


class ResumeDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, start_index=0, shuffle=True, seed=0, drop_last=False):
        super(ResumeDistributedSampler, self).__init__()

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        # self.start_index = start_index

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = len(self.dataset) // self.num_replicas
        else:
            self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.num_replicas))

        self.start_index = start_index % self.num_samples
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        if not self.drop_last:
            indices += indices[:(self.total_size - len(indices))]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # resume: skip the first `start_index` samples
        if self.start_index > 0:
            indices = indices[self.start_index:]

        return iter(indices)

    def __len__(self):
        return self.num_samples - self.start_index
