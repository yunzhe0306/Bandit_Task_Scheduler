
import pickle
from torch.utils.data import Dataset

#
import torch
import torchvision
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset
from sklearn.preprocessing import normalize


class MiniImagenet(Dataset):

    def __init__(self, args, mode, tau=0.7, alpha=3):
        super(MiniImagenet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.noise = args.noise
        self.mode = mode
        self.alpha = alpha
        self.tau = tau
        self.threshold = self.noise
        self.noisy_task_ratio = args.noisy_task_ratio
        # 
        self.random_state = np.random.RandomState(np.random.RandomState(self.args.seed).randint(1, 10000))
        self.copy_noisy_task_flag = args.copy_noisy_task_flag
        print("!!! --- Noisy task ratio: ", args.limit_classes, args.noisy_task_ratio, args.noise)

        #
        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
            self.data_file_val = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            if self.noise > 0:
                print("noise:", self.noise)
                self.noise_matrix = np.diag([1 - self.noise] * self.nb_classes)
                for i in range(self.nb_classes):
                    for j in range(self.nb_classes):
                        if j == i:
                            continue
                        self.noise_matrix[i][j] = self.noise / (self.nb_classes - 1)
                print(self.noise_matrix)
        elif mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            self.noise = 0
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)
            self.noise = 0

        self.data = pickle.load(open(self.data_file, 'rb'))

        if mode == 'train':
            self.data_val = pickle.load(open(self.data_file_val, 'rb'))
            self.data_val = torch.tensor(np.transpose(self.data_val / np.float32(255), (0, 1, 4, 2, 3)))
            self.classes_idx_val = np.arange(self.data_val.shape[0])

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))

        if mode == 'train':
            if args.limit_data:
                if args.limit_classes == 16:
                    chosen_classes_idx = [24, 63, 43, 34, 23, 50, 42, 19, 30, 29, 54, 35, 0, 21, 26, 45]
                elif args.limit_classes == 32:
                    chosen_classes_idx = [36, 62, 54, 5, 9, 41, 1, 6, 2, 0, 27, 55, 12, 22, 15, 3, 34,
                                          49, 59, 11, 16, 35, 32, 18, 17, 43, 21, 42, 28, 60, 61, 37]
                elif args.limit_classes == 48:
                    chosen_classes_idx = [7, 22, 61, 18, 14, 30, 46, 4, 32, 6, 15, 48, 5, 25, 41, 54, 42,
                                          60, 58, 29, 53, 27, 50, 55, 19, 45, 52, 9, 44, 13, 28, 63, 62, 57,
                                          56, 33, 20, 47, 3, 12, 39, 17, 51, 49, 31, 24, 34, 43]
                elif args.limit_classes >= 64:
                    chosen_classes_idx = np.arange(64)
                else:
                    raise NotImplementedError
                self.noisy_tasks = np.random.choice(chosen_classes_idx,
                                                    size=int(len(chosen_classes_idx) * self.noisy_task_ratio),
                                                    replace=False)
                print(" ---------------- Noisy tasks: ", self.noisy_tasks)
                self.data = self.data[chosen_classes_idx]
            else:
                if self.copy_noisy_task_flag:
                    this_noisy_tasks = np.load('./noisy_task_indices/noisy_tasks_FOR_ALL_{}.npy'.format(str(args.noisy_task_ratio)))
                    print(" ---------------- Noisy tasks: ", this_noisy_tasks)
                    noisy_task_num = len(this_noisy_tasks)
                    self.data = torch.cat([self.data[this_noisy_tasks], self.data], dim=0)
                    self.noisy_tasks = np.arange(noisy_task_num, dtype=int)
                else:
                    self.noisy_tasks = self.random_state.choice(np.arange(self.data.shape[0]),
                                                        size=int(self.data.shape[0] * self.noisy_task_ratio),
                                                        replace=False) 
                    print(" ---------------- Noisy tasks: ", self.noisy_tasks)

        
        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])

        #
        self.skewed_task_distribution_flag = args.skewed_task_distribution_flag
        if self.skewed_task_distribution_flag:
            self.skewed_sampling_prob = np.concatenate([0.07 * np.ones(5), 
                                                        (0.40 / (self.data.shape[0] - 10)) * np.ones(self.data.shape[0] - 10),
                                                        0.05 * np.ones(5)])
            self.skewed_sampling_prob = normalize(self.skewed_sampling_prob.reshape(1, -1), norm='l1').reshape(-1, )
        print()

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):

        support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        noisy_or_not = torch.zeros(self.args.meta_batch_size)
        task_classes = np.zeros([self.args.meta_batch_size, self.nb_classes])

        for meta_batch_id in range(self.args.meta_batch_size):

            if self.skewed_task_distribution_flag:
                self.choose_classes = self.random_state.choice(self.classes_idx, size=self.nb_classes, replace=False, p=self.skewed_sampling_prob)
            else:
                self.choose_classes = self.random_state.choice(self.classes_idx, size=self.nb_classes, replace=False)
            task_classes[meta_batch_id, :] = self.choose_classes

            noise = self.noise

            if not (noise > 0 and self.mode == 'train'):
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = \
                        self.data[self.choose_classes[j], choose_samples[:self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            else:
                noisy_or_not[meta_batch_id] = 1
                x = torch.zeros((self.set_size + self.query_size, 3, 84, 84))
                y = torch.zeros(self.set_size + self.query_size)
                # randomly sample 80 pictures
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    x[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] \
                        = self.data[self.choose_classes[j], choose_samples, ...]
                    y[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] \
                        = j

                noisy_y = y.clone().detach()
                y_np = y.numpy()
                for i in range(len(y)):
                    this_class_label_id = self.choose_classes[int(y_np[i])]
                    if this_class_label_id in self.noisy_tasks:
                        noisy_y[i] = self.random_state.choice(self.nb_classes, p=self.noise_matrix[int(y[i])])

                support_idxes = []
                for j in range(self.nb_classes):
                    idx = np.where(noisy_y == j)[0]
                    self.random_state.shuffle(idx)
                    idx = idx[:self.k_shot]
                    support_idxes.append(idx)

                support_idxes = np.concatenate(support_idxes)
                query_idxes = np.setdiff1d(np.arange(len(x)), support_idxes)
                support_idxes = np.concatenate([support_idxes, query_idxes[self.query_size:]])
                query_idxes = query_idxes[:self.query_size]

                support_x[meta_batch_id] = x[support_idxes]
                support_y[meta_batch_id] = y[support_idxes]
                query_x[meta_batch_id] = x[query_idxes]
                query_y[meta_batch_id] = noisy_y[query_idxes]

            if not self.args.mix:
                support_sample = np.arange(self.set_size)
                query_sample = np.arange(self.query_size)
                self.random_state.shuffle(support_sample)
                self.random_state.shuffle(query_sample)

                support_x[meta_batch_id] = support_x[meta_batch_id][support_sample]
                support_y[meta_batch_id] = support_y[meta_batch_id][support_sample]
                query_x[meta_batch_id] = query_x[meta_batch_id][query_sample]
                query_y[meta_batch_id] = query_y[meta_batch_id][query_sample]

        if self.mode == 'train':
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), noisy_or_not, task_classes
        else:
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), task_classes


class MiniImagenet_GCP(Dataset):

    def __init__(self, args, mode, tau=0.7, alpha=3):
        super(MiniImagenet_GCP, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.noise = args.noise
        self.mode = mode
        self.alpha = alpha
        self.tau = tau
        self.threshold = self.noise
        self.noisy_task_ratio = args.noisy_task_ratio
        # 
        self.random_state = np.random.RandomState(np.random.RandomState(self.args.seed).randint(1, 10000))
        self.copy_noisy_task_flag = args.copy_noisy_task_flag
        print("!!! --- Noisy task ratio: ", args.limit_classes, args.noisy_task_ratio, args.noise)

        #
        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
            self.data_file_val = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            if self.noise > 0:
                print("noise:", self.noise)
                self.noise_matrix = np.diag([1 - self.noise] * self.nb_classes)
                for i in range(self.nb_classes):
                    for j in range(self.nb_classes):
                        if j == i:
                            continue
                        self.noise_matrix[i][j] = self.noise / (self.nb_classes - 1)
                print(self.noise_matrix)
        elif mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            self.noise = 0
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)
            self.noise = 0

        self.data = pickle.load(open(self.data_file, 'rb'))

        if mode == 'train':
            self.data_val = pickle.load(open(self.data_file_val, 'rb'))
            self.data_val = torch.tensor(np.transpose(self.data_val / np.float32(255), (0, 1, 4, 2, 3)))
            self.classes_idx_val = np.arange(self.data_val.shape[0])

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))

        if mode == 'train':
            if args.limit_data:
                if args.limit_classes == 16:
                    chosen_classes_idx = [24, 63, 43, 34, 23, 50, 42, 19, 30, 29, 54, 35, 0, 21, 26, 45]
                elif args.limit_classes == 32:
                    chosen_classes_idx = [36, 62, 54, 5, 9, 41, 1, 6, 2, 0, 27, 55, 12, 22, 15, 3, 34,
                                          49, 59, 11, 16, 35, 32, 18, 17, 43, 21, 42, 28, 60, 61, 37]
                elif args.limit_classes == 48:
                    chosen_classes_idx = [7, 22, 61, 18, 14, 30, 46, 4, 32, 6, 15, 48, 5, 25, 41, 54, 42,
                                          60, 58, 29, 53, 27, 50, 55, 19, 45, 52, 9, 44, 13, 28, 63, 62, 57,
                                          56, 33, 20, 47, 3, 12, 39, 17, 51, 49, 31, 24, 34, 43]
                elif args.limit_classes >= 64:
                    chosen_classes_idx = np.arange(64)
                else:
                    raise NotImplementedError
                self.noisy_tasks = np.random.choice(chosen_classes_idx,
                                                    size=int(len(chosen_classes_idx) * self.noisy_task_ratio),
                                                    replace=False)
                print(" ---------------- Noisy tasks: ", self.noisy_tasks)
                self.data = self.data[chosen_classes_idx]
            else:
                if self.copy_noisy_task_flag:
                    this_noisy_tasks = np.load('./noisy_task_indices/noisy_tasks_FOR_ALL_{}.npy'.format(str(args.noisy_task_ratio)))
                    print(" ---------------- Noisy tasks: ", this_noisy_tasks)
                    noisy_task_num = len(this_noisy_tasks)
                    self.data = torch.cat([self.data[this_noisy_tasks], self.data], dim=0)
                    self.noisy_tasks = np.arange(noisy_task_num, dtype=int)
                else:
                    self.noisy_tasks = self.random_state.choice(np.arange(self.data.shape[0]),
                                                        size=int(self.data.shape[0] * self.noisy_task_ratio),
                                                        replace=False) 
                    print(" ---------------- Noisy tasks: ", self.noisy_tasks)

        
        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])

        #
        self.skewed_task_distribution_flag = args.skewed_task_distribution_flag
        if self.skewed_task_distribution_flag:
            self.skewed_sampling_prob = np.concatenate([0.07 * np.ones(5), 
                                                        (0.40 / (self.data.shape[0] - 10)) * np.ones(self.data.shape[0] - 10),
                                                        0.05 * np.ones(5)])
            self.skewed_sampling_prob = normalize(self.skewed_sampling_prob.reshape(1, -1), norm='l1').reshape(-1, )
        print()

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):

        support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        noisy_or_not = torch.zeros(self.args.meta_batch_size)
        task_classes = np.zeros([self.args.meta_batch_size, self.nb_classes])

        for meta_batch_id in range(self.args.meta_batch_size):

            if self.skewed_task_distribution_flag:
                self.choose_classes = self.random_state.choice(self.classes_idx, size=self.nb_classes, replace=False, p=self.skewed_sampling_prob)
            else:
                self.choose_classes = self.random_state.choice(self.classes_idx, size=self.nb_classes, replace=False)
            task_classes[meta_batch_id, :] = self.choose_classes

            noise = self.noise

            if not (noise > 0 and self.mode == 'train'):
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            else:
                noisy_or_not[meta_batch_id] = 1
                x = torch.zeros((self.set_size + self.query_size, 3, 84, 84))
                y = torch.zeros(self.set_size + self.query_size)
                # randomly sample 80 pictures
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    x[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] \
                        = self.data[self.choose_classes[j], choose_samples, ...]
                    y[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] \
                        = j

                noisy_y = y.clone().detach()
                y_np = y.numpy()
                for i in range(len(y)):
                    this_class_label_id = self.choose_classes[int(y_np[i])]
                    if this_class_label_id in self.noisy_tasks:
                        noisy_y[i] = self.random_state.choice(self.nb_classes, p=self.noise_matrix[int(y[i])])

                support_idxes = []
                for j in range(self.nb_classes):
                    idx = np.where(noisy_y == j)[0]
                    self.random_state.shuffle(idx)
                    idx = idx[:self.k_shot]
                    support_idxes.append(idx)

                support_idxes = np.concatenate(support_idxes)
                query_idxes = np.setdiff1d(np.arange(len(x)), support_idxes)
                support_idxes = np.concatenate([support_idxes, query_idxes[self.query_size:]])
                query_idxes = query_idxes[:self.query_size]

                support_x[meta_batch_id] = x[support_idxes]
                support_y[meta_batch_id] = y[support_idxes]
                query_x[meta_batch_id] = x[query_idxes]
                query_y[meta_batch_id] = noisy_y[query_idxes]

            if not self.args.mix:
                support_sample = np.arange(self.set_size)
                query_sample = np.arange(self.query_size)
                self.random_state.shuffle(support_sample)
                self.random_state.shuffle(query_sample)

                support_x[meta_batch_id] = support_x[meta_batch_id][support_sample]
                support_y[meta_batch_id] = support_y[meta_batch_id][support_sample]
                query_x[meta_batch_id] = query_x[meta_batch_id][query_sample]
                query_y[meta_batch_id] = query_y[meta_batch_id][query_sample]

        if self.mode == 'train':
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), noisy_or_not, \
                   task_classes
        else:
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), task_classes



class MiniImagenet_Noisy_Task(Dataset):
    def __init__(self, args, mode, diverse=True):
        super(MiniImagenet_Noisy_Task, self).__init__()
        self.args = args
        self.noise = args.noise
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.train_train_count = 0
        self.train_val_count = 0
        self.diverse = diverse
        self.mode = mode
        self.threshold = self.noise
        self.noisy_task_ratio = args.noisy_task_ratio
        self.GCP_model = None
        # 
        self.random_state = np.random.RandomState(np.random.RandomState(self.args.seed).randint(1, 10000))
        self.copy_noisy_task_flag = args.copy_noisy_task_flag
        print("!!! --- Noisy task ratio: ", args.limit_classes, args.noisy_task_ratio, args.noise)

        #
        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
            self.data_file_val = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            if self.noise > 0:
                print("noise:", self.noise)
                self.noise_matrix = np.diag([1 - self.noise] * self.nb_classes)
                for i in range(self.nb_classes):
                    for j in range(self.nb_classes):
                        if j == i:
                            continue
                        self.noise_matrix[i][j] = self.noise / (self.nb_classes - 1)
                print(self.noise_matrix)
        elif mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            self.noise = 0
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)
            self.noise = 0

        self.data = pickle.load(open(self.data_file, 'rb'))

        if mode == 'train':
            self.data_val = pickle.load(open(self.data_file_val, 'rb'))
            self.data_val = torch.tensor(np.transpose(self.data_val / np.float32(255), (0, 1, 4, 2, 3)))
            self.classes_idx_val = np.arange(self.data_val.shape[0])

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))

        if mode == 'train':
            if args.limit_data:
                if args.limit_classes == 16:
                    chosen_classes_idx = [24, 63, 43, 34, 23, 50, 42, 19, 30, 29, 54, 35, 0, 21, 26, 45]
                elif args.limit_classes == 32:
                    chosen_classes_idx = [36, 62, 54, 5, 9, 41, 1, 6, 2, 0, 27, 55, 12, 22, 15, 3, 34,
                                          49, 59, 11, 16, 35, 32, 18, 17, 43, 21, 42, 28, 60, 61, 37]
                elif args.limit_classes == 48:
                    chosen_classes_idx = [7, 22, 61, 18, 14, 30, 46, 4, 32, 6, 15, 48, 5, 25, 41, 54, 42,
                                          60, 58, 29, 53, 27, 50, 55, 19, 45, 52, 9, 44, 13, 28, 63, 62, 57,
                                          56, 33, 20, 47, 3, 12, 39, 17, 51, 49, 31, 24, 34, 43]
                elif args.limit_classes >= 64:
                    chosen_classes_idx = np.arange(64)
                else:
                    raise NotImplementedError
                self.noisy_tasks = np.random.choice(chosen_classes_idx,
                                                    size=int(len(chosen_classes_idx) * self.noisy_task_ratio),
                                                    replace=False)
                print(" ---------------- Noisy tasks: ", self.noisy_tasks)
                self.data = self.data[chosen_classes_idx]
            else:
                if self.copy_noisy_task_flag:
                    this_noisy_tasks = np.load('./noisy_task_indices/noisy_tasks_FOR_ALL_{}.npy'.format(str(args.noisy_task_ratio)))
                    print(" ---------------- Noisy tasks: ", this_noisy_tasks)
                    noisy_task_num = len(this_noisy_tasks)
                    self.data = torch.cat([self.data[this_noisy_tasks], self.data], dim=0)
                    self.noisy_tasks = np.arange(noisy_task_num, dtype=int)
                else:
                    self.noisy_tasks = self.random_state.choice(np.arange(self.data.shape[0]),
                                                        size=int(self.data.shape[0] * self.noisy_task_ratio),
                                                        replace=False) 
                    print(" ---------------- Noisy tasks: ", self.noisy_tasks)
     
        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])

        #
        self.skewed_task_distribution_flag = args.skewed_task_distribution_flag
        if self.skewed_task_distribution_flag:
            self.skewed_sampling_prob = np.concatenate([0.07 * np.ones(5), 
                                                        (0.40 / (self.data.shape[0] - 10)) * np.ones(self.data.shape[0] - 10),
                                                        0.05 * np.ones(5)])
            self.skewed_sampling_prob = normalize(self.skewed_sampling_prob.reshape(1, -1), norm='l1').reshape(-1, )

        print()

    def get_sampled_data(self, data, index, setting):

        if setting == 'train_train' and self.args.sampling_method in ['ATS', 'GCP']:
            task_num = self.args.buffer_size
        elif setting == 'train_train' and self.args.sampling_method == 'Bandit':
            task_num = self.args.buffer_size
        else:
            task_num = self.args.meta_batch_size

        support_x = torch.FloatTensor(torch.zeros((task_num, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((task_num, self.query_size, 3, 84, 84)))

        support_y = np.zeros([task_num, self.set_size])
        query_y = np.zeros([task_num, self.query_size])

        noisy_or_not = torch.zeros(task_num)
        task_classes = np.zeros([task_num, self.nb_classes])

        for meta_batch_id in range(task_num):
            if self.mode == 'train':
                if setting == 'train_train':
                    if self.skewed_task_distribution_flag:
                        self.choose_classes = self.random_state.choice(self.classes_idx, size=self.nb_classes, replace=False, p=self.skewed_sampling_prob)
                    else:
                        if not self.diverse or (self.train_train_count + 1) * self.nb_classes > len(self.classes_idx):
                            self.train_train_count = 0
                            self.random_state.shuffle(self.classes_idx)
                        self.choose_classes = self.classes_idx[self.train_train_count * self.nb_classes:
                                                            (self.train_train_count + 1) * self.nb_classes]
                        self.train_train_count += 1

                elif setting == 'train_val':
                    if not self.diverse or (self.train_val_count + 1) * self.nb_classes > len(self.classes_idx_val):
                        self.train_val_count = 0
                        self.random_state.shuffle(self.classes_idx_val)
                    self.choose_classes = self.classes_idx_val[self.train_val_count * self.nb_classes:
                                                               (self.train_val_count + 1) * self.nb_classes]
                    self.train_val_count += 1
            else:
                self.choose_classes = self.random_state.choice(self.classes_idx, size=self.nb_classes, replace=False)

            task_classes[meta_batch_id, :] = self.choose_classes
            noise = self.noise
            # if self.random_state.random() > self.threshold:
            #     noise = 0

            if noise > 0:
                noisy_or_not[meta_batch_id] = 1

            # --------------------------------------------------------------------------------
            if not (noise > 0 and setting == 'train_train'):
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]

                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = \
                        data[self.choose_classes[j], choose_samples[:self.k_shot], ...]

                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = \
                        data[self.choose_classes[j], choose_samples[self.k_shot:], ...]

                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            # Noise > 0 + Training -----------------------------------------------------------
            else:
                x = torch.zeros((self.set_size + self.query_size, 3, 84, 84))
                y = torch.zeros(self.set_size + self.query_size)
                # random sample 80 pictures
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    x[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] \
                        = data[self.choose_classes[j], choose_samples, ...]
                    y[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] = j

                # Random sample noisy labels
                noisy_y = y.clone().detach()
                y_np = y.numpy()
                for i in range(len(y)):
                    this_class_label_id = self.choose_classes[int(y_np[i])]
                    if this_class_label_id in self.noisy_tasks:
                        noisy_y[i] = self.random_state.choice(self.nb_classes, p=self.noise_matrix[int(y[i])])

                support_idxes = []
                for j in range(self.nb_classes):
                    idx = np.where(noisy_y == j)[0]
                    self.random_state.shuffle(idx)
                    idx = idx[:self.k_shot]
                    support_idxes.append(idx)

                support_idxes = np.concatenate(support_idxes)
                query_idxes = np.setdiff1d(np.arange(len(x)), support_idxes)
                support_idxes = np.concatenate([support_idxes, query_idxes[self.query_size:]])
                query_idxes = query_idxes[:self.query_size]

                support_x[meta_batch_id] = x[support_idxes]
                support_y[meta_batch_id] = y[support_idxes]
                query_x[meta_batch_id] = x[query_idxes]
                query_y[meta_batch_id] = noisy_y[query_idxes]

            support_sample = np.arange(self.set_size)
            query_sample = np.arange(self.query_size)
            self.random_state.shuffle(support_sample)
            self.random_state.shuffle(query_sample)

            support_x[meta_batch_id] = support_x[meta_batch_id][support_sample]
            support_y[meta_batch_id] = support_y[meta_batch_id][support_sample]
            query_x[meta_batch_id] = query_x[meta_batch_id][query_sample]
            query_y[meta_batch_id] = query_y[meta_batch_id][query_sample]
        if setting == 'train_train':
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), noisy_or_not, \
                   task_classes
        else:
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), task_classes

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.GCP_model is None:
                support_x, support_y, query_x, query_y, noisy_or_not, task_classes = \
                    self.get_sampled_data(self.data, index, setting='train_train')
                support_x_val, support_y_val, query_x_val, query_y_val, task_classes_val = \
                    self.get_sampled_data(self.data, index, setting='train_val')
                return support_x, support_y, query_x, query_y, support_x_val, support_y_val, query_x_val, query_y_val, \
                       noisy_or_not, task_classes
            else:
                support_x, support_y, query_x, query_y, noisy_or_not, task_classes = \
                    self.get_sampled_data(self.data, index, setting='train_train')
                return support_x, support_y, query_x, query_y, noisy_or_not, task_classes
        else:
            support_x, support_y, query_x, query_y, task_classes = self.get_sampled_data(self.data, index, setting='test')

            return support_x, support_y, query_x, query_y, task_classes


class MiniImagenet_Error_Analysis(Dataset):
    def __init__(self, args, mode, noisy_tasks=None):
        super(MiniImagenet_Error_Analysis, self).__init__()
        self.args = args
        self.noise = args.noise
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.train_train_count = 0
        self.train_val_count = 0
        self.mode = mode
        self.threshold = self.noise
        self.noisy_task_ratio = args.noisy_task_ratio
        self.target_class_counter = 0
        # 
        self.random_state = np.random.RandomState(np.random.RandomState(self.args.seed).randint(1, 10000))
        self.copy_noisy_task_flag = args.copy_noisy_task_flag
        print("!!! --- Noisy task ratio: ", args.limit_classes, args.noisy_task_ratio, args.noise)

        #
        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
            self.data_file_val = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            if self.noise > 0:
                print("noise:", self.noise)
                self.noise_matrix = np.diag([1 - self.noise] * self.nb_classes)
                for i in range(self.nb_classes):
                    for j in range(self.nb_classes):
                        if j == i:
                            continue
                        self.noise_matrix[i][j] = self.noise / (self.nb_classes - 1)
                print(self.noise_matrix)
        elif mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            self.noise = 0
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)
            self.noise = 0

        self.data = pickle.load(open(self.data_file, 'rb'))
        #
        if mode == 'train':
            self.data_val = pickle.load(open(self.data_file_val, 'rb'))
            self.data_val = torch.tensor(np.transpose(self.data_val / np.float32(255), (0, 1, 4, 2, 3)))
            self.classes_idx_val = np.arange(self.data_val.shape[0])

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))

        if mode == 'train':
            if args.limit_data:
                if args.limit_classes == 16:
                    chosen_classes_idx = [24, 63, 43, 34, 23, 50, 42, 19, 30, 29, 54, 35, 0, 21, 26, 45]
                elif args.limit_classes == 32:
                    chosen_classes_idx = [36, 62, 54, 5, 9, 41, 1, 6, 2, 0, 27, 55, 12, 22, 15, 3, 34,
                                          49, 59, 11, 16, 35, 32, 18, 17, 43, 21, 42, 28, 60, 61, 37]
                elif args.limit_classes == 48:
                    chosen_classes_idx = [7, 22, 61, 18, 14, 30, 46, 4, 32, 6, 15, 48, 5, 25, 41, 54, 42,
                                          60, 58, 29, 53, 27, 50, 55, 19, 45, 52, 9, 44, 13, 28, 63, 62, 57,
                                          56, 33, 20, 47, 3, 12, 39, 17, 51, 49, 31, 24, 34, 43]
                elif args.limit_classes >= 64:
                    chosen_classes_idx = np.arange(64)
                else:
                    raise NotImplementedError
                self.noisy_tasks = np.random.choice(chosen_classes_idx,
                                                    size=int(len(chosen_classes_idx) * self.noisy_task_ratio),
                                                    replace=False)
                print(" ---------------- Noisy tasks: ", self.noisy_tasks)
                self.data = self.data[chosen_classes_idx]
            else:
                if self.copy_noisy_task_flag:
                    this_noisy_tasks = noisy_tasks
                    print(" ---------------- Noisy tasks: ", this_noisy_tasks)
                    noisy_task_num = len(this_noisy_tasks)
                    self.data = torch.cat([self.data[this_noisy_tasks], self.data], dim=0)
                    self.noisy_tasks = np.arange(noisy_task_num, dtype=int)
                else:
                    self.noisy_tasks = noisy_tasks
                    print(" ---------------- Noisy tasks: ", self.noisy_tasks)
        
        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])

    def get_sampled_data(self, data, setting):

        task_num = 10

        support_x = torch.FloatTensor(torch.zeros((task_num, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((task_num, self.query_size, 3, 84, 84)))

        support_y = np.zeros([task_num, self.set_size])
        query_y = np.zeros([task_num, self.query_size])

        noisy_or_not = torch.zeros(task_num)
        task_classes = np.zeros([task_num, self.nb_classes])

        if setting == 'train':
            self.target_class = self.target_class_counter % 64
        elif setting == 'test':
            self.target_class = self.target_class_counter % 20
        else:
            quit(1)

        for meta_batch_id in range(task_num):
            if self.target_class is None:
                self.choose_classes = self.random_state.choice(self.classes_idx, size=self.nb_classes, replace=False)
            else:
                self.choose_classes = np.zeros([self.nb_classes, ])
                class_idx = np.concatenate([np.arange(self.target_class),
                                            np.arange(start=self.target_class+1, stop=self.data.shape[0])])
                randomly_chosen_classes = self.random_state.choice(class_idx, size=self.nb_classes-1, replace=False)
                self.choose_classes[:self.nb_classes-1] = randomly_chosen_classes
                self.choose_classes[self.nb_classes-1] = int(self.target_class)
            self.choose_classes = self.choose_classes.astype(int)
            task_classes[meta_batch_id, :] = self.choose_classes
            noise = self.noise

            if noise > 0 and meta_batch_id in self.noisy_tasks:
                noisy_or_not[meta_batch_id] = 1

            # --------------------------------------------------------------------------------
            if not (noise > 0 and setting == 'train'):
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]

                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = \
                        data[self.choose_classes[j], choose_samples[:self.k_shot], ...]

                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = \
                        data[self.choose_classes[j], choose_samples[self.k_shot:], ...]

                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            # Noise > 0 + Training -----------------------------------------------------------
            else:
                x = torch.zeros((self.set_size + self.query_size, 3, 84, 84))
                y = torch.zeros(self.set_size + self.query_size)
                # random sample 80 pictures
                for j in range(self.nb_classes):
                    self.random_state.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    x[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] \
                        = data[self.choose_classes[j], choose_samples, ...]
                    y[j * (self.k_shot + self.k_query): (j + 1) * (self.k_shot + self.k_query)] = j

                # Random sample noisy labels
                noisy_y = y.clone().detach()
                y_np = y.numpy()
                for i in range(len(y)):
                    this_class_label_id = self.choose_classes[int(y_np[i])]
                    if this_class_label_id in self.noisy_tasks:
                        noisy_y[i] = self.random_state.choice(self.nb_classes, p=self.noise_matrix[int(y[i])])

                support_idxes = []
                for j in range(self.nb_classes):
                    idx = np.where(noisy_y == j)[0]
                    self.random_state.shuffle(idx)
                    idx = idx[:self.k_shot]
                    support_idxes.append(idx)

                support_idxes = np.concatenate(support_idxes)
                query_idxes = np.setdiff1d(np.arange(len(x)), support_idxes)
                support_idxes = np.concatenate([support_idxes, query_idxes[self.query_size:]])
                query_idxes = query_idxes[:self.query_size]

                support_x[meta_batch_id] = x[support_idxes]
                support_y[meta_batch_id] = y[support_idxes]
                query_x[meta_batch_id] = x[query_idxes]
                query_y[meta_batch_id] = noisy_y[query_idxes]

            support_sample = np.arange(self.set_size)
            query_sample = np.arange(self.query_size)
            self.random_state.shuffle(support_sample)
            self.random_state.shuffle(query_sample)

            support_x[meta_batch_id] = support_x[meta_batch_id][support_sample]
            support_y[meta_batch_id] = support_y[meta_batch_id][support_sample]
            query_x[meta_batch_id] = query_x[meta_batch_id][query_sample]
            query_y[meta_batch_id] = query_y[meta_batch_id][query_sample]
        self.target_class_counter += 1
        if self.mode == 'train':
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), noisy_or_not, task_classes
        else:
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), task_classes

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size




