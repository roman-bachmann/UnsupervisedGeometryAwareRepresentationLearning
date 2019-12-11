import os
import csv
import numpy as np
import torch
import torchvision
import torch.utils.data as data

import h5py
import imageio
import cv2

import random
from random import shuffle

import IPython

import numpy.linalg as la

from utils import datasets as utils_data
from tqdm import tqdm
import pickle


class CollectedDataset(data.Dataset):
    def __init__(self, data_folder,
                 input_types, label_types,
                 useSubjectBatches=0, useCamBatches=0,
                 randomize=True,
                 mean=(0.485, 0.456, 0.406),
                 stdDev= (0.229, 0.224, 0.225),
                 useSequentialFrames=0,
                 augment_hue=False
                 ):
        args = list(locals().items())
        # save function arguments
        for arg,val in args:
            setattr(self, arg, val)

        class Image256toTensor(object):
            def __call__(self, pic):
                img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
                img = img.div(255)
                return img

            def __repr__(self):
                return self.__class__.__name__ + '()'

        self.transform_in = torchvision.transforms.Compose([
            Image256toTensor(), #torchvision.transforms.ToTensor() the torchvision one behaved differently for different pytorch versions, hence the custom one..
            torchvision.transforms.Normalize(self.mean, self.stdDev)
        ])

        # build cam/subject datastructure
        h5_label_file = h5py.File(data_folder+'/labels.h5', 'r')
        print('Loading h5 label file to memory')
        self.label_dict = {key: np.array(value) for key,value in h5_label_file.items()}
        all_keys_name = data_folder+'/all_keys.pickl'
        sequence_keys_name = data_folder+'/sequence_keys.pickl'
        camsets_name = data_folder+'/camsets.pickl'
        print('Done loading h5 label file')
        if os.path.exists(sequence_keys_name):
            print('Loading sequence-subject-cam association from pickle files {}.'.format(sequence_keys_name))
            self.all_keys = pickle.load( open(all_keys_name, "rb" ) )
            self.sequence_keys = pickle.load( open(sequence_keys_name, "rb" ) )
            self.camsets = pickle.load( open(camsets_name, "rb" ) )
            print('Done loading sequence association.')
        else:
            print('Establishing sequence association. Available labels:',list(h5_label_file.keys()))
            all_keys = set()
            camsets = {}
            sequence_keys = {}
            data_length = len(h5_label_file['frame'])
            with tqdm(total=data_length) as pbar:
                for index in range(data_length):
                    pbar.update(1)
                    sub_i = int(h5_label_file['subj'][index].item())
                    cam_i = int(h5_label_file['cam'][index].item())
                    seq_i = int(h5_label_file['seq'][index].item())
                    frame_i = int(h5_label_file['frame'][index].item())

                    key = (sub_i,seq_i,frame_i)
                    if key not in camsets:
                        camsets[key] = {}
                    camsets[key][cam_i] = index

                    # only add if accumulated enough cameras
                    if len(camsets[key])>=useCamBatches:
                        all_keys.add(key)

                        if seq_i not in sequence_keys:
                            sequence_keys[seq_i] = set()
                        sequence_keys[seq_i].add(key)

            self.all_keys = list(all_keys)
            self.camsets = camsets
            self.sequence_keys = {seq: list(keyset) for seq,keyset in sequence_keys.items()}
            pickle.dump(self.all_keys, open(all_keys_name, "wb" ) )
            pickle.dump(self.sequence_keys, open(sequence_keys_name, "wb" ) )
            pickle.dump(self.camsets, open(camsets_name, "wb" ) )
            print("DictDataset: Done initializing, listed {} camsets ({} frames) and {} sequences".format(self.__len__(), self.__len__()*useCamBatches, len(sequence_keys)))

    def __len__(self):
        if self.useCamBatches > 0:
            return len(self.all_keys)
        else:
            return len(self.label_dict['frame'])

    def getLocalIndices(self, index):
        input_dict = {}
        cam = int(self.label_dict['cam'][index].item())
        seq = int(self.label_dict['seq'][index].item())
        frame = int(self.label_dict['frame'][index].item())
        return cam, seq, frame, index

    def getItemIntern(self, cam, seq, frame, index, dataset_idx):
        def getImageName(key):
            return self.data_folder+'/seq_{:03d}/cam_{:02d}/{}_{:06d}.png'.format(seq,cam,key,frame)

        def knuths_hash(i):
            # Hashes indexes uniformly to pseudo-random hues
            # Alpha = next odd integer to 2^8 * (-1 + sqrt(5)) / 2
            alpha = 159
            return (i*alpha) % (2**8)

        def hue_shift(img, shift=0):
            '''
            Shifts the hue component of an image by a given factor.

            :param img: The original image (H x W x C)
            :param shift: Shift hue by this factor. Between 0 and 255.
            :return: Hue shifted image
            '''
            shift = shift % 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:,:,0] = (img[:,:,0] + shift) % 255
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        def loadImage(name):
            # if not os.path.exists(name):
            #     print('Image not available ({})'.format(name))
            #     raise Exception('Image not available')
            img = imageio.imread(name)
            if self.augment_hue:
                # Shift hue by pseudo-random index
                img = hue_shift(img, shift=knuths_hash(dataset_idx))
            return np.array(self.transform_in(img), dtype='float32')

        def loadData(types):
            new_dict = {}
            for key in types:
                if key in ['img_crop','bg_crop']:
                    new_dict[key] = loadImage(getImageName(key)) #np.array(self.transform_in(imageio.imread(getImageName(key))), dtype='float32')
                else:
                    new_dict[key] = np.array(self.label_dict[key][index], dtype='float32')
            return new_dict

        return loadData(self.input_types), loadData(self.label_types)



    def __getitem__(self, index):
        if self.useSequentialFrames > 1:
            frame_skip = 1  # 6:1 sec, since anyways subsampled at 5 frames
            #cam, seq, frame = getLocalIndices(index)

            # ensure that a sequence is not exceeded
            for i in range(self.useSequentialFrames):
                cam_skip = 4
                index_range = list(range(index+i, index+i + self.useSequentialFrames * frame_skip*cam_skip, frame_skip*cam_skip))
                #print('index_range',index_range,'seq',[int(self.label_dict['seq'][i]) for i in index_range])
                #print('index_range',index_range,'frame',[int(self.label_dict['frame'][i]) for i in index_range])
                #print('index_range',index_range,'cam',[int(self.label_dict['cam'][i]) for i in index_range])
                if len(self.label_dict['seq'])>index_range[-1] and self.label_dict['seq'][index_range[0]] == self.label_dict['seq'][index_range[-1]]:
                    break

            # collect single results
            single_examples = [self.getItemIntern(*self.getLocalIndices(i), dataset_idx=index) for i in index_range]
            collated_examples = utils_data.default_collate_with_string(single_examples) #accumulate list of single frame results
            return collated_examples
        if self.useCamBatches > 0:
            key = self.all_keys[index]
            def getCamSubbatch(key):
                camset = self.camsets[key]
                cam_keys = list(camset.keys())
                assert self.useCamBatches <= len(cam_keys)
                if self.randomize:
                    random.Random(500).shuffle(cam_keys)
                cam_keys_shuffled = cam_keys[:self.useCamBatches]
                return [self.getItemIntern(*self.getLocalIndices(camset[cami]), dataset_idx=index) for cami in cam_keys_shuffled]

            single_examples = getCamSubbatch(key)
            if self.useSubjectBatches > 0:
                #subj = key[0]
                seqi = key[1]
                potential_keys = self.sequence_keys[seqi]
                key_other = potential_keys[np.random.randint(len(potential_keys))]
                single_examples = single_examples + getCamSubbatch(key_other)

            collated_examples = utils_data.default_collate_with_string(single_examples) #accumulate list of single frame results
            return collated_examples
        else:
            return self.getItemIntern(*self.getLocalIndices(index, dataset_idx=index))

if __name__ == '__main__':
    dataset = CollectedDataset(
                 data_folder='/cvlabdata1/home/rbachman/DataSets/H36M/H36M-MultiView-test',
                 input_types=['img_crop','extrinsic_rot','extrinsic_rot_inv','bg_crop'], label_types=['img_crop','3D','bounding_box_cam','intrinsic_crop','extrinsic_rot','extrinsic_rot_inv'],
                 useSubjectBatches=2, useCamBatches=2,
                 randomize=True, augment_hue=True)

    # for i in range(len(dataset)):
    #     data = dataset.__getitem__(i)
    #     IPython.embed()

    import matplotlib.pyplot as plt

    def tensor_to_npimg(torch_array):
        return np.swapaxes(np.swapaxes(torch_array.numpy(), 0, 2), 0, 1)

    def denormalize(np_array):
        return np_array * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406),)

    # extract image
    def tensor_to_img(output_tensor):
        output_img = tensor_to_npimg(output_tensor)
        output_img = denormalize(output_img)
        output_img = np.clip(output_img, 0, 1)
        return output_img

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False, drop_last=True, collate_fn=utils_data.default_collate_with_string)
    trainloader = utils_data.PostFlattenInputSubbatchTensor(trainloader)
    data_iter = iter(trainloader)

    for i in range(10):

        # x,y = dataset.__getitem__(i)
        x,y = data_iter.__next__()

        for img in x['img_crop']:
            fig, ax_blank = plt.subplots(figsize=(5 * 400 / 400, 5 * 400 / 400))
            plt.axis('off')
            ax_in_img = plt.axes([0, 0, 1, 1])
            ax_in_img.axis('off')
            im_input = plt.imshow(tensor_to_img(img), animated=False)
            fig.canvas.draw_idle()
            plt.show()
            plt.close()
