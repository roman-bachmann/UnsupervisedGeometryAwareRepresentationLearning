import matplotlib.pyplot as plt

import sys, os
import torch
import numpy as np
import numpy.linalg as la
import IPython
from tqdm import tqdm

from utils import io as utils_io
from utils import datasets as utils_data
from utils import plotting as utils_plt
from utils import skeleton as utils_skel

import train_encodeDecode
from ignite._utils import convert_tensor
from ignite.engine import Events


class LatentExtractor(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict, save_train=False):

        # HACK
        if save_train:
            # config_dict['useSubjectBatches'] = 0
            # config_dict['useCamBatches'] = 0
            # config_dict['useCamBatches'] = 0
            batch_size = 4
        else:
            batch_size = 2

        # load data
        device='cuda'
        if save_train:
            data_loader = self.load_data_train(config_dict)
        else:
            data_loader = self.load_data_test(config_dict)
        print('Number of images:', len(data_loader))

        # load model
        model = self.load_network(config_dict)
        model = model.to(device)
        model.eval()

        # Create empty tensors to save latent spaces in
        nb_samples = len(data_loader) * batch_size
        if config_dict['variational_fg'] and config_dict['latent_fg'] > 0:
            mus_fg = torch.zeros(nb_samples, config_dict['latent_fg'])
            logvars_fg = torch.zeros(nb_samples, config_dict['latent_fg'])
        latent_fg = torch.zeros(nb_samples, config_dict['latent_fg'])
        if config_dict['variational_3d']:
            mus_3d = torch.zeros(nb_samples, config_dict['latent_3d']//3, 3)
            logvars_3d = torch.zeros(nb_samples, config_dict['latent_3d']//3, 3)
        latent_3d = torch.zeros(nb_samples, config_dict['latent_3d']//3, 3)

        # Iterate over all images and save their latent spaces
        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=len(data_loader), desc='Saving latent')
            for iter, (input_dict, label_dict) in enumerate(data_loader):
                input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')

                idx_lo, idx_hi = iter*batch_size, (iter+1)*batch_size

                if config_dict['variational_fg'] and config_dict['latent_fg'] > 0:
                    mus_fg[idx_lo:idx_hi] = output_dict['mu_fg']
                    logvars_fg[idx_lo:idx_hi] = output_dict['logvar_fg']
                latent_fg[idx_lo:idx_hi] = output_dict['latent_fg']

                if config_dict['variational_3d']:
                    mus_3d[idx_lo:idx_hi] = output_dict['mu_3d']
                    logvars_3d[idx_lo:idx_hi] = output_dict['logvar_3d']
                latent_3d[idx_lo:idx_hi] = output_dict['latent_3d']

                pbar.update(1)
            pbar.close()

        config_dict['network_path'] = config_dict.get('network_path', './output/latent')
        if save_train:
            save_path = os.path.join(config_dict['network_path'], 'latent', 'data_train')
        else:
            save_path = os.path.join(config_dict['network_path'], 'latent', 'data_test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if config_dict['variational_fg'] and config_dict['latent_fg'] > 0:
            np.save(os.path.join(save_path,'mus_fg.npy'), mus_fg.numpy())
            np.save(os.path.join(save_path,'logvars_fg.npy'), logvars_fg.numpy())
        np.save(os.path.join(save_path,'latent_fg.npy'), latent_fg.numpy())
        if config_dict['variational_3d']:
            np.save(os.path.join(save_path,'mus_3d.npy'), mus_3d.numpy())
            np.save(os.path.join(save_path,'logvars_3d.npy'), logvars_3d.numpy())
        np.save(os.path.join(save_path,'latent_3d.npy'), latent_3d.numpy())


if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    latent_extractor = LatentExtractor()
    # Save both training and testing image latent spaces
    latent_extractor.run(config_dict, save_train=False)
    latent_extractor.run(config_dict, save_train=True)
