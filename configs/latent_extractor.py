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
from torch.utils.data import Dataset, DataLoader
from models.vae import VAE
from datasets.latent_dataset import LatentDataset

import train_encodeDecode
from ignite._utils import convert_tensor
from ignite.engine import Events


class LatentExtractor(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict, save_train=False):
        # TODO: Find out why 4 and 2
        if save_train:
            batch_size = 4 * config_dict['batch_size_train']
        else:
            batch_size = 2 * config_dict['batch_size_test']

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
            for i, (input_dict, label_dict) in enumerate(data_loader):
                input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')

                idx_lo, idx_hi = i*batch_size, (i+1)*batch_size

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


class LatentExtractorStage2(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict, save_train=False):

        # # load data
        # config_dict['network_path'] = config_dict.get('network_path', './output/latent')
        # if save_train:
        #     save_path = os.path.join(config_dict['network_path'], 'latent', 'data_train')
        # else:
        #     save_path = os.path.join(config_dict['network_path'], 'latent', 'data_test')
        # device='cuda'
        # mus_fg = torch.Tensor(np.load(os.path.join(save_path,'mus_fg.npy'))).to(device)
        # mus_3d = torch.Tensor(np.load(os.path.join(save_path,'mus_3d.npy')).to(device)
        # logvars_fg = torch.Tensor(np.load(os.path.join(save_path,'logvars_fg.npy')).to(device)
        # logvars_3d = torch.Tensor(np.load(os.path.join(save_path,'logvars_3d.npy')).to(device)
        #
        # nb_samples = len(mus_fg)
        # print('Number of datapoints:', len(mus_fg))

        batch_size = 2
        device='cuda'

        z_dataset_fg = LatentDataset(config_dict, train=save_train,  mode='fg', sample_latent=True)
        z_dataset_3d  = LatentDataset(config_dict, train=save_train, mode='3d', sample_latent=True)
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True}
        loader_fg = DataLoader(z_dataset_fg, batch_size=batch_size, shuffle=False, **kwargs)
        loader_3d  = DataLoader(z_dataset_3d, batch_size=batch_size, shuffle=False, **kwargs)
        print('Data loaded and datasets created')


        # load models
        vae_fg = VAE(input_dim=config_dict['latent_fg'], hidden_dim=config_dict['second_stage_hidden_dim'], latent_dim=config_dict['second_stage_latent_dim']).to(device)
        print('Nb params FG model:', sum(p.numel() for p in vae_fg.parameters() if p.requires_grad))
        model_path_fg = os.path.join(config_dict['network_path'], 'models', 'second_stage_vae_fg_ldim{}.pth'.format(config_dict['second_stage_latent_dim']))
        vae_fg.load_state_dict(torch.load(model_path_fg))
        vae_fg.eval()



        vae_3d = VAE(input_dim=config_dict['latent_3d'], hidden_dim=config_dict['second_stage_hidden_dim'], latent_dim=config_dict['second_stage_latent_dim']).to(device)
        print('Nb params 3D model:', sum(p.numel() for p in vae_3d.parameters() if p.requires_grad))
        model_path_3d = os.path.join(config_dict['network_path'], 'models', 'second_stage_vae_3d_ldim{}.pth'.format(config_dict['second_stage_latent_dim']))
        vae_3d.load_state_dict(torch.load(model_path_3d))
        vae_3d.eval()


        # Create empty tensors to save latent spaces in
        nb_samples_fg = len(loader_fg) * batch_size
        nb_samples_3d = len(loader_3d) * batch_size
        w_mus_fg = torch.zeros(nb_samples_fg, config_dict['second_stage_latent_dim'])
        w_logvars_fg = torch.zeros(nb_samples_fg, config_dict['second_stage_latent_dim'])
        w_mus_3d = torch.zeros(nb_samples_3d, config_dict['second_stage_latent_dim'])
        w_logvars_3d = torch.zeros(nb_samples_3d, config_dict['second_stage_latent_dim'])


        with torch.no_grad():
            pbar = tqdm(total=len(loader_fg), desc='Saving latent FG')
            for i, data in enumerate(loader_fg):
                data = data.to(device)
                _, mu, logvar, _ = vae_fg(data)
                idx_lo, idx_hi = i*batch_size, (i+1)*batch_size
                w_mus_fg[idx_lo:idx_hi] = mu
                w_logvars_fg[idx_lo:idx_hi] = logvar
                pbar.update(1)
            pbar.close()

            pbar = tqdm(total=len(loader_3d), desc='Saving latent 3D')
            for i, data in enumerate(loader_3d):
                data = data.to(device)
                _, mu, logvar, _ = vae_3d(data)
                idx_lo, idx_hi = i*batch_size, (i+1)*batch_size
                w_mus_3d[idx_lo:idx_hi] = mu
                w_logvars_3d[idx_lo:idx_hi] = logvar
                pbar.update(1)
            pbar.close()


        config_dict['network_path'] = config_dict.get('network_path', './output/latent')
        if save_train:
            save_path = os.path.join(config_dict['network_path'], 'latent', 'data_train')
        else:
            save_path = os.path.join(config_dict['network_path'], 'latent', 'data_test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(os.path.join(save_path,'w_mus_fg.npy'), w_mus_fg.numpy())
        np.save(os.path.join(save_path,'w_logvars_fg.npy'), w_logvars_fg.numpy())
        np.save(os.path.join(save_path,'w_mus_3d.npy'), w_mus_3d.numpy())
        np.save(os.path.join(save_path,'w_logvars_3d.npy'), w_logvars_3d.numpy())


if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    # latent_extractor = LatentExtractor()
    latent_extractor = LatentExtractorStage2()
    # Save both training and testing image latent spaces
    # latent_extractor.run(config_dict, save_train=False)
    latent_extractor.run(config_dict, save_train=True)
