import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    # mode: {'fg', '3d', 'both'}
    def __init__(self, config_dict, train=True, mode='both', sample_latent=True):
        if mode not in ['fg', '3d', 'both']:
            raise ValueError('Please set parameter \'mode\' to one of \{\'fg\', \'3d\', \'both\'\}.')
        self.sample_latent = sample_latent
        self.config_dict = config_dict
        self.mode = mode

        if train:
            data_folder = os.path.join(config_dict['network_path'], 'latent', 'data_train')
        else:
            data_folder = os.path.join(config_dict['network_path'], 'latent',  'data_test')

        # Load saved training data from disk
        if mode == 'fg' or mode == 'both':
            if config_dict['variational_fg']:
                self.z_mus_fg = torch.Tensor(np.load(os.path.join(data_folder, 'mus_fg.npy')))
                self.z_logvars_fg = torch.Tensor(np.load(os.path.join(data_folder, 'logvars_fg.npy')))
                self.n_samples = self.z_mus_fg.shape[0]
            else:
                self.z_fg = torch.Tensor(np.load(os.path.join(data_folder, 'latent_fg.npy')))
                self.n_samples = self.z_fg.shape[0]

        if mode == '3d' or mode == 'both':
            if config_dict['variational_3d']:
                self.z_mus_3d = torch.Tensor(np.load(os.path.join(data_folder, 'mus_3d.npy')))
                self.z_logvars_3d = torch.Tensor(np.load(os.path.join(data_folder, 'logvars_3d.npy')))
                self.z_mus_3d = self.z_mus_3d.reshape(self.z_mus_3d.shape[0], -1)
                self.z_logvars_3d = self.z_logvars_3d.reshape(self.z_logvars_3d.shape[0], -1)
                self.n_samples = self.z_mus_3d.shape[0]
            else:
                self.z_3d = torch.Tensor(np.load(os.path.join(data_folder, 'latent_3d.npy')))
                self.z_3d = self.z_3d.reshape(self.z_3d.shape[0], -1)
                self.n_samples = self.z_3d.shape[0]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def __getitem__(self, index):
        if (self.mode == 'fg' or self.mode == 'both'):
            if self.config_dict['variational_fg']:
                if self.sample_latent:
                    out_fg = self.reparameterize(self.z_mus_fg[index], self.z_logvars_fg[index])
                else:
                    out_fg = self.z_mus_fg[index]
            else:
                out_fg = self.z_fg[index]

        if (self.mode == '3d' or self.mode == 'both'):
            if self.config_dict['variational_3d']:
                if self.sample_latent:
                    out_3d = self.reparameterize(self.z_mus_3d[index], self.z_logvars_3d[index])
                else:
                    out_3d = self.z_mus_3d[index]
            else:
                out_3d = self.z_3d[index]

        if self.mode == 'fg':
            return out_fg
        elif self.mode == '3d':
            return out_3d
        elif self.mode == 'both':
            return torch.cat([out_fg, out_3d], dim=1)

    def __len__(self):
        return self.n_samples
