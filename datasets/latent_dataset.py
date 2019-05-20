import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    # mode: {'fg', '3d', 'both'}
    def __init__(self, data_folder, mode='both', sample_latent=True):
        self.sample_latent = sample_latent
        # Load saved training data from disk
        if mode == 'fg' or mode == 'both':
            z_mus_fg = torch.Tensor(np.load(os.path.join(data_folder, 'mus_fg.npy')))
            z_logvars_fg = torch.Tensor(np.load(os.path.join(data_folder, 'logvars_fg.npy')))
        if mode == '3d' or mode == 'both':
            z_mus_3d = torch.Tensor(np.load(os.path.join(data_folder, 'mus_3d.npy')))
            z_logvars_3d = torch.Tensor(np.load(os.path.join(data_folder, 'logvars_3d.npy')))
            z_mus_3d = z_mus_3d.reshape(z_mus_3d.shape[0], -1)
            z_logvars_3d = z_logvars_3d.reshape(z_logvars_3d.shape[0], -1)

        # Add together if both are chosen
        if mode == 'fg':
            self.z_mus, self.z_logvars = z_mus_fg, z_logvars_fg
        elif mode == '3d':
            self.z_mus, self.z_logvars = z_mus_3d, z_logvars_3d
        elif mode == 'both':
            self.z_mus = torch.cat([z_mus_fg, z_mus_3d], dim=1)
            self.z_logvars = torch.cat([z_logvars_fg, z_logvars_3d], dim=1)
        else:
            raise ValueError('Please set parameter which to one of \{fg, 3d, both\}.')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def __getitem__(self, index):
        if self.sample_latent:
            return self.reparameterize(self.z_mus[index], self.z_logvars[index])
        else:
            return self.z_mus[index]

    def __len__(self):
        return len(self.z_mus)
