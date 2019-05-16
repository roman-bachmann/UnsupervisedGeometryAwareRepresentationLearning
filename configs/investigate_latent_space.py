import matplotlib.pyplot as plt

import sys, os
import torch
import numpy as np
import numpy.linalg as la
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import plotting as utils_plt
from utils import skeleton as utils_skel

import train_encodeDecode
from ignite._utils import convert_tensor
from ignite.engine import Events


class IgniteTestNVS(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict):
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)

        # load data
        device='cuda'
        if 0: # load small example data
            import pickle
            data_loader = pickle.load(open('examples/test_set.pickl',"rb"))
        else:
            data_loader = self.load_data_test(config_dict)
        print('Number test images:', len(data_loader))

        # load model
        model = self.load_network(config_dict)
        model = model.to(device)
        model.eval()

        mus_fg = torch.zeros(len(data_loader), config_dict['latent_fg'])
        logvars_fg = torch.zeros(len(data_loader), config_dict['latent_fg'])
        mus_3d = torch.zeros(len(data_loader), config_dict['latent_3d']//3, 3)
        logvars_3d = torch.zeros(len(data_loader), config_dict['latent_3d']//3, 3)

        with torch.no_grad():
            for iter, (input_dict, label_dict) in enumerate(data_loader):
                input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().cuda()
                input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')

                mus_fg[iter] = output_dict['mu_fg'][0]
                logvars_fg[iter] = output_dict['logvar_fg'][0]
                mus_3d[iter] = output_dict['mu_3d'][0]
                logvars_3d[iter] = output_dict['logvar_3d'][0]

        save_path = './output/latent/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(mus_fg, os.path.join(save_path,'mus_fg.pt'))
        torch.save(logvars_fg, os.path.join(save_path,'logvars_fg.pt'))
        torch.save(mus_3d, os.path.join(save_path,'mus_3d.pt'))
        torch.save(logvars_3d, os.path.join(save_path,'logvars_3d.pt'))



if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTestNVS()
    ignite.run(config_dict)
