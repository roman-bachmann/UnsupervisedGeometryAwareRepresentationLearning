import matplotlib.pyplot as plt

import sys
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

from matplotlib.widgets import Slider, Button


class IgniteTestNVS(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)

        # load data
        device='cuda'
        if 1: # load small example data
            import pickle
            data_loader = pickle.load(open('examples/test_set.pickl',"rb"))
        else:
            data_loader = self.load_data_test(config_dict)
            # save example data
            if 0:
                import pickle
                IPython.embed()
                data_iterator = iter(data_loader)
                data_cach = [next(data_iterator) for i in range(10)]
                data_cach = tuple(data_cach)
                pickle.dump(data_cach, open('examples/test_set.pickl', "wb"))

        # load model
        model = self.load_network(config_dict)
        model = model.to(device)

        # get next image
        input_dict, label_dict = None, None
        data_iterator = iter(data_loader)
        def nextImage():
            nonlocal input_dict, label_dict
            input_dict, label_dict = next(data_iterator)
            input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().cuda()
        nextImage()

        # apply model on images
        output_dict = None
        def predict():
            nonlocal output_dict
            model.eval()
            with torch.no_grad():
                input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')
        predict()

        print(output_dict['mu_fg'])


if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTestNVS()
    ignite.run(config_dict_module.__file__, config_dict)
