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
        batch_size = 1

        # load model
        model = self.load_network(config_dict)
        model = model.to('cuda')

        def tensor_to_npimg(torch_array):
            return np.swapaxes(np.swapaxes(torch_array.numpy(), 0, 2), 0, 1)

        def denormalize(np_array):
            return np_array * np.array(config_dict['img_std']) + np.array(config_dict['img_mean'])

        # extract image
        def tensor_to_img(output_tensor):
            output_img = tensor_to_npimg(output_tensor)
            output_img = denormalize(output_img)
            output_img = np.clip(output_img, 0, 1)
            return output_img

        def rotationMatrixXZY(theta, phi, psi):
            Ax = np.matrix([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
            Ay = np.matrix([[np.cos(phi), 0, -np.sin(phi)],
                            [0, 1, 0],
                            [np.sin(phi), 0, np.cos(phi)]])
            Az = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1], ])
            return Az * Ay * Ax

        input_bg = torch.ones(batch_size, 3, config_dict['inputDimension'], config_dict['inputDimension']).float().cuda()

        # Sample from standard normal distribution
        latent_fg, latent_3d = None, None
        def sample_fg():
            nonlocal latent_fg
            latent_fg = torch.randn(batch_size, config_dict['latent_fg']).cuda()
        def sample_3d():
            nonlocal latent_3d
            latent_3d = torch.randn(batch_size, config_dict['latent_3d']).view(batch_size,-1,3).cuda()
        sample_fg()
        sample_3d()

        # apply model on images
        output_img = None
        def predict():
            nonlocal output_img
            model.eval()
            with torch.no_grad():
                output_img = model.decoder(latent_fg, latent_3d, input_bg=input_bg)[0].cpu()
        predict()

        # init figure
        my_dpi = 400
        fig, ax_blank = plt.subplots(figsize=(5 * 650 / my_dpi, 5 * 300 / my_dpi))
        plt.axis('off')

        latent_fg_dim = 0
        latent_3d_dim = 0

        # output image
        ax_out_img = plt.axes([0.3, 0.05, 0.95, 0.89])
        ax_out_img.axis('off')
        im_pred = plt.imshow(tensor_to_img(output_img), animated=True)
        ax_out_img.set_title("Generated sample")

        # update figure with new data
        def update_figure():
            # images
            im_pred.set_array(tensor_to_img(output_img))
            # flush drawings
            fig.canvas.draw_idle()

        def update_fg_slider():
            nonlocal latent_fg_dim, latent_fg
            slider_latent_fg_value.set_val(latent_fg[0,latent_fg_dim])

        def update_3d_slider():
            nonlocal latent_3d_dim, latent_3d
            dim1 = latent_3d_dim // 3
            dim2 = latent_3d_dim % 3
            slider_latent_3d_value.set_val(latent_3d[0,dim1,dim2])

        def update_rotation(event):
            rot = slider_yaw_glob.val
            external_rotation_global = torch.from_numpy(rotationMatrixXZY(theta=0, phi=0, psi=rot)).float().cuda()
            external_rotation_global = external_rotation_global.view(1,3,3).expand( (batch_size, 3, 3) )
            nonlocal latent_3d
            latent_3d = torch.bmm(latent_3d, external_rotation_global.transpose(1,2))
            predict()
            update_figure()

        # Button to sample FG from N(0,1)
        ax_sample_fg_normal = plt.axes([0.08, 0.65, 0.2, 0.1])
        button_sample_fg_normal = Button(ax_sample_fg_normal, 'Sample FG ~ N(0,1)', color='lightgray', hovercolor='0.975')
        def sampleFgNormalButtonPressed(event):
            sample_fg()
            predict()
            update_figure()
            update_fg_slider()
        button_sample_fg_normal.on_clicked(sampleFgNormalButtonPressed)

        # Button to sample 3D from N(0,1)
        ax_sample_3d_normal = plt.axes([0.08, 0.25, 0.2, 0.1])
        button_sample_3d_normal = Button(ax_sample_3d_normal, 'Sample 3D ~ N(0,1)', color='lightgray', hovercolor='0.975')
        def sample3dNormalButtonPressed(event):
            sample_3d()
            predict()
            update_figure()
            update_3d_slider()
        button_sample_3d_normal.on_clicked(sample3dNormalButtonPressed)

        # Slider to select FG latent dimension
        ax_latent_fg_selector = plt.axes([0.08, 0.9, 0.40, 0.03], facecolor='lightgray')
        slider_latent_fg_selector = Slider(ax_latent_fg_selector, 'FG dim', 0, config_dict['latent_fg']-1, valinit=0, valstep=1, valfmt='%1.0f')
        def update_latent_fg_dim(event):
            nonlocal latent_fg_dim
            latent_fg_dim = int(slider_latent_fg_selector.val)
            update_fg_slider()
        slider_latent_fg_selector.on_changed(update_latent_fg_dim)

        # Slider to select 3D latent dimension
        ax_latent_3d_selector = plt.axes([0.08, 0.5, 0.40, 0.03], facecolor='lightgray')
        slider_latent_3d_selector = Slider(ax_latent_3d_selector, '3D dim', 0, config_dict['latent_3d']-1, valinit=0, valstep=1, valfmt='%1.0f')
        def update_latent_3d_dim(event):
            nonlocal latent_3d_dim
            latent_3d_dim = int(slider_latent_3d_selector.val)
            update_3d_slider()
        slider_latent_3d_selector.on_changed(update_latent_3d_dim)

        # Slider to modify FG latent dimension value
        ax_latent_fg_value = plt.axes([0.08, 0.8, 0.40, 0.03], facecolor='lightgray')
        slider_latent_fg_value = Slider(ax_latent_fg_value, 'FG val', -3, 3, valinit=0, valfmt='%1.2f')
        def update_latent_fg_value(event):
            latent_fg[0,latent_fg_dim] = slider_latent_fg_value.val
            predict()
            update_figure()
        slider_latent_fg_value.on_changed(update_latent_fg_value)

        # Slider to modify 3D latent dimension value
        ax_latent_3d_value = plt.axes([0.08, 0.4, 0.40, 0.03], facecolor='lightgray')
        slider_latent_3d_value = Slider(ax_latent_3d_value, '3D val', -3, 3, valinit=0, valfmt='%1.2f')
        def update_latent_3d_value(event):
            dim1 = latent_3d_dim // 3
            dim2 = latent_3d_dim % 3
            latent_3d[0,dim1,dim2] = slider_latent_fg_value.val
            predict()
            update_figure()
        slider_latent_3d_value.on_changed(update_latent_3d_value)

        # Slider to modify rotation
        ax_yaw_glob = plt.axes([0.08, 0.1, 0.40, 0.03], facecolor='lightgray')
        slider_range = 2 * np.pi
        slider_yaw_glob = Slider(ax_yaw_glob, 'Yaw', -slider_range, slider_range, valinit=0)
        slider_yaw_glob.on_changed(update_rotation)
        plt.show()

if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTestNVS()
    ignite.run(config_dict_module.__file__, config_dict)