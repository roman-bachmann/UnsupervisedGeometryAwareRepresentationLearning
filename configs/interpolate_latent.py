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
from models.vae import VAE

import train_encodeDecode
from ignite._utils import convert_tensor
from ignite.engine import Events

from matplotlib.widgets import Slider, Button

def save_img(img_array, name):
    path = '/cvlabdata1/home/rbachman/imgs_out/'
    import scipy.misc
    scipy.misc.toimage(img_array, cmin=0.0, cmax=1.0).save(path + '{}.png'.format(name))

class IgniteTestNVS(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict, use_second_stage=False):
        batch_size = 1

        # load model
        device = 'cuda'
        nvs_model = self.load_network(config_dict).to(device)
        nvs_model.eval()

        if use_second_stage:
            vae_model_fg = VAE(input_dim=config_dict['latent_fg'], hidden_dim=512, latent_dim=30).to(device)
            model_path_fg = os.path.join(config_dict['network_path'], 'models', 'second_stage_vae_fg_ldim{}.pth'.format(config_dict['second_stage_latent_dim']))
            vae_model_fg.load_state_dict(torch.load(model_path_fg))
            vae_model_fg.eval()
            z_dim_fg = 30

            vae_model_3d = VAE(input_dim=config_dict['latent_3d'], hidden_dim=512, latent_dim=30).to(device)
            model_path_3d = os.path.join(config_dict['network_path'], 'models', 'second_stage_vae_3d_ldim{}.pth'.format(config_dict['second_stage_latent_dim']))
            vae_model_3d.load_state_dict(torch.load(model_path_3d))
            vae_model_3d.eval()
            z_dim_3d = 30
        else:
            z_dim_fg = config_dict['latent_fg']
            z_dim_3d = config_dict['latent_3d']

        if 0: # load small example data
            import pickle
            data_loader_left = pickle.load(open('examples/test_set.pickl',"rb"))
            data_loader_right = pickle.load(open('examples/test_set.pickl',"rb"))
        else:
            data_loader_left = self.load_data_test(config_dict)
            data_loader_right = self.load_data_test(config_dict)
            # data_loader_left = self.load_data_train(config_dict)
            # data_loader_right = self.load_data_train(config_dict)

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

        index_left = -1
        index_right = -1

        # get next left input image
        input_dict_left, label_dict_left = None, None
        data_iterator_left = iter(data_loader_left)
        def nextImageLeft():
            nonlocal input_dict_left, label_dict_left, data_iterator_left, index_left
            try:
                input_dict_left, label_dict_left = next(data_iterator_left)
            except StopIteration:
                data_iterator_left = iter(data_loader_left)
                input_dict_left, label_dict_left = next(data_iterator_left)
            input_dict_left['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().cuda()
            index_left += 1
            print('left {}, right {}'.format(index_left, index_right))
        nextImageLeft()

        # get next right input image
        input_dict_right, label_dict_right = None, None
        data_iterator_right = iter(data_loader_right)
        def nextImageRight():
            nonlocal input_dict_right, label_dict_right, data_iterator_right, index_right
            try:
                input_dict_right, label_dict_right = next(data_iterator_right)
            except StopIteration:
                data_iterator_right = iter(data_loader_right)
                input_dict_right, label_dict_right = next(data_iterator_right)
            input_dict_right['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().cuda()
            index_right += 1
            print('left {}, right {}'.format(index_left, index_right))
        nextImageRight()
        # nextImageRight()


        # White background. Could be replaced by arbitrary background
        input_bg = torch.ones(batch_size, 3, config_dict['inputDimension'], config_dict['inputDimension']).float().cuda()

        # Interpolation slider values
        alpha_fg = 0.5
        alpha_3d = 0.5

        # Rotation matrix
        external_rotation_global = torch.from_numpy(rotationMatrixXZY(theta=0, phi=0, psi=0)).float().cuda()
        external_rotation_global = external_rotation_global.view(1,3,3).expand( (batch_size, 3, 3) )

        # apply model on images
        latent_fg_left, latent_3d_left, latent_fg_right, latent_3d_right = None, None, None, None
        def predict_latent():
            nonlocal latent_fg_left, latent_3d_left, latent_fg_right, latent_3d_right
            nvs_model.eval()
            with torch.no_grad():
                input_dict_left_cuda, label_dict_left_cuda = utils_data.nestedDictToDevice((input_dict_left, label_dict_left), device=device)
                output_dict_left_cuda = nvs_model(input_dict_left_cuda)
                # output_dict_left = utils_data.nestedDictToDevice(output_dict_left_cuda, device='cpu')
                latent_fg_left = output_dict_left_cuda['latent_fg'][0].unsqueeze(0)
                latent_3d_left = output_dict_left_cuda['latent_3d'][0].unsqueeze(0)

                input_dict_right_cuda, label_dict_right_cuda = utils_data.nestedDictToDevice((input_dict_right, label_dict_right), device=device)
                output_dict_right_cuda = nvs_model(input_dict_right_cuda)
                # output_dict_right = utils_data.nestedDictToDevice(output_dict_right_cuda, device='cpu')
                latent_fg_right = output_dict_right_cuda['latent_fg'][0].unsqueeze(0)
                latent_3d_right = output_dict_right_cuda['latent_3d'][0].unsqueeze(0)

                if use_second_stage:
                    latent_fg_left, _ = vae_model_fg.encode(latent_fg_left)
                    latent_3d_left, _ = vae_model_3d.encode(latent_3d_left.view(batch_size,-1))
                    latent_fg_right, _ = vae_model_fg.encode(latent_fg_right)
                    latent_3d_right, _ = vae_model_3d.encode(latent_3d_right.view(batch_size,-1))


                # if use_second_stage:
                #     latent_fg_decoded = vae_model_fg.decode(latent_fg)
                #     latent_3d_decoded = vae_model_3d.decode(latent_3d.view(batch_size,-1)).view(batch_size,-1,3)
                #     latent_3d_decoded = torch.bmm(latent_3d_decoded, external_rotation_global.transpose(1,2))
                #     output_img = nvs_model.decoder(latent_fg_decoded, latent_3d_decoded, input_bg=input_bg)[0].cpu()
                # else:
                #     latent_3d_rot = torch.bmm(latent_3d, external_rotation_global.transpose(1,2))
                #     output_img = nvs_model.decoder(latent_fg, latent_3d_rot, input_bg=input_bg)[0].cpu()
        predict_latent()

        # Use previously predicted latent vectors from both images to create interpolated output
        output_img = None
        def predict_interpolated():
            nonlocal output_img

            latent_fg = latent_fg_left + alpha_fg * (latent_fg_right - latent_fg_left)
            latent_3d = latent_3d_left + alpha_3d * (latent_3d_right - latent_3d_left)

            if use_second_stage:
                latent_fg = vae_model_fg.decode(latent_fg)
                latent_3d = vae_model_3d.decode(latent_3d).view(batch_size,-1,3)

            latent_3d_rot = torch.bmm(latent_3d, external_rotation_global.transpose(1,2))
            output_img = nvs_model.decoder(latent_fg, latent_3d_rot, input_bg=input_bg)[0].detach().cpu()
        predict_interpolated()


        # init figure
        my_dpi = 400
        fig, ax_blank = plt.subplots(figsize=(5 * 900 / my_dpi, 5 * 380 / my_dpi))
        plt.axis('off')

        # Left input image
        ax_in_img_left = plt.axes([-0.20, 0.18, 0.75, 0.75])
        ax_in_img_left.axis('off')
        im_input_left = plt.imshow(tensor_to_img(input_dict_left['img_crop'][0]), animated=True)
        ax_in_img_left.set_title("Input 1")

        # Output image
        ax_out_img = plt.axes([0.125, 0.18, 0.75, 0.75])
        ax_out_img.axis('off')
        im_pred = plt.imshow(tensor_to_img(output_img), animated=True)
        ax_out_img.set_title("Interpolated")

        # Right input image
        ax_in_img_right = plt.axes([0.45, 0.18, 0.75, 0.75])
        ax_in_img_right.axis('off')
        im_input_right = plt.imshow(tensor_to_img(input_dict_right['img_crop'][0]), animated=True)
        ax_in_img_right.set_title("Input 2")

        # update figure with new data
        def update_figure():
            # images
            im_input_left.set_array(tensor_to_img(input_dict_left['img_crop'][0]))
            im_pred.set_array(tensor_to_img(output_img))
            im_input_right.set_array(tensor_to_img(input_dict_right['img_crop'][0]))
            # flush drawings
            fig.canvas.draw_idle()

        def update_rotation(event):
            rot = slider_yaw_glob.val
            # nonlocal external_rotation_global
            # external_rotation_global = torch.from_numpy(rotationMatrixXZY(theta=0, phi=rot, psi=0)).float().cuda()
            # external_rotation_global = external_rotation_global.view(1,3,3).expand( (batch_size, 3, 3) )
            # predict_interpolated()
            # update_figure()
            nonlocal alpha_3d, alpha_fg
            save_img(tensor_to_img(input_dict_left['img_crop'][0]), '{}_{}-in_left'.format(index_left, index_right))
            save_img(tensor_to_img(input_dict_right['img_crop'][0]), '{}_{}-in_right'.format(index_left, index_right))
            # alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            alpha_values = np.linspace(0,1,100)
            for idx, a in enumerate(alpha_values):
                print(idx)
                alpha_3d = float(a)
                alpha_fg = float(a)
                predict_interpolated()
                # save_img(tensor_to_img(output_img), '{}_{}-sample_{}'.format(index_left, index_right, idx))
                save_img(tensor_to_img(output_img), 'sample_{:06d}'.format(idx))
            update_figure()


        def update_alpha_fg(event):
            nonlocal alpha_fg
            alpha_fg = float(slider_alpha_fg_glob.val)
            predict_interpolated()
            update_figure()

        def update_alpha_3d(event):
            nonlocal alpha_3d
            alpha_3d = float(slider_alpha_3d_glob.val)
            predict_interpolated()
            update_figure()

        # Slider to modify interpolation alpha fg
        ax_alpha_fg_glob = plt.axes([0.35, 0.12, 0.30, 0.03], facecolor='lightgray')
        slider_alpha_fg_glob = Slider(ax_alpha_fg_glob, 'Alpha FG', 0, 1, valinit=0.5)
        slider_alpha_fg_glob.on_changed(update_alpha_fg)

        # Slider to modify interpolation alpha 3d
        ax_alpha_3d_glob = plt.axes([0.35, 0.07, 0.30, 0.03], facecolor='lightgray')
        slider_alpha_3d_glob = Slider(ax_alpha_3d_glob, 'Alpha 3D', 0, 1, valinit=0.5)
        slider_alpha_3d_glob.on_changed(update_alpha_3d)

        # Slider to modify rotation
        ax_yaw_glob = plt.axes([0.35, 0.02, 0.30, 0.03], facecolor='lightgray')
        slider_range = 2 * np.pi
        slider_yaw_glob = Slider(ax_yaw_glob, 'Yaw', -slider_range, slider_range, valinit=0)
        slider_yaw_glob.on_changed(update_rotation)

        ax_next_left = plt.axes([0.05, 0.1, 0.15, 0.04])
        button_next_left = Button(ax_next_left, 'Next image', color='lightgray', hovercolor='0.975')
        def nextButtonLeftPressed(event):
            nextImageLeft()
            predict_latent()
            predict_interpolated()
            update_figure()
        button_next_left.on_clicked(nextButtonLeftPressed)

        ax_next_right = plt.axes([0.8, 0.1, 0.15, 0.04])
        button_next_right = Button(ax_next_right, 'Next image', color='lightgray', hovercolor='0.975')
        def nextButtonRightPressed(event):
            nextImageRight()
            predict_latent()
            predict_interpolated()
            update_figure()
        button_next_right.on_clicked(nextButtonRightPressed)

        plt.show()

if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTestNVS()
    ignite.run(config_dict_module.__file__, config_dict, use_second_stage=True)
