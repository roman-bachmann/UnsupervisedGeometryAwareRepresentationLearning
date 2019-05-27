import torch
import train_encodeDecode


class DreamAppearance(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
        config_dict['use_second_stage'] = False

        # load data
        device='cuda'
        if 0: # load small example data
            import pickle
            data_loader = pickle.load(open('examples/test_set.pickl',"rb"))
        else:
            # data_loader = self.load_data_train(config_dict)
            data_loader = self.load_data_test(config_dict)

        # load model
        model = self.load_network(config_dict)
        model = model.to(device)

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

        # Rotation matrix
        external_rotation_global = torch.from_numpy(np.eye(3)).float().cuda().view(1,3,3)

        # get next image
        input_dict, label_dict, input_img, input_bg = None, None, None, None
        data_iterator = iter(data_loader)
        def nextImage():
            nonlocal input_dict, label_dict
            try:
                input_dict, label_dict = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                input_dict, label_dict = next(data_iterator)

            input_img, input_bg = input_dict['img_crop'], input_dict['bg_crop']
        nextImage()


        # apply model on images to get first prediction
        output_dict_orig, latent_fg_orig, latent_3d_orig = None, None, None
        latent_fg, latent_3d = None, None
        def initial_prediction():
            nonlocal output_dict, latent_fg_orig, latent_3d_orig, latent_fg, latent_3d
            model.eval()
            input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().cuda()
            with torch.no_grad():
                input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict_orig = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')
                latent_fg_orig, latent_3d_orig = output_dict_orig['latent_fg'], output_dict_orig['latent_3d']
                latent_fg, latent_3d = latent_fg_orig.copy(), latent_3d_orig.copy()
        initial_prediction()

        output_img_dream = None
        def subseq_prediction():
            nonlocal output_img_dream
            model.eval()
            with torch.no_grad():
                latent_3d_rot = torch.bmm(latent_3d, external_rotation_global.transpose(1,2))
                output_img_dream = nvs_model.decoder(latent_fg, latent_3d_rot, input_bg=input_bg)[0].detach().cpu()
        subseq_prediction()



        # init figure
        my_dpi = 400
        fig, ax_blank = plt.subplots(figsize=(5 * 900 / my_dpi, 5 * 380 / my_dpi))
        plt.axis('off')

        # Left input image
        ax_in_img = plt.axes([-0.20, 0.18, 0.75, 0.75])
        ax_in_img.axis('off')
        im_input = plt.imshow(tensor_to_img(input_dict['img_crop'][0]), animated=True)
        ax_in_img.set_title("Input")

        # Original output image
        ax_out_img_orig = plt.axes([0.125, 0.18, 0.75, 0.75])
        ax_out_img_orig.axis('off')
        im_pred_orig = plt.imshow(tensor_to_img(output_dict_orig['img_crop'][0]), animated=True)
        ax_out_img_orig.set_title("Original prediction")

        # Dreamed output image
        ax_out_img_dream = plt.axes([0.45, 0.18, 0.75, 0.75])
        ax_out_img_dream.axis('off')
        im_pred_dream = plt.imshow(tensor_to_img(output_img_dream), animated=True)
        ax_out_img_dream.set_title("Dreamed prediction")

        # Update figure with new data
        def update_figure():
            # images
            im_input.set_array(tensor_to_img(input_dict['img_crop'][0]))
            im_pred_orig.set_array(tensor_to_img(output_dict_orig['img_crop'][0]))
            im_pred_dream.set_array(tensor_to_img(output_img_dream))
            # flush drawings
            fig.canvas.draw_idle()

        def update_rotation(event):
            rot = slider_yaw_glob.val
            nonlocal external_rotation_global
            external_rotation_global = torch.from_numpy(rotationMatrixXZY(theta=0, phi=rot, psi=0)).float().cuda()
            external_rotation_global = external_rotation_global.view(1,3,3).expand( (batch_size, 3, 3) )
            predict()
            update_figure()

        # Slider to modify rotation
        ax_yaw_glob = plt.axes([0.35, 0.02, 0.30, 0.03], facecolor='lightgray')
        slider_range = 2 * np.pi
        slider_yaw_glob = Slider(ax_yaw_glob, 'Yaw', -slider_range, slider_range, valinit=0)
        slider_yaw_glob.on_changed(update_rotation)

        # Button to get next image
        ax_next = plt.axes([0.05, 0.1, 0.15, 0.04])
        button_next = Button(ax_next, 'Next image', color='lightgray', hovercolor='0.975')
        def nextButtonPressed(event):
            nextImage()
            initial_prediction()
            subseq_prediction()
            update_figure()
        button_next.on_clicked(nextButtonPressed)


        plt.show()


if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = DreamAppearance()
    ignite.run(config_dict_module.__file__, config_dict)
