import matplotlib.pyplot as plt

from datasets import collected_dataset
import sys, os, shutil

import numpy as np
#import pickle
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import training as utils_train
from utils import plot_dict_batch as utils_plot_batch
from utils import plotting as utils_plt

from models import unet_encode3D
from losses import generic as losses_generic
from losses import images as losses_images

import math
import torch
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models_tv

# for loading of training sets
#sys.path.insert(0,'../pytorch_human_reconstruction')
#import pytorch_datasets.dataset_factory as dataset_factory

import sys
sys.path.insert(0,'./ignite')
from ignite._utils import convert_tensor
from ignite.engine import Events

class IgniteTrainNVS:
    def run(self, config_dict_file, config_dict):
        config_dict_test = {k:v for k,v in config_dict.items()}
        config_dict_cams = {k:v for k,v in config_dict.items()}

        # no appearance fliping on test
        config_dict_test['useSubjectBatches'] = 0

        # ensure correct batch size after flattening
        config_dict_test    ['batch_size_test'] = config_dict['batch_size_test']//max(1,config_dict['useCamBatches'])
        config_dict_cams    ['batch_size_train'] = config_dict['batch_size_train']//max(1,config_dict['useSubjectBatches'])//max(1,config_dict['useCamBatches'])
        config_dict_cams    ['batch_size_test']  = config_dict['batch_size_test'] //max(1,config_dict['useSubjectBatches'])//max(1,config_dict['useCamBatches'])

        # some default values
        config_dict['implicit_rotation'] = config_dict.get('implicit_rotation', False)
        config_dict['skip_background'] = config_dict.get('skip_background', True)
        config_dict['loss_weight_pose3D'] = config_dict.get('loss_weight_pose3D', 0)
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)

        # create visualization windows
        try:
            import visdom
            vis = visdom.Visdom()
            if not vis.check_connection():
                vis = None
            print("WARNING: Visdom server not running. Please run python -m visdom.server to see visual output")
        except ImportError:
            vis = None
            raise RuntimeError("WARNING: No visdom package is found. Please install it with command: \n pip install visdom to see visual output")
        vis_windows = {}

        # save path and config files
        save_path = self.get_parameter_description(config_dict)
        model_path = os.path.join(save_path,"models/")
        utils_io.savePythonFile(config_dict_file, save_path)
        utils_io.savePythonFile(__file__, save_path)

        # now do training stuff
        epochs = 40
        device='cuda'
        train_loader = self.load_data_train(config_dict_cams)
        test_loader = self.load_data_test(config_dict_test)
        model = self.load_network(config_dict)
        model = model.to(device)
        optimizer = self.loadOptimizer(model,config_dict)
        loss_train,loss_test = self.load_loss(config_dict)

        trainer = utils_train.create_supervised_trainer(model, optimizer, loss_train, device=device)
        evaluator = utils_train.create_supervised_evaluator(model,
                                                metrics={#'accuracy': CategoricalAccuracy(),
                                                         'primary': utils_train.AccumulatedLoss(loss_test)},
                                                device=device)

        #@trainer.on(Events.STARTED)
        def load_previous_state(engine):
            utils_train.load_previous_state(save_path, model, optimizer, engine.state)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_progress(engine):
            #nonlocal vis_windows

            # log the loss
            iteration = engine.state.iteration - 1
            if iteration % config_dict['print_every'] == 0:
                utils_train.save_training_error(save_path, engine, vis, vis_windows)

            # log batch example image
            if iteration in [0,100] or iteration % config_dict['plot_every'] == 0:
                utils_train.save_training_example(save_path, engine, vis, vis_windows, config_dict)

        #@trainer.on(Events.EPOCH_COMPLETED)
        @trainer.on(Events.ITERATION_COMPLETED)
        def validate_model(engine):
            iteration = engine.state.iteration - 1
            if (iteration+1) % config_dict['test_every'] != 0: # +1 to prevent evaluation at iteration 0
                return
            #nonlocal vis_windows
            print("Running evaluation at iteration",iteration)
            evaluator.run(test_loader)
            avg_accuracy = utils_train.save_testing_error(save_path, engine, evaluator, vis, vis_windows)

            # save the best model
            utils_train.save_model_state(save_path, trainer, avg_accuracy, model, optimizer, engine.state)

        # Save latent variables and plot statistics about them
        @trainer.on(Events.ITERATION_COMPLETED)
        def validate_latent_space(engine):
            iteration = engine.state.iteration - 1
            if (iteration+1) % config_dict['test_every'] != 0: # +1 to prevent evaluation at iteration 0
                return

            # Create empty tensors to save latent spaces in
            if config_dict['variational_fg'] and config_dict['latent_fg'] > 0:
                mus_fg = torch.zeros(len(test_loader), config_dict['latent_fg'])
                logvars_fg = torch.zeros(len(test_loader), config_dict['latent_fg'])
            else:
                mus_fg, logvars_fg = None, None
            latent_fg = torch.zeros(len(test_loader), config_dict['latent_fg'])
            if config_dict['variational_3d']:
                mus_3d = torch.zeros(len(test_loader), config_dict['latent_3d']//3, 3)
                logvars_3d = torch.zeros(len(test_loader), config_dict['latent_3d']//3, 3)
            else:
                mus_3d, logvars_3d = None, None
            latent_3d = torch.zeros(len(test_loader), config_dict['latent_3d']//3, 3)

            # Iterate over all images and save their latent spaces
            model.eval()
            with torch.no_grad():
                for iter, (input_dict, label_dict) in enumerate(test_loader):
                    input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().cuda()
                    input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                    output_dict_cuda = model(input_dict_cuda)
                    output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')

                    if config_dict['variational_fg'] and config_dict['latent_fg'] > 0:
                        mus_fg[iter] = output_dict['mu_fg'][0]
                        logvars_fg[iter] = output_dict['logvar_fg'][0]
                    latent_fg[iter] = output_dict['latent_fg'][0]

                    if config_dict['variational_3d']:
                        mus_3d[iter] = output_dict['mu_3d'][0]
                        logvars_3d[iter] = output_dict['logvar_3d'][0]
                    latent_3d[iter] = output_dict['latent_3d'][0]
            model.train()

            # Save to disk
            latent_path = os.path.join(save_path, 'latent')
            latent_data_path = os.path.join(latent_path, 'data')
            if not os.path.exists(latent_data_path):
                os.makedirs(latent_data_path)
            if config_dict['variational_fg'] and config_dict['latent_fg'] > 0:
                mus_fg = mus_fg.numpy()
                np.save(os.path.join(latent_data_path,'mus_fg.npy'), mus_fg)
                logvars_fg = logvars_fg.numpy()
                np.save(os.path.join(latent_data_path,'logvars_fg.npy'), logvars_fg)
            latent_fg = latent_fg.numpy()
            np.save(os.path.join(latent_data_path,'latent_fg.npy'), latent_fg)
            if config_dict['variational_3d']:
                mus_3d = mus_3d.numpy()
                np.save(os.path.join(latent_data_path,'mus_3d.npy'), mus_3d)
                logvars_3d = logvars_3d.numpy()
                np.save(os.path.join(latent_data_path,'logvars_3d.npy'), logvars_3d)
            latent_3d = latent_3d.numpy()
            np.save(os.path.join(latent_data_path,'latent_3d.npy'), latent_3d)

            # Save plots
            if config_dict['variational_fg'] or config_dict['variational_3d']:
                utils_plt.plot_mu_std_hists(mus_fg, logvars_fg, mus_3d, logvars_3d, save_path=latent_path, save_iter=iteration+1)
                utils_plt.analyze_mu_std(mus_fg, logvars_fg, mus_3d, logvars_3d, save_path=latent_path, save_iter=iteration+1)
            else:
                utils_plt.plot_mu_std_hists(latent_fg, None, latent_3d, None, save_path=latent_path, save_iter=iteration+1)
                utils_plt.analyze_mu_std(latent_fg, None, latent_3d, None, save_path=latent_path, save_iter=iteration+1)

            if config_dict['variational_fg'] and config_dict['latent_fg'] > 0:
                utils_plt.plot_tsne(mus_fg, save_path=latent_path, save_iter=iteration+1, data_name='mus_fg')
                utils_plt.plot_tsne(logvars_fg, save_path=latent_path, save_iter=iteration+1, data_name='logvars_fg')
            else:
                utils_plt.plot_tsne(latent_fg, save_path=latent_path, save_iter=iteration+1, data_name='latent_fg')

            if config_dict['variational_3d']:
                utils_plt.plot_tsne(mus_3d, save_path=latent_path, save_iter=iteration+1, data_name='mus_3d')
                utils_plt.plot_tsne(logvars_3d, save_path=latent_path, save_iter=iteration+1, data_name='logvars_3d')
            else:
                utils_plt.plot_tsne(latent_3d, save_path=latent_path, save_iter=iteration+1, data_name='latent_3d')

            # TODO: Fix this to enable plotting using mayavi on headless servers
            # outlier_idxs = utils_plt.get_3d_outlier_idxs(logvars_3d) if config_dict['variational_3d'] else None
            # utils_plt.plot_scatter_3d(mus_3d, idxs=None, complement=False, save_path=latent_save_path, save_iter=iteration)
            # utils_plt.plot_scatter_3d(mus_3d, idxs=outlier_idxs, complement=False, save_path=latent_save_path, save_iter=iteration)

        # print test result
        @evaluator.on(Events.ITERATION_COMPLETED)
        def log_test_loss(engine):
            iteration = engine.state.iteration - 1
            if iteration in [0,100]:
                utils_train.save_test_example(save_path, trainer, evaluator, vis, vis_windows, config_dict)

    # kick everything off
        trainer.run(train_loader, max_epochs=epochs)

    def load_network(self, config_dict):
        output_types= config_dict['output_types']

        use_billinear_upsampling = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear']
        lower_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'half'
        upper_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'upper'

        from_latent_hidden_layers = config_dict.get('from_latent_hidden_layers', 0)
        num_encoding_layers = config_dict.get('num_encoding_layers', 4)

        num_cameras = 4
        if config_dict['active_cameras']: # for H36M it is set to False
            num_cameras = len(config_dict['active_cameras'])

        if lower_billinear:
            use_billinear_upsampling = False
        network_single = unet_encode3D.unet(config_dict,
                                            dimension_bg=config_dict['latent_bg'],
                                            dimension_fg=config_dict['latent_fg'],
                                            dimension_3d=config_dict['latent_3d'],
                                            feature_scale=config_dict['feature_scale'],
                                            shuffle_fg=config_dict['shuffle_fg'],
                                            shuffle_3d=config_dict['shuffle_3d'],
                                            latent_dropout=config_dict['latent_dropout'],
                                            in_resolution=config_dict['inputDimension'],
                                            encoderType=config_dict['encoderType'],
                                            is_deconv=not use_billinear_upsampling,
                                            upper_billinear=upper_billinear,
                                            lower_billinear=lower_billinear,
                                            from_latent_hidden_layers=from_latent_hidden_layers,
                                            n_hidden_to3Dpose=config_dict['n_hidden_to3Dpose'],
                                            num_encoding_layers=num_encoding_layers,
                                            output_types=output_types,
                                            subbatch_size=config_dict['useCamBatches'],
                                            implicit_rotation=config_dict['implicit_rotation'],
                                            skip_background=config_dict['skip_background'],
                                            num_cameras=num_cameras,
                                            variational_fg=config_dict['variational_fg'],
                                            variational_3d=config_dict['variational_3d'],
                                            use_second_stage=config_dict['use_second_stage'],
                                            )

        if 'pretrained_network_path' in config_dict.keys(): # automatic
            if config_dict['pretrained_network_path'] == 'MPII2Dpose':
                pretrained_network_path = '/cvlabdata1/home/rhodin/code/humanposeannotation/output_save/CVPR18_H36M/TransferLearning2DNetwork/h36m_23d_crop_relative_s1_s5_aug_from2D_2017-08-22_15-52_3d_resnet/models/network_000000.pth'
                print("Loading weights from MPII2Dpose")
                pretrained_states = torch.load(pretrained_network_path)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0, add_prefix='encoder.') # last argument is to remove "network.single" prefix in saved network
            else:
                print("Loading weights from config_dict['pretrained_network_path']")
                pretrained_network_path = config_dict['pretrained_network_path']
                pretrained_states = torch.load(pretrained_network_path)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0) # last argument is to remove "network.single" prefix in saved network
                print("Done loading weights from config_dict['pretrained_network_path']")

        if 'pretrained_posenet_network_path' in config_dict.keys(): # automatic
            print("Loading weights from config_dict['pretrained_posenet_network_path']")
            pretrained_network_path = config_dict['pretrained_posenet_network_path']
            pretrained_states = torch.load(pretrained_network_path)
            utils_train.transfer_partial_weights(pretrained_states, network_single.to_pose, submodule=0) # last argument is to remove "network.single" prefix in saved network
            print("Done loading weights from config_dict['pretrained_posenet_network_path']")
        return network_single

    def loadOptimizer(self,network, config_dict):
        if network.encoderType == "ResNet":
            params_all_id = list(map(id, network.parameters()))
            params_resnet_id = list(map(id, network.encoder.parameters()))
            params_except_resnet = [i for i in params_all_id if i not in params_resnet_id]

            # for the more complex setup
            params_toOptimize_id = (params_except_resnet
                             + list(map(id, network.encoder.layer4_reg.parameters()))
                             + list(map(id, network.encoder.layer3.parameters()))
                             + list(map(id, network.encoder.l4_reg_toVec.parameters()))
                             + list(map(id, network.encoder.fc.parameters())))
            params_toOptimize    = [p for p in network.parameters() if id(p) in params_toOptimize_id]

            params_static_id = [id_p for id_p in params_all_id if not id_p in params_toOptimize_id]

            # disable gradient computation for static params, saves memory and computation
            for p in network.parameters():
                if id(p) in params_static_id:
                    p.requires_grad = False

            print("Normal learning rate: {} params".format(len(params_toOptimize_id)))
            print("Static learning rate: {} params".format(len(params_static_id)))
            print("Total: {} params".format(len(params_all_id)))

            opt_params = [{'params': params_toOptimize, 'lr': config_dict['learning_rate']}]
            optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=config_dict['learning_rate'])
        return optimizer

    def load_data_train(self,config_dict):
        #return load_data_test(config_dict) # HACK

        #factory = dataset_factory.DatasetFactory()
        #trainloader, valloader_UNUSED = factory.load_data_train(config_dict_cams)
        dataset = collected_dataset.CollectedDataset(
                 data_folder='/cvlabdata1/home/rbachman/DataSets/H36M/H36M-MultiView-train',
                 input_types=config_dict['input_types'], label_types=config_dict['label_types_train'],
                 useSubjectBatches=config_dict['useSubjectBatches'], useCamBatches=config_dict['useCamBatches'], # HACK
                 useSequentialFrames=config_dict.get('useSequentialFrames',0),
                 randomize=True, augment_hue=config_dict['augment_hue'])
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=config_dict['batch_size_train'], shuffle=False, num_workers=config_dict['num_workers'], pin_memory=False, drop_last=True, collate_fn=utils_data.default_collate_with_string)
        trainloader = utils_data.PostFlattenInputSubbatchTensor(trainloader)
        return trainloader

    def load_data_test(self,config_dict):
        #factory = dataset_factory.DatasetFactory()
        #testloader = factory.load_data_test(config_dict_test)
        dataset = collected_dataset.CollectedDataset(
                 data_folder='/cvlabdata1/home/rbachman/DataSets/H36M/H36M-MultiView-test',
                 input_types=config_dict['input_types'], label_types=config_dict['label_types_test'],
                 useSubjectBatches=0, useCamBatches=config_dict['useCamBatches'],
                 randomize=False, augment_hue=False)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=config_dict['batch_size_test'], shuffle=False, num_workers=config_dict['num_workers'], pin_memory=False, drop_last=True, collate_fn=utils_data.default_collate_with_string)
        testloader = utils_data.PostFlattenInputSubbatchTensor(testloader)
        return testloader

    def load_loss(self, config_dict):
        weight = 1
        if config_dict['training_set'] in ['h36m','h36m_mpii']:
            weight = 17 / 16  # becasue spine is set to root = 0
        print("MPJPE test weight = {}, to normalize different number of joints".format(weight))

        # normal
        if config_dict.get('MAE', False):
            pairwise_loss = torch.nn.modules.loss.L1Loss()
        else:
            pairwise_loss = torch.nn.modules.loss.MSELoss()
        image_pixel_loss = losses_generic.LossOnDict(key='img_crop', loss=pairwise_loss)

        image_imgNet_bare = losses_images.ImageNetCriterium(criterion=pairwise_loss, weight=config_dict['loss_weight_imageNet'], do_maxpooling=config_dict.get('do_maxpooling',True))
        image_imgNet_loss = losses_generic.LossOnDict(key='img_crop', loss=image_imgNet_bare)

        losses_train = {}
        losses_test = {}
        loss_weights = {}

        if 'img_crop' in config_dict['output_types']:
            if config_dict['loss_weight_rgb']>0:
                losses_train['rgb'] = image_pixel_loss
                losses_test['rgb'] = image_pixel_loss
                loss_weights['rgb'] = config_dict['loss_weight_rgb']
            if config_dict['loss_weight_imageNet']>0:
                losses_train['imageNet'] = image_imgNet_loss
                losses_test['imageNet'] = image_imgNet_loss
                loss_weights['imageNet'] = config_dict['loss_weight_imageNet']

        if config_dict.get('variational', False):
            if config_dict['latent_fg'] > 0 and config_dict['variational_fg'] and config_dict['loss_weight_kl_fg'] > 0:
                kl_div_loss_fg = losses_generic.KLLoss(mu_key='mu_fg', logvar_key='logvar_fg')
                losses_train['kl_fg'] = kl_div_loss_fg
                losses_test['kl_fg'] = kl_div_loss_fg
                loss_weights['kl_fg'] = config_dict['loss_weight_kl_fg']
            if config_dict['variational_3d'] and config_dict['loss_weight_kl_3d'] > 0:
                kl_div_loss_3d = losses_generic.KLLoss(mu_key='mu_3d', logvar_key='logvar_3d')
                losses_train['kl_3d'] = kl_div_loss_3d
                losses_test['kl_3d'] = kl_div_loss_3d
                loss_weights['kl_3d'] = config_dict['loss_weight_kl_3d']

        loss_train = losses_generic.PreApplyCriterionDictDict(losses_train, sum_losses=True,
                                                              loss_weights=loss_weights,
                                                              KL_annealing=config_dict['kl_annealing'],
                                                              cyclical_beta=config_dict['cyclical_beta'])
        loss_test  = losses_generic.PreApplyCriterionDictDict(losses_test,  sum_losses=True,
                                                              loss_weights=loss_weights,
                                                              KL_annealing=config_dict['kl_annealing'],
                                                              cyclical_beta=config_dict['cyclical_beta'])

        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self, config_dict):#, config_dict):
        # folder = "./output/trainNVS_{note}_{encoderType}_layers{num_encoding_layers}_implR{implicit_rotation}_s3Dp{actor_subset_3Dpose}_w3Dp{loss_weight_pose3D}_w3D{loss_weight_3d}_wRGB{loss_weight_rgb}_wGrad{loss_weight_gradient}_wImgNet{loss_weight_imageNet}_skipBG{latent_bg}_fg{latent_fg}_3d{skip_background}_lh3Dp{n_hidden_to3Dpose}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_{training_set}_nth{every_nth_frame}_c{active_cameras}_sub{actor_subset}_bs{useCamBatches}_lr{learning_rate}_vaeFG{variational_fg}_vae3d{variational_3d}_kl3d{loss_weight_kl_3d}_kla{kl_annealing}_".format(**config_dict)
        folder = "./output/trainNVS_wRGB{loss_weight_rgb}_wImgNet{loss_weight_imageNet}_wKL3d{loss_weight_kl_3d}_wKLfg{loss_weight_kl_fg}_KLa{kl_annealing}_KLcycl{cyclical_beta}_skipBG{latent_bg}_3d{latent_3d}_fg{latent_fg}_3d{skip_background}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_{training_set}_lr{learning_rate}_vaeFG{variational_fg}_vae3d{variational_3d}_hue{augment_hue}_".format(**config_dict)
        folder = folder.replace(' ','').replace('./','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','./').replace(',','_')
        #config_dict['storage_folder'] = folder
        return folder


if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_train_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTrainNVS()
    ignite.run(config_dict_module.__file__, config_dict)
