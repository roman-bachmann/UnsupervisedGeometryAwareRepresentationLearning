import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout

import os
import IPython
import random
import torch
import torch.autograd as A
import torch.nn.functional as F
import numpy as np

from models import resnet_transfer
from models import resnet_VNECT_3Donly
from models.vae import VAE

from models.unet_utils import *
from models import MLP

def quatCompact2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: coeff of quaternion of rotation, are normalized during conversion
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat/quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


class unet(nn.Module):
    def __init__(self,
                 config_dict,
                 feature_scale=4, # to reduce dimensionality
                 in_resolution=256,
                 output_channels=3, is_deconv=True,
                 upper_billinear=False,
                 lower_billinear=False,
                 in_channels=3, is_batchnorm=True,
                 skip_background=True,
                 num_joints=17, nb_dims=3, # ecoding transformation
                 encoderType='UNet',
                 num_encoding_layers=5,
                 dimension_bg=256,
                 dimension_fg=256,
                 dimension_3d=3*64, # needs to be devidable by 3
                 latent_dropout=0.3,
                 shuffle_fg=True,
                 shuffle_3d=True,
                 from_latent_hidden_layers=0,
                 n_hidden_to3Dpose=2,
                 subbatch_size = 4,
                 implicit_rotation = False,
                 nb_stage=1, # number of U-net stacks
                 output_types=['3D', 'img_crop', 'shuffled_pose', 'shuffled_appearance' ],
                 num_cameras=4,
                 variational_fg=False,
                 variational_3d=False,
                 use_second_stage=False
                 ):
        super(unet, self).__init__()
        self.in_resolution = in_resolution
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nb_stage = nb_stage
        self.dimension_bg=dimension_bg
        self.dimension_fg=dimension_fg
        self.dimension_3d=dimension_3d
        self.shuffle_fg=shuffle_fg
        self.shuffle_3d=shuffle_3d
        self.num_encoding_layers = num_encoding_layers
        self.output_types = output_types
        self.encoderType = encoderType
        assert dimension_3d % 3 == 0
        self.implicit_rotation = implicit_rotation
        self.num_cameras = num_cameras
        self.variational_fg = variational_fg
        self.variational_3d = variational_3d
        self.use_second_stage = use_second_stage

        self.skip_connections = False
        self.skip_background = skip_background
        self.subbatch_size = subbatch_size
        self.latent_dropout = latent_dropout

        #filters = [64, 128, 256, 512, 1024]
        self.filters = [64, 128, 256, 512, 512, 512] # HACK
        self.filters = [int(x / self.feature_scale) for x in self.filters]
        self.bottleneck_resolution = in_resolution//(2**(num_encoding_layers-1))
        num_output_features = self.bottleneck_resolution**2 * self.filters[num_encoding_layers-1]
        print('bottleneck_resolution',self.bottleneck_resolution,'num_output_features',num_output_features)

        ####################################
        ############ encoder ###############
        if self.encoderType == "ResNet":
            # Twice the latent space output size if variational
            num_classes = (variational_fg+1) * self.dimension_fg + (variational_3d+1) * self.dimension_3d
            self.encoder = resnet_VNECT_3Donly.resnet50(pretrained=True, input_key='img_crop', output_keys=['latent_3d','2D_heat'],
                                                        input_width=in_resolution, num_classes=num_classes)

        self.ns = 0
        setattr(self, 'conv_1_stage' + str(self.ns), unetConv2(self.in_channels, self.filters[0], self.is_batchnorm, padding=1))
        setattr(self, 'pool_1_stage' + str(self.ns), nn.MaxPool2d(kernel_size=2))
        for li in range(2,num_encoding_layers): # note, first layer(li==1) is already created, last layer(li==num_encoding_layers) is created externally
            setattr(self, 'conv_'+str(li)+'_stage' + str(self.ns), unetConv2(self.filters[li-2], self.filters[li-1], self.is_batchnorm, padding=1))
            setattr(self, 'pool_'+str(li)+'_stage' + str(self.ns), nn.MaxPool2d(kernel_size=2))

        if from_latent_hidden_layers:
            setattr(self, 'conv_'+str(num_encoding_layers)+'_stage' + str(self.ns),  nn.Sequential( unetConv2(self.filters[num_encoding_layers-2], self.filters[num_encoding_layers-1], self.is_batchnorm, padding=1),
                                                                    nn.MaxPool2d(kernel_size=2)
                                                                    ))
        else:
            setattr(self, 'conv_'+str(num_encoding_layers)+'_stage' + str(self.ns), unetConv2(self.filters[num_encoding_layers-2], self.filters[num_encoding_layers-1], self.is_batchnorm, padding=1))

        ####################################
        ############ background ###############
        if skip_background:
            setattr(self, 'conv_1_stage_bg' + str(self.ns), unetConv2(self.in_channels, self.filters[0], self.is_batchnorm, padding=1))

        ###########################################################
        ############ latent transformation and pose ###############
        assert self.dimension_fg < self.filters[num_encoding_layers-1]
        num_output_features_3d = self.bottleneck_resolution**2 * (self.filters[num_encoding_layers-1] - self.dimension_fg)
        #setattr(self, 'fc_1_stage' + str(ns), Linear(num_output_features, 1024))
        setattr(self, 'fc_1_stage' + str(self.ns), Linear(self.dimension_3d, 128))
        setattr(self, 'fc_2_stage' + str(self.ns), Linear(128, num_joints * nb_dims))

        self.to_pose = MLP.MLP_fromLatent(d_in=self.dimension_3d, d_hidden=2048, d_out=51, n_hidden=n_hidden_to3Dpose, dropout=0.5)

        self.to_3d =  nn.Sequential( Linear(num_output_features, self.dimension_3d),
                                     Dropout(inplace=True, p=self.latent_dropout) # removing dropout degrades results
                                   )

        if self.implicit_rotation:
            print("WARNING: doing implicit rotation!")
            rotation_encoding_dimension = 128
            self.encode_angle =  nn.Sequential(Linear(3*3, rotation_encoding_dimension//2),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),
                                         Linear(rotation_encoding_dimension//2, rotation_encoding_dimension),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),
                                         Linear(rotation_encoding_dimension, rotation_encoding_dimension),
                                         )

            self.rotate_implicitely = nn.Sequential(Linear(self.dimension_3d + rotation_encoding_dimension, self.dimension_3d),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))

        if from_latent_hidden_layers:
            hidden_layer_dimension = 1024
            if self.dimension_fg > 0:
                self.to_fg =  nn.Sequential( Linear(num_output_features, 256), # HACK pooling
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),
                                         Linear(256, self.dimension_fg),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))
            self.from_latent =  nn.Sequential( Linear(self.dimension_3d, hidden_layer_dimension),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),
                                         Linear(hidden_layer_dimension, num_output_features_3d),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))
        else:
            if self.dimension_fg > 0:
                self.to_fg =  nn.Sequential( Linear(num_output_features, self.dimension_fg),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))
            self.from_latent =  nn.Sequential( Linear(self.dimension_3d, num_output_features_3d),
                             Dropout(inplace=True, p=self.latent_dropout),
                             ReLU(inplace=False))

        ####################################
        ############ decoder ###############
        upper_conv = self.is_deconv and not upper_billinear
        lower_conv = self.is_deconv and not lower_billinear
        if self.skip_connections:
            for li in range(1,num_encoding_layers-1):
                setattr(self, 'upconv_'+str(li)+'_stage' + str(self.ns), unetUp(self.filters[num_encoding_layers-li], self.filters[num_encoding_layers-li-1], upper_conv, padding=1))
                #setattr(self, 'upconv_2_stage' + str(self.ns), unetUp(self.filters[2], self.filters[1], upper_conv, padding=1))
        else:
            for li in range(1,num_encoding_layers-1):
                setattr(self, 'upconv_'+str(li)+'_stage' + str(self.ns), unetUpNoSKip(self.filters[num_encoding_layers-li], self.filters[num_encoding_layers-li-1], upper_conv, padding=1))
            #setattr(self, 'upconv_2_stage' + str(self.ns), unetUpNoSKip(self.filters[2], self.filters[1], upper_conv, padding=1))

        if self.skip_connections or self.skip_background:
            setattr(self, 'upconv_'+str(num_encoding_layers-1)+'_stage' + str(self.ns), unetUp(self.filters[1], self.filters[0], lower_conv, padding=1))
        else:
            setattr(self, 'upconv_'+str(num_encoding_layers-1)+'_stage' + str(self.ns), unetUpNoSKip(self.filters[1], self.filters[0], lower_conv, padding=1))

        setattr(self, 'final_stage' + str(self.ns), nn.Conv2d(self.filters[0], output_channels, 1))

        self.relu = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.dropout = Dropout(inplace=True, p=0.3)

        ####################################
        ########## 2-stage VAE #############
        if use_second_stage:
            self.vae_fg = VAE(input_dim=self.dimension_fg, hidden_dim=config_dict['second_stage_hidden_dim'], latent_dim=config_dict['second_stage_latent_dim']).to('cuda')
            model_path_fg = os.path.join(config_dict['network_path'], 'models', 'second_stage_vae_fg_ldim{}.pth'.format(config_dict['second_stage_latent_dim']))
            self.vae_fg.load_state_dict(torch.load(model_path_fg))
            self.vae_fg.eval()

            self.vae_3d = VAE(input_dim=self.dimension_3d, hidden_dim=config_dict['second_stage_hidden_dim'], latent_dim=config_dict['second_stage_latent_dim']).to('cuda')
            model_path_3d = os.path.join(config_dict['network_path'], 'models', 'second_stage_vae_3d_ldim{}.pth'.format(config_dict['second_stage_latent_dim']))
            self.vae_3d.load_state_dict(torch.load(model_path_3d))
            self.vae_3d.eval()

    # Determine shuffling
    def shuffle_segment(self, list, start, end):
        selected = list[start:end]
        if self.training:
            if 0 and end-start == 2: # Note, was not enabled in ECCV submission, diabled now too HACK
                prob = np.random.random([1])
                if prob[0] > 1/self.num_cameras: # assuming four cameras, make it more often that one of the others is taken, rather than just autoencoding (no flip, which would happen 50% otherwise)
                    selected = selected[::-1] # reverse
                else:
                    pass # let it as it is
            else:
                random.shuffle(selected)

        else: # deterministic shuffling for testing
            selected = np.roll(selected,1).tolist()
        list[start:end] = selected

    def flip_segment(self, list, start, width):
        selected = list[start:start+width]
        list[start:start+width] = list[start+width:start+2*width]
        list[start+width:start+2*width] = selected

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # Sample from N(0,1)
        return mu + eps*std

    # TODO
    def encode(self, input_dict):
        pass

    def decoder(self, latent_fg_shuffled, latent_3d_rotated, shuffled_pose=None, shuffled_pose_inv=None, input_bg=None):
        batch_size = latent_3d_rotated.shape[0]

        map_from_3d = self.from_latent(latent_3d_rotated.view(batch_size,-1))
        map_width = self.bottleneck_resolution #out_enc_conv.size()[2]
        map_channels = self.filters[self.num_encoding_layers-1] #out_enc_conv.size()[1]
        if hasattr(self, "to_fg"):
            latent_fg_shuffled_replicated = latent_fg_shuffled.view(batch_size,self.dimension_fg,1,1).expand(batch_size, self.dimension_fg, map_width, map_width)
            latent_shuffled = torch.cat([latent_fg_shuffled_replicated, map_from_3d.view(batch_size, map_channels-self.dimension_fg, map_width, map_width)], dim=1)
        else:
            latent_shuffled = map_from_3d.view(batch_size, map_channels, map_width, map_width)

        if self.skip_connections:
            assert False
        else:
            out_deconv = latent_shuffled
            for li in range(1,self.num_encoding_layers-1):
                out_deconv = getattr(self, 'upconv_'+str(li)+'_stage' + str(self.ns))(out_deconv)

            if self.skip_background and input_bg is not None:
                if shuffled_pose is not None:
                    input_bg_shuffled = torch.index_select(input_bg, dim=0, index=shuffled_pose)
                else:
                    input_bg_shuffled = input_bg
                conv1_bg_shuffled = getattr(self, 'conv_1_stage_bg' + str(self.ns))(input_bg_shuffled)
                out_deconv = getattr(self, 'upconv_'+str(self.num_encoding_layers-1)+'_stage' + str(self.ns))(conv1_bg_shuffled, out_deconv)
            else:
                out_deconv = getattr(self, 'upconv_'+str(self.num_encoding_layers-1)+'_stage' + str(self.ns))(out_deconv)

        output_img_shuffled = getattr(self, 'final_stage' + str(self.ns))(out_deconv)

        # de-shuffling if wanted
        if shuffled_pose_inv is not None:
            return torch.index_select(output_img_shuffled, dim=0, index=shuffled_pose_inv)
        else:
            return output_img_shuffled


    def forward(self, input_dict):
        input = input_dict['img_crop']
        batch_size = input.size()[0]
        num_pose_examples = batch_size//2
        num_appearance_examples = batch_size//2
        num_appearance_subbatches = num_appearance_examples//np.maximum(self.subbatch_size,1)

        shuffled_appearance = list(range(batch_size))
        shuffled_pose       = list(range(batch_size))
        num_pose_subbatches = batch_size//np.maximum(self.subbatch_size,1)

        rotation_by_user = self.training==False and 'external_rotation_cam' in input_dict.keys()

        if not rotation_by_user:
            if self.shuffle_fg and self.training==True:
                for i in range(0,num_pose_subbatches):
                    self.shuffle_segment(shuffled_appearance, i*self.subbatch_size, (i+1)*self.subbatch_size)
                for i in range(0,num_pose_subbatches//2): # flip first with second subbatch
                    self.flip_segment(shuffled_appearance, i*2*self.subbatch_size, self.subbatch_size)
            if self.shuffle_3d:
                for i in range(0,num_pose_subbatches):
                    self.shuffle_segment(shuffled_pose, i*self.subbatch_size, (i+1)*self.subbatch_size)

        # infer inverse mapping
        shuffled_pose_inv = [-1] * batch_size
        for i,v in enumerate(shuffled_pose):
            shuffled_pose_inv[v]=i

        # print('self.training',self.training,"shuffled_appearance",shuffled_appearance)
        # print("shuffled_pose      ",shuffled_pose)

        shuffled_appearance = A.Variable(torch.LongTensor(shuffled_appearance)).cuda()
        shuffled_pose       = A.Variable(torch.LongTensor(shuffled_pose)).cuda()
        shuffled_pose_inv   = A.Variable(torch.LongTensor(shuffled_pose_inv)).cuda()

        if rotation_by_user:
            if 'shuffled_appearance' in input_dict.keys():
                shuffled_appearance = input_dict['shuffled_appearance'].long()

        ###############################################
        # determine shuffled rotation
        cam_2_world = input_dict['extrinsic_rot_inv'].view( (batch_size, 3, 3) ).float()
        world_2_cam = input_dict['extrinsic_rot'].    view( (batch_size, 3, 3) ).float()
        if rotation_by_user:
            external_cam = input_dict['external_rotation_cam'].view(1,3,3).expand( (batch_size, 3, 3) )
            external_glob = input_dict['external_rotation_global'].view(1,3,3).expand( (batch_size, 3, 3) )
            cam2cam = torch.bmm(external_cam,torch.bmm(world_2_cam, torch.bmm(external_glob, cam_2_world)))
        else:
            world_2_cam_suffled = torch.index_select(world_2_cam, dim=0, index=shuffled_pose)
            cam2cam = torch.bmm(world_2_cam_suffled, cam_2_world)

        input_dict_cropped = input_dict # fallback to using crops

        ###############################################
        # encoding stage
        ns=0
        has_fg = hasattr(self, "to_fg")
        if self.encoderType == "ResNet":
            #IPython.embed()
            output = self.encoder.forward(input_dict_cropped)['latent_3d']
            if has_fg:
                if self.variational_fg:
                    mu_from, mu_to = 0, self.dimension_fg
                    logvar_from, logvar_to = self.dimension_fg, 2*self.dimension_fg
                    mu_fg = output[:,mu_from:mu_to]
                    logvar_fg = output[:,logvar_from:logvar_to]
                    if self.training:
                        latent_fg = self.reparameterize(mu_fg, logvar_fg)
                    else:
                        latent_fg = mu_fg
                        # latent_fg = self.reparameterize(mu_fg, logvar_fg)
                else:
                    latent_fg = output[:,:self.dimension_fg]
            if self.variational_3d:
                mu_from, mu_to = 2*self.dimension_fg, 2*self.dimension_fg + self.dimension_3d
                logvar_from, logvar_to = 2*self.dimension_fg + self.dimension_3d, 2*self.dimension_fg + 2*self.dimension_3d
                mu_3d = output[:,mu_from:mu_to].contiguous().view(batch_size,-1,3)
                logvar_3d = output[:,logvar_from:logvar_to].contiguous().view(batch_size,-1,3)
                if self.training:
                    latent_3d = self.reparameterize(mu_3d, logvar_3d)
                else:
                    latent_3d = mu_3d
                    # latent_3d = self.reparameterize(mu_3d, logvar_3d)
            else:
                latent_3d = output[:,self.dimension_fg:self.dimension_fg+self.dimension_3d].contiguous().view(batch_size,-1,3)
        else: # UNet encoder
            # TODO: Implement VAE here too
            out_enc_conv = input_dict_cropped['img_crop']
            for li in range(1,self.num_encoding_layers): # note, first layer(li==1) is already created, last layer(li==num_encoding_layers) is created externally
                out_enc_conv = getattr(self, 'conv_'+str(li)+'_stage' + str(self.ns))(out_enc_conv)
                out_enc_conv = getattr(self, 'pool_'+str(li)+'_stage' + str(self.ns))(out_enc_conv)
            out_enc_conv = getattr(self, 'conv_'+str(self.num_encoding_layers)+'_stage' + str(self.ns))(out_enc_conv)
            # fully-connected
            center_flat = out_enc_conv.view(batch_size,-1)
            if has_fg:
                latent_fg = self.to_fg(center_flat)
            latent_3d = self.to_3d(center_flat).view(batch_size,-1,3)

        # Use second stage VAE
        if self.use_second_stage:
            if has_fg:
                latent_fg, _, _, _ = self.vae_fg(latent_fg, sample=False)
            latent_3d, _, _, _ = self.vae_3d(latent_3d.view(batch_size,-1), sample=False)
            latent_3d = latent_3d.view(batch_size,-1,3)

        ###############################################
        # latent rotation (to shuffled view)
        if self.implicit_rotation:
            encoded_angle = self.encode_angle(cam2cam.view(batch_size,-1))
            encoded_latent_and_angle = torch.cat([latent_3d.view(batch_size,-1), encoded_angle], dim=1)
            latent_3d_rotated = self.rotate_implicitely(encoded_latent_and_angle)
        else:
            latent_3d_rotated = torch.bmm(latent_3d, cam2cam.transpose(1,2))

        if 'shuffled_pose_weight' in input_dict.keys():
            w = input_dict['shuffled_pose_weight']
            # weighted average with the last one
            latent_3d_rotated = (1-w.expand_as(latent_3d))*latent_3d + w.expand_as(latent_3d)*latent_3d_rotated[-1:].expand_as(latent_3d)

        if has_fg:
            latent_fg_shuffled = torch.index_select(latent_fg, dim=0, index=shuffled_appearance)
            if 'shuffled_appearance_weight' in input_dict.keys():
                w = input_dict['shuffled_appearance_weight']
                latent_fg_shuffled = (1-w.expand_as(latent_fg))*latent_fg + w.expand_as(latent_fg)*latent_fg_shuffled

        ###############################################
        # decoding
        input_bg = input_dict['bg_crop'] if self.skip_background else None
        output_img = self.decoder(latent_fg_shuffled, latent_3d_rotated, shuffled_pose, shuffled_pose_inv, input_bg)

        ###############################################
        # 3D pose stage (parallel to image decoder)
        output_pose = self.to_pose.forward({'latent_3d': latent_3d})['3D']

        ###############################################
        # Select the right output
        output_dict_all = {'3D' : output_pose, 'img_crop' : output_img, 'shuffled_pose' : shuffled_pose,
                           'shuffled_appearance' : shuffled_appearance, 'cam2cam': cam2cam,
                           'latent_3d': latent_3d, 'latent_fg': latent_fg_shuffled }
        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        if self.variational_fg and has_fg:
            output_dict['mu_fg'] = mu_fg
            output_dict['logvar_fg'] = logvar_fg
        if self.variational_3d:
            output_dict['mu_3d'] = mu_3d
            output_dict['logvar_3d'] = logvar_3d

        return output_dict
