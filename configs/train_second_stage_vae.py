import os
import numpy as np
import argparse
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from models.vae import VAE
from datasets.latent_dataset import LatentDataset
from utils import io as utils_io


parser = argparse.ArgumentParser(description='Seconds stage VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)


class SecondStageTrainer():
    def __init__(self, config_dict, mode='fg'):
        if mode not in ['fg', '3d', 'both']:
            raise ValueError('Please set parameter \'mode\' to one of \{\'fg\', \'3d\', \'both\'\}.')
        self.config_dict = config_dict
        self.mode = mode

        z_dataset_train = LatentDataset(config_dict, train=True,  mode=mode, sample_latent=True)
        z_dataset_test  = LatentDataset(config_dict, train=False, mode=mode, sample_latent=True)
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
        self.train_loader = DataLoader(z_dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader  = DataLoader( z_dataset_test, batch_size=args.batch_size, shuffle=True, **kwargs)
        print('Data loaded and datasets created')

        # Input dimension
        if mode == 'fg':
            self.k_dim = config_dict['latent_fg']
        elif mode == '3d':
            self.k_dim = config_dict['latent_3d']
        elif mode == 'both':
            self.k_dim = config_dict['latent_fg'] + config_dict['latent_3d']

        self.model = VAE(input_dim=self.k_dim, hidden_dim=config_dict['second_stage_hidden_dim'], latent_dim=config_dict['second_stage_latent_dim']).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.best_test_loss = np.inf
        print('Training second stage VAE for {}'.format(mode))
        for epoch in range(1, args.epochs + 1):
            train_loss = self.train(epoch)
            test_loss = self.test(epoch)
        print('Training done!')


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        RCL = F.mse_loss(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return RCL + KLD


    def train(self, epoch):
        base_iter = (epoch-1) * len(self.train_loader)
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar, z = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            # writer.add_scalar('loss_curves/iteration_loss_train', loss.item() / len(data), base_iter+batch_idx)
            self.optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        train_loss /= len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss))
        return train_loss

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        mus, logvars, zs = [], [], []

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data.to(device)
                recon_batch, mu, logvar, z = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                mus.append(mu.cpu().numpy())
                logvars.append(logvar.cpu().numpy())
                zs.append(z.cpu().numpy())

        mus = np.vstack(mus)
        logvars = np.vstack(logvars)
        zs = np.vstack(zs)

        # writer.add_histogram('hist/mu_test', mus, epoch)
        # writer.add_histogram('hist/std_test', np.exp(0.5*logvars), epoch)
        # writer.add_histogram('hist/z_test', zs, epoch)
        # writer.add_embedding(zs, global_step=epoch)

        KLD = -0.5 * np.sum(1 + logvars - mus**2 - np.exp(logvars))
        KLD /= len(mus)
        # writer.add_scalar('loss_curves/epoch_KL_test', KLD, epoch)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        if test_loss < self.best_test_loss:
            model_save_path = os.path.join(self.config_dict['network_path'], 'models', 'second_stage_vae_{}.pth'.format(self.mode))
            torch.save(self.model.state_dict(), model_save_path)
        return test_loss


if __name__ == '__main__':
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    SecondStageTrainer(config_dict, mode='fg')
    SecondStageTrainer(config_dict, mode='3d')
