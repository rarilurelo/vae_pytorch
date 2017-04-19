import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class VAE(nn.Module):
    def __init__(self, z_dim=32, size=(64, 64), cuda=True):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.size = (64, 64)
        self.cuda = cuda
        self.encode_cnn = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                )
        self.encode_mu = nn.Sequential(
                nn.Linear(2592, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, self.z_dim),
                )
        self.encode_log_std = nn.Sequential(
                nn.Linear(2592, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, self.z_dim),
                )
        self.decode_linear = nn.Sequential(
                nn.Linear(self.z_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 2592),
                )
        self.decode_cnn = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 5, 2, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                )
        self.decode_mu = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 5, 2, output_padding=1),
                nn.Sigmoid(),
                )
        self.decode_log_std = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 5, 2, output_padding=1),
                )

    def forward(self, x):
        x = self.encode_cnn(x)
        x = x.view(-1, 2592)
        encode_mu = self.encode_mu(x)
        encode_log_std = self.encode_log_std(x)
        if self.cuda:
            eps = torch.cuda.FloatTensor(encode_log_std.size()).normal_()
        else:
            eps = torch.FloatTensor(encode_log_std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(encode_log_std.exp_()).add_(encode_mu)

        h = self.decode_linear(z)
        h = h.view(-1, 32, 9, 9)
        h = self.decode_cnn(h)
        decode_mu = self.decode_mu(h)
        decode_log_std = self.decode_log_std(h)
        if self.cuda:
            eps = torch.cuda.FloatTensor(decode_log_std.size()).normal_()
        else:
            eps = torch.FloatTensor(decode_log_std.size()).normal_()
        eps = Variable(eps)
        reconstruct_x = eps.mul(decode_log_std.exp_()).add_(decode_mu)

        return reconstruct_x, z, encode_mu, encode_log_std, decode_mu, decode_log_std

def kl()

model = VAE(cuda=False)
x = Variable(torch.from_numpy(np.random.normal(size=[10, 3, 64, 64])).float())

model(x)
