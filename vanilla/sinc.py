import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,torch.ones(1),y_right]) # makes the mathematical condition here

    return y
    

class SincConv_fast(nn.Module):
    @staticmethod
    def to_mel(hz):
        hz = torch.tensor(hz, dtype=torch.float32)
        return 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        mel = torch.tensor(mel, dtype=torch.float32)
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1, stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast, self).__init__()
        if in_channels !=1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})"%(in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size%2==0:
            self.kernel_size = kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not bias")
        if groups >1:
            raise ValueError('SincConv does not suppooer groups')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),self.to_mel(high_hz),self.out_channels+1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1,1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1,1))

        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        window_ = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        n_ = 2*math.pi*torch.arange(-n,0).view(1,-1) / self.sample_rate # due to symmerty, I only need half of the time axes

        self.register_buffer("window_", window_)
        self.register_buffer("n_", n_)

    def forward(self, waveforms):
        # device = waveforms.device
        # self.n_ = self.n_.to(device)
        # self.window_ = self.window_.to(device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]

        f_times_t_low = torch.matmul(low,self.n_)
        f_times_t_high = torch.matmul(high,self.n_)

        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_
        band_pass_center = 2*band.view(-1,1)
        band_pass_right = torch.flip(band_pass_left,dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2*band[:,None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=self.padding, 
                        dilation=self.dilation, bias=None, groups=1)
