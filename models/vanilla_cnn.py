import torch
import torch.nn as nn


class vanilla_cnn(nn.Module):   
    def __init__(self, in_channels:int=4, out_channels:int=1,num_features:int=64, 
                 kernel_size:int=3, stride:str=1, padding:str=1, pooling_size:tuple=(1200, 1200),
                 between_blocks:int=2):
        super(vanilla_cnn, self).__init__()
        inc, nf, k = in_channels, num_features, kernel_size
        pool_size=pooling_size
        s, p, b = stride, padding, int(between_blocks/2)
        
        first = [cnn_block(inc, nf, k, s, p, pool_size)]
        ups = [cnn_block(nf*(2**i), nf*(2**(i+1)), k, s, p, pool_size) for i in range(0, b)]
        between = [cnn_block(nf*(2**b), nf*(2**b), k, s, p, pool_size)]
        downs = [cnn_block(nf*(2**(i+1)), nf*(2**(i)), k, s, p, pool_size) for i in reversed(range(0, b))]
        last = [cnn_block(nf, 1, k, s, p, pool_size)]
        all_blocks = first + ups + between + downs + last
        self.model = nn.Sequential(*all_blocks)
        
    def forward(self, x):
        return self.model(x)


class cnn_block(nn.Module):   
    def __init__(self, in_channels:int=4, out_channels:int=256, kernel_size:int=3, 
                 stride:str=1, padding:str=1, pooling_size:tuple=(1200, 1200)):
        super(cnn_block, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                   nn.BatchNorm2d(out_channels),
                                   nn.AdaptiveAvgPool2d(pooling_size),
                                   nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.model(x)