
from torch.utils.data import DataLoader


from collections import OrderedDict

import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import sys

import math

import torch
import torch.nn as nn
import os
import numpy as np
from args import args

class Discriminator(nn.Module):
    def __init__(self, args, model_path=None):
        super(Discriminator, self).__init__()

        self.disMLP = nn.Sequential(
            nn.Linear(args.node_embed_size, 50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(50, 32),
            #nn.LeakyReLU(0.2, inplace=True),
        )

        self.adv_layer = nn.Sequential(nn.Linear(32,16),nn.Linear(16,8),nn.Linear(8,1), nn.Sigmoid())
        #self.aux_layer = nn.Sequential(nn.Linear(32, 2), nn.Softmax(dim =1))
        self.aux_layer  = nn.Sequential(nn.Linear(32,16),nn.Linear(16,8),nn.Linear(8,1), nn.Sigmoid())

       #0 : Sigmoid + Softmax sharing same MLP
       #1 : 2 Sigmoid sharing same MLP -->change to BCE
       #2 : 2 Sigmoid sharing different MLP --> change to BCE
       #3 : Consider making adv/aux layer more complex
       

        

    def forward(self, data):
        data_flat = data.view(data.size(0), -1)
        out = self.disMLP(data_flat)
        # out1 = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)  #*1
        label = self.aux_layer(out)     #*4

        return validity, label
    
class Generator(nn.Module):
    def __init__(self, args, model_path=None):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            return layers

        self.genMLP = nn.Sequential(
            *block(args.node_embed_size, args.node_embed_size),
            *block(args.node_embed_size, args.node_embed_size),
            nn.Linear(args.node_embed_size, args.node_embed_size),  #batchnorm avoided for Gen output
            nn.Tanh()
            #nn.Sigmoid()

        )

    def forward(self, pos_head):

        z = torch.normal(mean=0, std=0.01, size =pos_head.shape).detach().cuda()
        x = pos_head
        #z = torch.mul(pos_head,pos_rel_batch)
        y = x + z
        oupt = self.genMLP(y)
        return oupt

class Generator2(nn.Module):
    def __init__(self, args, model_path=None):
        super(Generator2, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            return layers

        self.genMLP = nn.Sequential(
            *block(args.node_embed_size, args.node_embed_size, normalize=False),
            *block(args.node_embed_size, args.node_embed_size),
            nn.Linear(args.node_embed_size, args.node_embed_size), #batchnorm avoided for Gen output
            nn.Tanh()
            #nn.Sigmoid()

        )

    def forward(self, pos_head, pos_rel):

        z = torch.normal(mean=0, std=0.01, size =pos_rel.shape).detach().cuda()
        #x = pos_rel
        x = torch.mul(pos_head,pos_rel)
        y = x + z
        oupt = self.genMLP(y)
        return oupt




class WayGAN2(object):
    def __init__(self, args, model_path=None):
        self.args = args
        logging.info("Building Generator...")
        generator = Generator(self.args, model_path)
        self.generator = generator.cuda()
        generator2 = Generator2(self.args, model_path)
        self.generator2 = generator2.cuda()
        #self.generator = generator
        logging.info("Building Discriminator...")
        discriminator = Discriminator(self.args, model_path)
        self.discriminator = discriminator.cuda()
        #self.discriminator = discriminator
        #Return No of 27269 unique nodes , 6 relations and dict of triplets (h,r,[t]) - test set

    def getVariables2(self):
        return (self.generator,self.generator2, self.discriminator)

    def getWayGanInstance(self):
        return self.waygan1


