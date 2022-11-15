
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

class Filter(nn.Module):
    def __init__(self,args,model_path=None):
        super(Filter, self).__init__()

        self.film_alpha_layers1 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)
        self.film_beta_layers1 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)
        self.film_alpha_layers2 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)
        self.film_beta_layers2 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)

        # self.bn_layers = nn.BatchNormalization()

    def forward(self,input):
        film_alpha_layer1 = self.film_alpha_layers1(input)
        film_beta_layer1 = self.film_beta_layers1(input)

        film_alpha_layer2 = self.film_alpha_layers2(input)
        film_beta_layer2 = self.film_beta_layers2(input)

        return film_alpha_layer1,film_beta_layer1,film_alpha_layer2, film_beta_layer2

class Filter2(nn.Module):
    def __init__(self,args,model_path=None):
        super(Filter2, self).__init__()

        self.film_alpha_layers1 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)
        self.film_beta_layers1 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)
        self.film_alpha_layers2 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)
        self.film_beta_layers2 = nn.Linear(args.node_embed_size, args.node_embed_size,bias=False)

        # self.bn_layers = nn.BatchNormalization()

    def forward(self,input):
        film_alpha_layer1 = self.film_alpha_layers1(input)
        film_beta_layer1 = self.film_beta_layers1(input)

        film_alpha_layer2 = self.film_alpha_layers2(input)
        film_beta_layer2 = self.film_beta_layers2(input)

        return film_alpha_layer1,film_beta_layer1,film_alpha_layer2, film_beta_layer2




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
    def __init__(self, args,model_path=None):
        super(Generator, self).__init__()

        self.genMLP1 = nn.Sequential(
            nn.Linear(args.node_embed_size,args.node_embed_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.genMLP2 = nn.Sequential(
            nn.Linear(args.node_embed_size,args.node_embed_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.output= nn.Sequential(
            nn.Linear(args.node_embed_size,args.node_embed_size),
            nn.Tanh())

        self.activation= nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(args.node_embed_size, 0.8)


    def forward(self, pos_head):
        film_alpha_layer1,film_beta_layer1,film_alpha_layer2, film_beta_layer2 = args.filt(pos_head)

        z = torch.normal(mean=0, std=0.01, size =pos_head.shape).detach().cuda()
        x = pos_head
        y = x + z

        MLP1 = self.genMLP1(y)
        MLP1 = torch.mul(MLP1,film_alpha_layer1) + film_beta_layer1
        MLP1 = self.bn(MLP1)
        MLP1 = self.activation(MLP1)

        MLP2 = self.genMLP2(MLP1)
        MLP2 = torch.mul(MLP2,film_alpha_layer2) + film_beta_layer2
        MLP2 = self.bn(MLP2)
        MLP2 = self.activation(MLP2)

        oupt = self.output(MLP2)

        return oupt


class Generator2(nn.Module):
    def __init__(self, args, model_path=None):
        super(Generator2, self).__init__()

        self.genMLP1 = nn.Sequential(
            nn.Linear(args.node_embed_size, args.node_embed_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.genMLP2 = nn.Sequential(
            nn.Linear(args.node_embed_size, args.node_embed_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Sequential(
            nn.Linear(args.node_embed_size, args.node_embed_size),
            nn.Tanh())

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(args.node_embed_size, 0.8)

    def forward(self, pos_head, pos_rel):
        film_alpha_layer1, film_beta_layer1, film_alpha_layer2, film_beta_layer2 = args.filt2(torch.mul(pos_head,pos_rel))

        z = torch.normal(mean=0, std=0.01, size=pos_rel.shape).detach().cuda()
        x = torch.mul(pos_head,pos_rel)
        y = x + z

        MLP1 = self.genMLP1(y)
        MLP1 = torch.mul(MLP1, film_alpha_layer1) + film_beta_layer1
        MLP1 = self.bn(MLP1)
        MLP1 = self.activation(MLP1)

        MLP2 = self.genMLP2(MLP1)
        MLP2 = torch.mul(MLP2, film_alpha_layer2) + film_beta_layer2
        MLP2 = self.bn(MLP2)
        MLP2 = self.activation(MLP2)

        oupt = self.output(MLP2)

        return oupt

#
# class Generator2(nn.Module):
#     def __init__(self, args, model_path=None):
#         super(Generator2, self).__init__()
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#
#             return layers
#
#         self.genMLP = nn.Sequential(
#             *block(args.node_embed_size, args.node_embed_size, normalize=False),
#             *block(args.node_embed_size, args.node_embed_size),
#             nn.Linear(args.node_embed_size, args.node_embed_size), #batchnorm avoided for Gen output
#             nn.Tanh()
#             #nn.Sigmoid()
#
#         )
#
#     def forward(self, pos_head, pos_rel):
#
#         z = torch.normal(mean=0, std=0.01, size =pos_rel.shape).detach().cuda()
#         #x = pos_rel
#         x = torch.mul(pos_head,pos_rel)
#         y = x + z
#         oupt = self.genMLP(y)
#         return oupt




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

        logging.info("Building Filter...")
        filt = Filter(self.args, model_path)
        self.filt = filt.cuda()

        filt2 = Filter(self.args, model_path)
        self.filt2 = filt2.cuda()

        #self.discriminator = discriminator
        #Return No of 27269 unique nodes , 6 relations and dict of triplets (h,r,[t]) - test set

    def getVariables2(self):
        return (self.generator,self.generator2, self.discriminator,self.filt, self.filt2)

    def getWayGanInstance(self):
        return self.waygan1


