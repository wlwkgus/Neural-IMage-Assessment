#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import OrderedDict

import numpy as np
import matplotlib

from data_loader import get_data_loader, get_val_data_loader
from option_parser import OptionParser
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt

from torch import autograd
from torch import optim
from torchvision import models
from model import *

matplotlib.use('Agg')


def main(option):
    data_loader = get_data_loader(option)
    val_data_loader = get_val_data_loader(option)

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)

    if option.warm_start:
        model.load_state_dict(torch.load(os.path.join(option.ckpt_dir, 'epoch-%d.pkl' % option.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' % option.warm_start_epoch)

    if option.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=option.gpu_ids)
        model = model.cuda()
    else:
        model = model.cuda()

    conv_base_lr = option.conv_base_lr
    dense_lr = option.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

    # send hyperparams
    # lrs.send({
    #     'title': 'EMD Loss',
    #     'train_batch_size': option.train_batch_size,
    #     'val_batch_size': option.val_batch_size,
    #     'optimizer': 'SGD',
    #     'conv_base_lr': option.conv_base_lr,
    #     'dense_lr': option.dense_lr,
    #     'momentum': 0.9
    #     })

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if option.is_train:
        # for early stopping
        images = torch.FloatTensor(
            option.train_batch_size,
            3,
            224,
            224
        )
        labels = torch.FloatTensor(
            option.train_batch_size,
            10
        )
        count = 0
        visualizer = Visualizer(option)
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(option.warm_start_epoch, option.epochs):
            batch_losses = []
            for i, data in enumerate(data_loader):
                images.resize_(data['image'].size()).copy_(data['image'])
                labels.resize_(data['annotations'].size()).copy_(data['annotations'].float())
                if torch.cuda.is_available():
                    images.cuda()
                    labels.cuda()
                outputs = model(autograd.Variable(images, requires_grad=True))
                outputs = outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                loss = emd_loss(autograd.Variable(labels), outputs)
                batch_losses.append(loss.data[0])

                loss.backward()

                optimizer.step()

                visualizer.plot_current_nums(epoch * len(data_loader) + i, 0, OrderedDict([
                    ('batch_loss', loss.data[0]),
                    ('none', 0.)
                ]))
                # visualizer.plot_current_nums(epoch * len(data_loader) + i, 0, OrderedDict([
                #     ('conv_base_lr', conv_base_lr),
                #     ('dense_lr', dense_lr)
                # ]),
                #                              display_id=2, title='lrs')


                # lrs.send('train_emd_loss', loss.data[0])
                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, option.epochs, i + 1, len(data_loader), loss.data[0]))

            avg_loss = sum(batch_losses) / (len(data_loader))
            train_losses.append(avg_loss)
            print('Epoch %d averaged training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if (epoch + 1) % 10 == 0:
                conv_base_lr *= option.lr_decay_rate ** ((epoch + 1) / option.lr_decay_freq)
                dense_lr *= option.lr_decay_rate ** ((epoch + 1) / option.lr_decay_freq)
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9
                )

                # send decay hyperparams
                # lrs.send({
                #     'lr_decay_rate': option.lr_decay_rate,
                #     'lr_decay_freq': option.lr_decay_freq,
                #     'conv_base_lr': option.conv_base_lr,
                #     'dense_lr': option.dense_lr
                #     })

            # do validation after each epoch
            batch_val_losses = []
            for data in val_data_loader:
                images.resize_(data['image'].size()).copy_(data['image'])
                labels.resize_(data['annotations'].size()).copy_(data['annotations'].float())
                if torch.cuda.is_available():
                    images.cuda()
                    labels.cuda()
                model.eval()
                outputs = model(autograd.Variable(images))
                model.train()
                outputs = outputs.view(-1, 10, 1)
                val_loss = emd_loss(autograd.Variable(labels), outputs)
                batch_val_losses.append(val_loss.data[0])
            avg_val_loss = sum(batch_val_losses) / (len(val_data_loader) // option.val_batch_size + 1)
            val_losses.append(avg_val_loss)

            # lrs.send('val_emd_loss', avg_val_loss)

            print('Epoch %d completed. Averaged EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                torch.save(model.state_dict(), os.path.join(option.ckpt_dir, 'epoch-%d.pkl' % (epoch + 1)))
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == option.early_stopping_patience:
                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % option.early_stopping_patience)
                    break

        print('Training completed.')

        if option.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')

    else:
        # compute mean score

        mean_preds = []
        std_preds = []
        for data in data_loader:
            image = data['image']
            if torch.cuda.is_available():
                image = image.cuda()
            output = model(image)
            output = output.view(10, 1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    option_parser = OptionParser()
    opt = option_parser.parse_args()
    main(opt)

