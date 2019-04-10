# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:01:20 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:36:29 2019

@author: Administrator
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:08:06 2019

@author: xiamu
"""
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V


from module import DinkNet34

from loss import dice_bce_loss
from data import ImageFolder



def update_lr(model,old_lr,new_lr, mylog, factor=False):
    if factor:
        new_lr = old_lr / new_lr
    for param_group in model.param_groups:
        param_group['lr'] = new_lr

    #print(>> mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
    print('update learning rate: %f -> %f' % (old_lr, new_lr))
    old_lr = new_lr
    return old_lr


def get_train(traindata_path,weights_path,NAME,lr=2e-4):
    NAME=NAME+'.pth'


    SHAPE=(768,768)
    model = DinkNet34()
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss=dice_bce_loss()
    
    
    
    
    batchsize=2
    img_root=os.path.join(traindata_path,'imgs')
    mask_root=os.path.join(traindata_path,'save_masks')
    
    dataset = ImageFolder(img_root,mask_root,SHAPE)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)
    
    
    
    
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    if os.path.exists(os.path.join(weights_path,NAME)):
        model.load_state_dict(torch.load(os.path.join(weights_path,NAME)))
        print('....load weights successfully')    
    
    mylog = open('logs/'+NAME+'.log','w')
    
    
    old_lr=lr
    no_optim = 0
    total_epoch = 300
    train_epoch_best_loss = 100
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img1,img2,mask1,mask2 in data_loader_iter:
            img1=V(img1.cuda())
            
            img2=V(img2.cuda())
            mask1=V(mask1.cuda())
            mask2=V(mask2.cuda())
            
            #print('.........',img1.shape,img2.shape,mask1.shape,mask2.shape)
            pred1,pred2=model(img1,img2)
            
            loss_1=loss(mask1,pred1)
            loss_2=loss(mask2,pred2)
            loss_all=loss_1+loss_2
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            
            
           
            train_epoch_loss += loss_1.item()
            train_epoch_loss += loss_2.item()
            
        train_epoch_loss /= len(data_loader_iter)
        print ('epoch:',epoch)
        print ('train_loss:',train_epoch_loss)
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            torch.save(model.state_dict(),os.path.join(weights_path,NAME))
    
        if no_optim > 10:
            print('early stop at %d epoch' % epoch)
            break
    
        if no_optim > 6:
            if old_lr < 5e-7:
                break
    
            model.load_state_dict(torch.load(os.path.join(weights_path,NAME)),strict=False)
            old_lr=update_lr(optimizer,old_lr=old_lr,new_lr=5.0, factor = True, mylog = mylog)
        mylog.flush()


if __name__=="__main__":
    get_train('traindata','weights','exp')


