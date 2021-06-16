# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:11:34 2021

@author: skukm
"""

import os
import shutil
import random

fdir = r"fdir//" #

flist = os.listdir(fdir+"train")
class_list = os.listdir(fdir+"test")
rate = 0.3
split_list=[]
for cn in class_list:
    tmp=[]
    train_dir = fdir+"train/"+cn+"//"
    val_dir = fdir+"val/"+cn+"//"
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)
    
    for fn in flist:
        if fn[0:2] in cn:
            tmp.append(fn)
    split_list.append(tmp)
    for f in tmp:
        shutil.move(fdir+"train/"+f, train_dir)
        
    for f in random.sample(tmp, int(len(tmp) * rate)):
        shutil.move(train_dir+f, val_dir)