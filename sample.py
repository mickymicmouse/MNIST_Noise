# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 08:34:38 2020

@author: seungjun
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

path = r"E:\\replaceMNIST_2"
dis=['training', 'testing']
y=open('log.txt',mode='w',encoding='euc-kr')

for k in dis:
    
    for i in range(len(os.listdir(os.path.join(path, k)))):
        in_path = os.path.join(path, k ,str(i))
        file_name = k+"_class_"+str(i)+".png"
        image_list=os.listdir(in_path)
        y.write(k+"_class_"+str(i)+": "+ str(len(image_list)))
        height=[]
        for j in range(10):
            a=cv2.imread(os.path.join(in_path, image_list[np.random.randint(0,len(image_list))]))
            b=cv2.imread(os.path.join(in_path, image_list[np.random.randint(0,len(image_list))]))
            c=cv2.imread(os.path.join(in_path, image_list[np.random.randint(0,len(image_list))]))
            d=cv2.imread(os.path.join(in_path, image_list[np.random.randint(0,len(image_list))]))
            e=cv2.imread(os.path.join(in_path, image_list[np.random.randint(0,len(image_list))]))
            f=cv2.imread(os.path.join(in_path, image_list[np.random.randint(0,len(image_list))]))
            
            first=np.concatenate([a,b,c,d,e,f],axis=1)
            height.append(first)
        
        final = np.concatenate([height[x] for x in range(10)],axis=0)
        plt.imshow(final)
        cv2.imwrite(file_name, final)
y.close()
