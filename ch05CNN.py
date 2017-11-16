# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:46:16 2017

@author: SKJ
"""

import glob
image_filenames = glob.glob("./imagenet-dogs/n02*/*.jpg")


from itertools import groupby
from collections import defaultdict

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

#将文件名分解为品种和相对应的文件名，品种对应于文件夹名称
image_filename_with_breed = map(lambda filename: \
(filename.split("/")[1],filename),image_filenames)

#依据品种(上述返回的元组的第0个分量)对图像分组
for dog_breed,breed_images in \
    groupby(image_filename_with_breed,lambda x: x[0]):
        #枚举图像，将20%划入测试集
    for i,breed_images in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_images[1])
        else:
            training_dataset[dog_breed].append(breed_images[1])
        
    #检查每个品种的测试图像是否至少有全部图像的18%
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])
    
    assert round(breed_testing_count / (breed_training_count \
                 + breed_testing_count),2) > 0.18,"Not enough \
                 testing images"        


def write