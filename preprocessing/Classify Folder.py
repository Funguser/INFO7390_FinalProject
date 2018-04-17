# MIT License

# Copyright (c) 2018 ZiyaoQiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
from glob import glob
import os
from shutil import copyfile,rmtree

train_images = glob("Datas/train/*jpg")
test_images = glob("Datas/test/*jpg")
df = pd.read_csv("Datas/train.csv")

ImageToLabelDict = dict( zip( df["Image"], df["Id"]))
df["Image"].head()

new_data_folder ='keras/train/'
if(os.path.exists(new_data_folder)):
    rmtree(new_data_folder)

def save_images(df,ImageToLabelDict):
    for key in df["Image"]:
        image_class = ImageToLabelDict[key]
        img_full_path = new_data_folder + image_class + '/' + key
        img_class_path = new_data_folder + image_class
        if not os.path.exists(img_class_path):
            os.makedirs(img_class_path)
        copyfile("Datas/train/"+key, img_full_path)

save_images(df, ImageToLabelDict)