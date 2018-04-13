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