import os
import shutil
import random
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

dataset_dir = './'
dataset_folder = 'CNNStegoCSV'
dataset_path = os.path.join(dataset_dir, dataset_folder)
training_folder_dir = './Linnaeus 5 64X64/train'

def set_dataset(a):
    dataset_path = a
    return dataset_path

def delete_image(image_path):
    if '.txt' in image_path:
        os.remove(image_path)
        return

    img = Image.open(image_path)
    img_shape = transforms.ToTensor()(img).size()
    if img_shape[0] == 1:
        os.remove(image_path)


if not os.path.exists(dataset_path+"/train"):
    os.makedirs(dataset_path+"/train")
if not os.path.exists(dataset_path+"/valid"):
    os.makedirs(dataset_path+"/valid")

for i, folder in enumerate(os.listdir(training_folder_dir)):
    files = os.listdir(os.path.join(training_folder_dir, folder, 'images'))

    for _file in files:
        delete_image(os.path.join(training_folder_dir, folder, 'images', _file))

    selected_files = random.sample(files, 64)
    validation_images, train_images = selected_files[:30], selected_files[30:]
    for image in train_images:
        shutil.copyfile(os.path.join(training_folder_dir, folder, 'images', image),
                        os.path.join(dataset_path, 'train', image))
    for image in validation_images:
        shutil.copyfile(os.path.join(training_folder_dir, folder, 'images', image),
                        os.path.join(dataset_path, 'valid', image))

training_images = os.listdir(os.path.join(dataset_path, 'train'))
random.shuffle(training_images)

cover_images = training_images[:30] #30
secret_images_1 = training_images[30:80] #50
secret_images_2 = training_images[80:130] #50
secret_images_3 = training_images[130:] #sisa
print(len(training_images))
print(len(cover_images))
print(len(secret_images_1))
print(len(secret_images_2))
print(len(secret_images_3))
dataset = []
for i in range(30):
    dataset.append({
        'cover_image': cover_images[i],
        'secret_image_1': secret_images_1[i],
        'secret_image_2': secret_images_2[i],
        'secret_image_3': secret_images_3[i]
    })

dataframe = pd.DataFrame(dataset)
dataframe.to_csv('./train_dataset.csv')

validation_images = os.listdir(os.path.join(dataset_path, 'valid'))
random.shuffle(validation_images)

cover_images = validation_images[:10]
secret_images_1 = validation_images[10:30]
secret_images_2 = validation_images[30:50]
secret_images_3 = validation_images[50:]

dataset = []
for i in range(10):
    dataset.append({
        'cover_image': cover_images[i],
        'secret_image_1': secret_images_1[i],
        'secret_image_2': secret_images_2[i],
        'secret_image_3': secret_images_3[i]
    })

dataframe = pd.DataFrame(dataset)
dataframe.to_csv('./validation_dataset.csv')

