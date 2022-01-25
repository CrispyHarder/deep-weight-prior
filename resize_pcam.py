import os 
import torch 
import torchvision.transforms as transforms 
from PIL import Image

path_to_pcam_data = os.path.join('data','pcam','train')

comp = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(32),
    transforms.ToPILImage()
])

i = 0
for id in os.listdir(path_to_pcam_data):
    full_path = os.path.join(path_to_pcam_data,id)
    image = Image.open(full_path)
    image = comp(image)
    image.save(full_path)
    i += 1 
    if i %10000 == 0:
        print(i, 'Images converted ')