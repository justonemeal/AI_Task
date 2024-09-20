import glob
import os
from torch.utils.data import Dataset
import cv2
import torch

data_dir = "/content/drive/MyDrive/lettuce/train"

def image_rad():

  all_image_path = []

  all_image_path.extend(glob.glob(data_dir + '/**/*.jpg'))
  all_image_path.extend(glob.glob(data_dir + '/**/*.png'))
  all_image_path.extend(glob.glob(data_dir + '/**/*.jpeg'))

  print(f"Total images: {len(all_image_path)}")
  print(all_image_path[:5])

  return all_image_path

def image_path_lbale():
  image_path_lable = []

  all_image_path = image_rad()

  for path in all_image_path:
    label = os.path.basename(os.path.dirname(path))
    image_path_lable.append((path, label))

  return image_path_lable

train_data_dir = "/content/drive/MyDrive/lettuce/train/"

class Chest_dataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.files_path = image_path_lbale()
    self.transform = transform

  def __len__(self):
    return len(self.files_path)

  def __getitem__(self, index):
    image_file = self.files_path[index][0]
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_VGR2GB)

    target = self.files_path[index][1]

    if self.transform:
      image = self.transform(image)
      target = torch.Tensor([target]).long()

    return {"image": image, "target": target}


