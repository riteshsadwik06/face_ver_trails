import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
# from src import detect_faces
# from facenet_pytorch import MTCNN
from mtcnn import MTCNN
from PIL import Image
import cv2

# Preprocessing
import torchvision.transforms as T
from torch.utils.data import DataLoader

# # Transform landscape/horizontal pictures to portrait/vertical to get same shape for all inputs
# transpose_tfm = Lambda(lambda x: x.transpose((1,0,2)) if x.shape[0] > x.shape[1] else x) # H x W
def getFace(img, detector):
  # print(detector.detect_faces(img))

  img = np.asarray(img)
  # img = img.reshape((img.shape[0],img.shape[1],1))
  detected = detector.detect_faces(img)
  # print(detected)

  if detected:
    # print(detected)
    x, y, w, h = detected[0]['box']

    # cv2_imshow(img[y:y+h, x:x+w, :]) # correct cropped
    face = img[y:y+h, x:x+w, :]

    # face: array [width, height, channel]
    # return Image.fromarray(face)
    face = Image.fromarray(face)
    face = face.convert('L')
    return np.asarray(face)

  # if not detected return original img
  img = Image.fromarray(img)
  img = img.convert('L')
  return np.asarray(img)
  
detector = MTCNN()
extract_face_tfm = Lambda(lambda x: getFace(x, detector))

final_size = (128, 128)
rescale_tfm = Lambda(lambda x: 255*((x - x.min())/(x.max() - x.min())))

transform = {
              'train':T.Compose([
                        extract_face_tfm, # get same shape of inputs with lambda transform
                        T.ToTensor(), # convert each to torch tensor
                        T.Resize(final_size),
                        # rescale_tfm,
                        T.RandomPerspective(distortion_scale=0.2, p=0.5),
                        # T.RandomRotation(degrees=(0, 270)),
                        T.RandomAffine(degrees=(0, 90), 
                                       translate=(0, 0.10),
                                       #scale=(0.5, 0.75)
                                       ),
                        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalize to 0 mean unit variance
                        T.Normalize([0.5], [0.5]), # normalize to 0 mean unit variance
                        # rescale_tfm
                                  ]),
             
             'val':T.Compose([
                        extract_face_tfm, # get same shape of inputs with lambda transform
                        T.ToTensor(), # convert each to torch tensor
                        T.Resize(final_size),
                        # rescale_tfm,
                        T.Normalize([0.5], [0.5]), # normalize to 0 mean unit variance
                        # rescale_tfm
                                  ]),
             
             'test':T.Compose([
                        extract_face_tfm, # get same shape of inputs with lambda transform
                        T.ToTensor(), # convert each to torch tensor
                        T.Resize(final_size),
                        # rescale_tfm,
                        T.Normalize([0.5], [0.5]), # normalize to 0 mean unit variance
                        # rescale_tfm
                                  ]),
             }
              
# define a custom dataset loader for regression purpose
class CustomDataset(Dataset):
    # ADD TEST/VAL LOADING AS LOADING WILL BE PAIRWISE

    def __init__(self, img_root_dir, transform, df, phase = 'train'):

        self.img_root_dir = img_root_dir
        self.transform = transform
        self.df = df 
        self.phase = phase
        # images = os.listdir(img_root_dir)
        # images = self.triplet.keys()
        # self.imgs = natsorted(images) # sort directory files so that input and target are from same image
        # print(self.imgs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        label = data['id']
        # print(data['img_name'], label)

        img_path = os.path.join(self.img_root_dir, data['img_name']) # get file path
        # img = Image.open(img_path).convert("RGB") # read image from file path
        # img = Image.open(img_path)
        # img = img.convert('L')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # img = img.convert('L')
        
        # if self.phase == 'train':
        tensor_img = self.transform[self.phase](np.asarray(img))
        # tensor_img = tensor_img.numpy().astype('uint8')
        # data=Image.fromarray(tensor_img).convert('L')
        # print('data',data.shape) # transformation
        # else:
        #   tensor_img = se
        
        return tensor_img, label




