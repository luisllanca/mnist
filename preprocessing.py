from PIL import Image
from torchvision import datasets,transforms
import torch
from pathlib import Path
import argparse
import os

def image_to_tensor(img_path):
    image = Image.open(img_path)
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    return tensor_image
def load_image_label(path,path_tensors):
    images_tensor_train = []
    labels_tensor_train = []
    path_train = os.path.join(path,'train')
    for label in os.listdir(path_train):
        path_label = os.path.join(path_train,label)
        for image_name in os.listdir(path_label):
            image = image_to_tensor(os.path.join(path_label,image_name))
            label_tensor = torch.tensor(int(label))
            images_tensor_train.append(image)
            labels_tensor_train.append(label_tensor)
    images_tensor_test = []
    labels_tensor_test = []
    path_test = os.path.join(path,'test')
    for label in os.listdir(path_test):
        path_label = os.path.join(path_test,label)
        for image_name in os.listdir(path_label):
            image = image_to_tensor(os.path.join(path_label,image_name))
            label_tensor = torch.tensor(int(label))
            images_tensor_test.append(image)
            labels_tensor_test.append(label_tensor)
    tensor_itr = torch.stack(images_tensor_train)
    tensor_ltr = torch.stack(labels_tensor_train)
    tensor_ite = torch.stack(images_tensor_test)
    tensor_lte = torch.stack(labels_tensor_test)
    os.makedirs(path_tensors,exist_ok=True)
    torch.save(tensor_itr,os.path.join(path_tensors,'train_images.pt'))
    torch.save(tensor_ltr, os.path.join(path_tensors, 'train_labels.pt'))
    torch.save(tensor_ite, os.path.join(path_tensors, 'test_images.pt'))
    torch.save(tensor_lte, os.path.join(path_tensors, 'test_labels.pt'))
def main():
    parser = argparse.ArgumentParser(description="Cargar imágenes, convertirlas a tensores y guardarlas.")
    parser.add_argument("path_dataset",type=str,help="Ruta del directorio que contiene el dataset.")
    parser.add_argument("path_output",type=str,help="Ruta donde se guardarán los tensores.")
    args = parser.parse_args()
    load_image_label(args.path_dataset,args.path_output)
if __name__ == "__main__":
    main()