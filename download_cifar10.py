"""
Download and prepare CIFAR-10 dataset in the required folder structure
"""
import os
from torchvision import datasets
import shutil
from PIL import Image

def download_and_prepare_cifar10(save_path):
    """
    Download CIFAR-10 and organize it into class folders
    """
    print(f"Downloading CIFAR-10 to {save_path}...")
    
    # Download CIFAR-10 using torchvision
    temp_path = os.path.join(save_path, 'temp')
    os.makedirs(temp_path, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(root=temp_path, train=True, download=True)
    
    # Create class folders
    classes = train_dataset.classes
    train_path = os.path.join(save_path, 'train')
    
    for class_name in classes:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    
    print("Organizing images into class folders...")
    # Save images in class folders
    for idx, (img, label) in enumerate(train_dataset):
        class_name = classes[label]
        img_path = os.path.join(train_path, class_name, f'{idx}.png')
        img.save(img_path)
        
        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1}/{len(train_dataset)} images")
    
    # Clean up temp folder
    shutil.rmtree(temp_path)
    
    print(f"✅ CIFAR-10 prepared successfully at {train_path}")
    print(f"Total images: {len(train_dataset)}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    save_path = "/home/kaiwei/diffusion/DCTdiff/datasets/cifar10"
    download_and_prepare_cifar10(save_path)
