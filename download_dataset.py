import io
import os
import torch
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from torchvision.transforms.functional import to_tensor


class TensorFlowFlowersDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', save_dir='train_dataset'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.data = tfds.load('tf_flowers', split=split, as_supervised=True)

        if save_dir == 'train_dataset':
            self.data = self.data.skip(400)
        else:
            self.data = self.data.take(400)

        self._download_and_save()

    def __len__(self):
        # This method of determining length won't work directly with TensorFlow datasets.
        # It's better to convert the dataset to a list (if small enough) or keep track of the length manually.
        # For demonstration purposes, we'll set a placeholder value and recommend counting the dataset size appropriately.
        return len(list(self.data))

    @staticmethod
    def _transform(image):
        cropped = tf.image.resize(image, (128, 128))
        lr = tf.image.resize(cropped, (32, 32))
        cropped = tf.cast(cropped / 255.0, tf.float32)
        lr = tf.cast(lr / 255.0, tf.float32)
        return lr.numpy(), cropped.numpy()

    def _download_and_save(self):
        for i, (image, label) in enumerate(self.data):
            lr, hr = self._transform(image)
            lr_img = Image.fromarray((lr * 255).astype('uint8'))
            hr_img = Image.fromarray((hr * 255).astype('uint8'))
            lr_img_path = os.path.join(self.save_dir, f"image_{i}_lr.jpg")
            hr_img_path = os.path.join(self.save_dir, f"image_{i}.jpg")
            lr_img.save(lr_img_path)
            hr_img.save(hr_img_path)

    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.save_dir, f"image_{idx}_lr.jpg")
        hr_img_path = os.path.join(self.save_dir, f"image_{idx}.jpg")

        lr_img = Image.open(lr_img_path)
        hr_img = Image.open(hr_img_path)

        return to_tensor(lr_img), to_tensor(hr_img)

# Usage example (specify the save directory when calling the function)
save_directory = "train_dataset"  
train_dataset = TensorFlowFlowersDataset(save_dir=save_directory)
save_directory = "test_dataset"  
train_dataset = TensorFlowFlowersDataset(save_dir=save_directory)
