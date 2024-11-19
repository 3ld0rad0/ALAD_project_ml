import random
import os
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Created directory: {dir_path}")

def copy_images(images, source_dir, target_dir, class_name):
    for img in images:
        src_path = os.path.join(source_dir, class_name, img)
        dst_path = os.path.join(target_dir, class_name, img)
        create_dir_if_not_exists(os.path.dirname(dst_path))
        copyfile(src_path, dst_path)

def augment_image(image_path, datagen, save_dir, prefix='aug', num_augmented=2):
    img = load_img(image_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=prefix, save_format='png'):
        i += 1
        if i >= num_augmented:
            break

def augment_training_set(source_dir, train_dir, val_dir, train_split, num_augmented):
    create_dir_if_not_exists(train_dir)
    create_dir_if_not_exists(val_dir)

    classes = os.listdir(source_dir)
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[1.3, 1.5]
    )
    
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        if os.path.isdir(class_dir):
            images = [img for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)
            
            train_split_idx = int(len(images) * train_split)
            train_images = images[:train_split_idx]
            val_images = images[train_split_idx:]
            
            logging.info(f"Processing class {cls}: {len(train_images)} training images, {len(val_images)} validation images")
            
            copy_images(train_images, source_dir, train_dir, cls)
            copy_images(val_images, source_dir, val_dir, cls)
            
            for img in train_images:
                src_path = os.path.join(source_dir, cls, img)
                augment_image(src_path, datagen, os.path.join(train_dir, cls), num_augmented=num_augmented)

source_dataset_dir = './data/train_original'
train_dataset_dir = './data/train_aug'
val_dataset_dir = './data/valid'

# Splitta il training set destinando il 10% al validation set
augment_training_set(source_dataset_dir, train_dataset_dir, val_dataset_dir, train_split=0.9, num_augmented=2)