import tensorflow as tf
from cnnClassifier import logger
from cnnClassifier.utils.common import read_yaml
from dataclasses import dataclass
from cnnClassifier.constants import *
import os

@dataclass
class DataTransformationConfig:
    params_config = read_yaml(PARAMS_PATH)
    config = read_yaml(CONFIG_PATH)
    
    batch_size: int = params_config.BATCH_SIZE
    image_size:tuple = tuple(params_config.IMAGE_SIZE)
    color_mode:str = params_config.COLOR_MODE
    class_mode:str = params_config.CLASS_MODE
    horizontal_flip:int=params_config.HORIZONTAL_FLIP
    zoom_range:int=params_config.ZOOM_RANGE
    w_shift:int=params_config.W_SHIFT
    h_shift:int=params_config.H_SHIFT
    rotation_range:int=params_config.ROTATION_RANGE
    shear_range:int=params_config.SHEAR_RANGE
    fill_zeros:int=params_config.FILL_ZEROS
    
    dataset_path = config.data_transformation.dataset_path
    
class DataTransformation:
    def __init__(self):
        self.instance_config = DataTransformationConfig()
    
    def initiate_data_transformation(self):
        try:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1 / 255.0,
                rotation_range=self.instance_config.rotation_range,
                zoom_range=self.instance_config.zoom_range,
                width_shift_range=self.instance_config.w_shift,
                height_shift_range=self.instance_config.h_shift,
                shear_range=self.instance_config.shear_range,
                horizontal_flip=self.instance_config.horizontal_flip,
                fill_mode="nearest")

            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

            src_path_train = os.path.join(self.instance_config.dataset_path, 'train')
            src_path_test = os.path.join(self.instance_config.dataset_path, 'val')
            print(src_path_test)

            self.classes = os.listdir(src_path_train)

            height, width, _ = self.instance_config.image_size

            train_generator = train_datagen.flow_from_directory(
                directory=src_path_train,
                target_size=(height, width),
                color_mode=self.instance_config.color_mode,
                batch_size=self.instance_config.batch_size,
                class_mode=self.instance_config.class_mode,
                shuffle=True,
                seed=42
            )

            valid_generator = val_datagen.flow_from_directory(
                directory=src_path_test,
                target_size=(height, width),
                color_mode=self.instance_config.color_mode,
                class_mode=self.instance_config.class_mode,
                shuffle=False,
                seed=42
            )
            class_names = list(train_generator.class_indices.keys())
            return train_generator, valid_generator, class_names
        
        except Exception as e:
            logger.info(e)