import tensorflow as tf
from dataclasses import dataclass
from cnnClassifier.utils.common import read_yaml, save_json
from cnnClassifier.constants import *
import os
import numpy as np

@dataclass
class EvaluationConfig:
    config = read_yaml(CONFIG_PATH) 
    params_config = read_yaml(PARAMS_PATH)
    trained_model_path:str = config.trainer.trained_model_path
    dataset:str = config.data_ingestion.unzip_dir
    image_size:tuple=tuple(params_config.IMAGE_SIZE)
    color_mode:str=params_config.COLOR_MODE
    class_mode:str=params_config.CLASS_MODE
    
class Evaluation:
    def __init__(self):
        self.config = EvaluationConfig()
        
    def load_model(self):
        self.model = tf.keras.models.load_model(self.config.trained_model_path)
        
    def evaluate(self):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
        src_path_test = os.path.join(self.config.dataset, 'test')
        height, width, _ = self.config.image_size
        test_generator = test_datagen.flow_from_directory(
                directory=src_path_test,
                target_size=(height, width),
                color_mode=self.config.color_mode,
                batch_size=1,
                class_mode=self.config.class_mode,
                shuffle=False,
                seed=42
            )
        
        self.load_model()
        self.score=self.model.evaluate(test_generator)
    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
        
if __name__=="__main__":
    eval_obj = Evaluation()
    eval_obj.evaluate()
    eval_obj.save_score()