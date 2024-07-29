import tensorflow as tf
from dataclasses import dataclass
from cnnClassifier.utils.common import read_yaml, save_json
from cnnClassifier.constants import *
import os
import numpy as np
import dagshub
import mlflow
from urllib.parse import urlparse

@dataclass
class EvaluationConfig:
    config = read_yaml(CONFIG_PATH) 
    params_config = read_yaml(PARAMS_PATH)
    trained_model_path:str = config.trainer.trained_model_path
    dataset:str = config.data_ingestion.unzip_dir
    mlflow_uri:str=config.evaluation.mlflow_uri
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
        
    def log_into_mlflow(self):
        import logging
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        logging.basicConfig(level=logging.DEBUG)

        with mlflow.start_run() as run:
            mlflow.log_params(self.config.params_config)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            mlflow.keras.log_model(self.model, "model", registered_model_name="Resnet50V2Model")

        
if __name__=="__main__":
    eval_obj = Evaluation()
    eval_obj.evaluate()
    eval_obj.save_score()
    eval_obj.log_into_mlflow()