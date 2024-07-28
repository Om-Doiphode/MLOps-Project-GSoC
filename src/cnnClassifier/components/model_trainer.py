from cnnClassifier import logger
from dataclasses import dataclass
from cnnClassifier.utils.common import read_yaml
from cnnClassifier.constants import *
import tensorflow as tf

@dataclass
class ModelTrainerConfig:
    config = read_yaml(PARAMS_PATH)
    trainer_config = read_yaml(CONFIG_PATH)
    
    root_dir:str=trainer_config.trainer.root_dir
    trained_model_path: str = trainer_config.trainer.trained_model_path
    log_file:str = trainer_config.trainer.log_file
    
    image_size:tuple=tuple(config.IMAGE_SIZE) 
    batch_size:int=config.BATCH_SIZE
    include_top:bool=config.INCLUDE_TOP
    epochs:int=config.EPOCHS
    classes:int=config.CLASSES
    weights:str=config.WEIGHTS
    learning_rate:int=config.LEARNING_RATE
    color_mode:str=config.COLOR_MODE
    class_mode:str=config.CLASS_MODE
    optimizer: str=config.OPTIMIZER
    
class CallbackConfig:
    @staticmethod
    def early_stopping_callback(watch='loss', stop_after=3, min_delta=0.001):
        return tf.keras.callbacks.EarlyStopping(monitor=watch, patience=stop_after, min_delta=min_delta)

    @staticmethod
    def reduce_lr_on_plateau_callback(monitor='loss', factor=0.2, patience=5, min_lr=0.000001, cooldown=0):
        return tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=min_lr, cooldown=cooldown)

    @staticmethod
    def model_checkpoint_callback(path_to_model):
        return tf.keras.callbacks.ModelCheckpoint(filepath=path_to_model, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)

    @staticmethod
    def csv_logger_callback(log_file):
        return tf.keras.callbacks.CSVLogger(log_file)
    
class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
        self.callback_config=CallbackConfig()
        
    def train(self,train_dataset,val_dataset, class_names):
        base_model = tf.keras.applications.ResNet50V2(include_top=self.model_config.include_top,weights=self.model_config.weights,pooling='avg')
        base_model.trainable = False

        i = tf.keras.layers.Input(shape=(224,224,3))
        x = base_model(i, training=False)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

        model = tf.keras.models.Model(i,x)
        if self.model_config.class_mode=='categorical':
            model.compile(optimizer=self.model_config.optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
        elif self.model_config.class_mode=='sparse':
            model.compile(optimizer=self.model_config.optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        else:
            model.compile(optimizer=self.model_config.optimizer,loss='binary',metrics=['accuracy'])
            
        reduceLR=self.callback_config.reduce_lr_on_plateau_callback()
        checkpoint = self.callback_config.model_checkpoint_callback(self.model_config.root_dir)
        csv_logger = self.callback_config.csv_logger_callback(self.model_config.log_file)
        early_stop = self.callback_config.early_stopping_callback()
        callbacks_list = [checkpoint, reduceLR, csv_logger, early_stop]
        
        try:
            logger.info("Model training started")
            model.fit(train_dataset,validation_data=val_dataset,epochs=self.model_config.epochs,callbacks=callbacks_list)
            logger.info("Model training completed")
            
            model.save(self.model_config.trained_model_path)
            logger.info(f"Trained model saved at {self.model_config.trained_model_path}")
            
        except Exception as e:
            logger.info(f"Error: {e}")