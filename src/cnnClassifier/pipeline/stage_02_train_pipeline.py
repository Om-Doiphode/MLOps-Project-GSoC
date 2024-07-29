from cnnClassifier.components.data_transformation import DataTransformation
from cnnClassifier.components.model_trainer import ModelTrainer
from cnnClassifier import logger
import pickle

STAGE_NAME = "Training stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        data_transformer = DataTransformation()
        train_dataset, val_dataset, class_names=data_transformer.initiate_data_transformation()
        
        with open('classes.pkl', 'wb') as f:
            pickle.dump(class_names, f)
        model_trainer=ModelTrainer()
        model_trainer.train(train_dataset=train_dataset,val_dataset=val_dataset,class_names=class_names)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e