from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_train_pipeline import TrainingPipeline
from cnnClassifier.pipeline.stage_03_evaluation_pipeline import EvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
# data_ingestion = DataIngestion()
# data_ingestion.download_zip()
# data_ingestion.extract_zip_file()
# data_ingestion.split_dataset()

# data_transformer = DataTransformation()
# train_dataset, val_dataset, class_names=data_transformer.initiate_data_transformation()

# model_trainer=ModelTrainer()
# model_trainer.train(train_dataset=train_dataset,val_dataset=val_dataset,class_names=class_names)