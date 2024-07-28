import os
import shutil
from dataclasses import dataclass
from cnnClassifier.utils.common import read_yaml
from pathlib import Path
from cnnClassifier.constants import *
import stat
from kaggle.api.kaggle_api_extended import KaggleApi
from cnnClassifier import logger
import zipfile
import pandas as pd
from cnnClassifier.components.data_transformation import DataTransformation
import splitfolders

@dataclass
class DataIngestionConfig:
    config = read_yaml(CONFIG_PATH)
    download_url:str=config.data_ingestion.source_URL
    root_dir:str = config.data_ingestion.root_dir
    unzip_dir:str=config.data_ingestion.unzip_dir
    zip_file:str=config.data_ingestion.local_data_file
    
    params_config = read_yaml(PARAMS_PATH)
    test_ratio:int = params_config.TEST_RATIO
    
class DataIngestion:
    def __init__(self):
        self.instance_config=DataIngestionConfig()
        self.dataset_path=os.path.join(self.instance_config.unzip_dir,'Gano-Cat-Breeds-V1_1')
        
    def make_dataFrame(self,data):
        path=Path(data)
        filepaths=list(path.glob(r"*/*.jpg"))
        labels=list(map(lambda x: os.path.split(os.path.split(x)[0])[1],filepaths))
        d1=pd.Series(filepaths,name='filepaths').astype(str)
        d2=pd.Series(labels,name='labels')
        df=pd.concat([d1,d2],axis=1)
        return df
    
    def download_zip(self):
        if not os.path.exists(self.dataset_path):
            # os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
            # shutil.copy('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
            # os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), stat.S_IRUSR | stat.S_IWUSR)
            
            api = KaggleApi()
            api.authenticate()
            
            logger.info("Downloading dataset")
            api.dataset_download_files(self.instance_config.download_url, path=self.instance_config.root_dir, unzip=False)
            logger.info("Dataset downloaded")
        
    def extract_zip_file(self):
        if not os.path.exists(self.dataset_path):
            logger.info("Extracting dataset")
            os.makedirs(self.instance_config.unzip_dir,exist_ok=True)
            with zipfile.ZipFile(self.instance_config.zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.instance_config.unzip_dir)
            logger.info(f"Dataset extracted to {self.instance_config.unzip_dir}")
            logger.info("Deleting local zip file to free space")
            os.remove(self.instance_config.zip_file)
            logger.info("Zip file deleted successfully")
        
    def split_dataset(self):
        try:  
            splitfolders.ratio(self.dataset_path, seed=123, output=self.instance_config.unzip_dir, ratio=(1 - self.instance_config.test_ratio, self.instance_config.test_ratio/2, self.instance_config.test_ratio/2))
            logger.info("Dataset splitted into train, test and val folders")
            shutil.rmtree(self.dataset_path)
            logger.info("Old dataset deleted successfully")
                
        except Exception as e:
            logger.info(e)

        

    
if __name__=="__main__":
    data_ingestion = DataIngestion()
    data_ingestion.download_zip()
    data_ingestion.extract_zip_file()
    data_ingestion.split_dataset()
    
    data_transformer = DataTransformation()
    data_transformer.initiate_data_transformation()