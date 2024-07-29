import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
    def predict(self):
        model = tf.keras.models.load_model('artifacts/training/model.h5')
        image = self.filename
        img = tf.keras.preprocessing.image.load_img(image)
        img_arr = tf.keras.utils.img_to_array(img)
        img_arr = img_arr/255.
        img_arr = tf.image.resize(img_arr,[224,224])
        with open('classes.pkl', 'rb') as f:
            class_names = pickle.load(f)
        pred = np.argmax(model.predict(np.expand_dims(img_arr, axis=0)))
        prediction = class_names[pred]
        
        return [{ "image" : prediction}]
        
if __name__ == "__main__":
    obj = PredictionPipeline('artifacts/data_ingestion/val/Egyptian Mau/Egyptian_Mau_172.jpg')
    obj.predict()