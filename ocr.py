import pandas as pd
import cv2
import numpy as np
from sklearn import preprocessing

label = []
images = []

def loadData():
    data=pd.read_csv('gbn-fonts.tsv',sep='\t')
    images = "dataset/Fraktur/" + data['segment_id'] + ".png"
    label = data['text_equiv']

    data = [cv2.resize(image, (224, 224)) for image in images]
    data = np.array(data, dtype="float32")

    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    le = preprocessing.LabelBinarizer()
    labels = le.fit_transform(labels)

