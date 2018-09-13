import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split
import os
from Chapter2.CreateDataset import CreateDataset


def cut_length(path_csv, length):
    data = pd.read_csv(path_csv)
    max_time = np.max(data["Time (s)"])
    if (max_time < 3 and (path_csv.__contains__('Gyroscope') or path_csv.__contains__('Accelerometer'))):
        print path_csv
        print max_time
    data_filtered = data[data['Time (s)'] > max_time - length]
    data_filtered['Time (s)'] = data_filtered['Time (s)'] * 1000000000
    data_filtered.to_csv(path_csv)

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path,"..\\data\daten-neu")
folders = os.listdir(path)  # labels


for label in folders:
    samples = os.listdir(path + "\\" + label)
    for sample in samples:
        sensors = os.listdir(path + "\\" + label + "\\" + sample)

        for sensor in sensors:
            cut_length(path + "\\" + label + "\\" + sample + "\\" + sensor, 3.0)
