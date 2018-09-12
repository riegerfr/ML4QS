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
from Chapter3.DataTransformation import LowPassFilter
from Chapter4.FrequencyAbstraction import FourierTransformation
import re
import os
import inspect

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path,"..\\data\daten-neu")

#print os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

folders = os.listdir(path)  # labels

milliseconds_per_instance = 50
samples_dataframe = pd.DataFrame()
frames = []

for label in folders:
    samples = os.listdir(path + "\\" + label)
    for sample in samples:
        sensors = os.listdir(path + "\\" + label + "\\" + sample)
        dataSet = CreateDataset(path + "\\" + label + "\\" + sample + "\\", milliseconds_per_instance)
        # for sensor in sensors:
        dataSet.add_numerical_dataset("Accelerometer.csv", 'Time (s)', ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'], 'avg',
                                      "Accelerometer")
        dataSet.add_numerical_dataset("Gyroscope.csv", 'Time (s)', ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'], 'avg',
                                      "Gyroscope")

        dataSet.data_table = dataSet.data_table[~(np.isnan(dataSet.data_table['GyroscopeZ (rad/s)']))]  # todo: useful?

        length = len(dataSet.data_table)

        dataSet.data_table = dataSet.data_table[(length - 53): (length - 1)]  # same length for every sample

        FreqAbs = FourierTransformation()
        transformations = []
        number_frequencies = 50
        for column in list(dataSet.data_table.columns):
            transformation = np.abs(np.fft.fft(dataSet.data_table[column], number_frequencies))
            transformations.append(transformation)

        cutoff_frequency = 20
        sampling_frequency = 50
        order = 3
        LowPass = LowPassFilter()
        new_dataset = LowPass.low_pass_filter(dataSet.data_table, 'AccelerometerX (m/s^2)', sampling_frequency,
                                              cutoff_frequency,
                                              order=order, phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'AccelerometerY (m/s^2)', sampling_frequency,
                                              cutoff_frequency, order=order,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'AccelerometerZ (m/s^2)', sampling_frequency,
                                              cutoff_frequency, order=order,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'GyroscopeX (rad/s)', sampling_frequency, cutoff_frequency,
                                              order=order,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'GyroscopeY (rad/s)', sampling_frequency, cutoff_frequency,
                                              order=order,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'GyroscopeZ (rad/s)', sampling_frequency, cutoff_frequency,
                                              order=order,
                                              phase_shift=True)

        flattened_values = new_dataset.values.flatten()

        for transformation in transformations:
            flattened_values = np.append(flattened_values, transformation)

        df = pd.DataFrame(data=flattened_values).T
        df['class'] = [str(label)]

        frames.append(df)

result = pd.concat(frames)

result.columns = result.columns.astype(str)

#result = result.sample(frac=1)
prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(result, ['class'], 'unlike', 0.8,
                                                                               filter=True, temporal=False)

number_training_samples = len(train_X)
val_split = int(0.7 * number_training_samples)
val_X = train_X[val_split:-1]
val_y = train_y[val_split:-1]
train_X = train_X[0:val_split - 1]
train_y = train_y[0:val_split - 1]



learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

print(len( val_X))

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(train_X,
                                                                                                        train_y,
                                                                                                        val_X,
                                                                                                        hidden_layer_sizes=(
                                                                                                            250, 50,),
                                                                                                        # alpha=reg_param,
                                                                                                        max_iter=500,
                                                                                                        gridsearch=False)

performance_tr_nn = eval.accuracy(train_y, class_train_y)
performance_te_nn = eval.accuracy(val_y, class_test_y)

print(performance_te_nn)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
    train_X, train_y, val_X, gridsearch=True)
performance_tr_nn = eval.accuracy(train_y, class_train_y)
performance_te_nn = eval.accuracy(val_y, class_test_y)

print(performance_te_nn)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(train_X, train_y,
                                                                                           val_X,
                                                                                           gridsearch=True)
performance_tr_rf = eval.accuracy(train_y, class_train_y)
performance_te_rf = eval.accuracy(val_y, class_test_y)
print(performance_te_rf)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
    train_X, train_y, val_X, gridsearch=True)
performance_tr_svm = eval.accuracy(train_y, class_train_y)
performance_te_svm = eval.accuracy(val_y, class_test_y)

print(performance_te_svm)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(train_X,
                                                                                                train_y,
                                                                                                val_X,
                                                                                                gridsearch=True)
performance_tr_knn = eval.accuracy(train_y, class_train_y)
performance_te_knn = eval.accuracy(val_y, class_test_y)

print(performance_te_knn)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X, train_y,
                                                                                           val_X,
                                                                                           gridsearch=True)
performance_tr_dt = eval.accuracy(train_y, class_train_y)
performance_te_dt = eval.accuracy(val_y, class_test_y)

print(performance_te_dt)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(train_X, train_y,
                                                                                         val_X)
performance_tr_nb = eval.accuracy(train_y, class_train_y)
performance_te_nb = eval.accuracy(val_y, class_test_y)

print(performance_te_nb)
