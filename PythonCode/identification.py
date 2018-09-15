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
path = os.path.join(my_path, "..\\data\daten-neu")

# print os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

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
            transformations.append((column,transformation))

        cutoff_frequency = 20
        sampling_frequency = 50
        order = 3
        LowPass = LowPassFilter()

        if len(dataSet.data_table[ 'AccelerometerX (m/s^2)']) < 50:
            print path + "\\" + label + "\\" + sample

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

        # flattened_values = new_dataset.values.flatten()
        flattened_values = pd.DataFrame()
        flattened_values['gyroX_avg'] = [np.mean(new_dataset['GyroscopeX (rad/s)'])]
        flattened_values["gyroY_avg"] = np.mean(new_dataset['GyroscopeY (rad/s)'])
        flattened_values["gyroZ_avg"] = np.mean(new_dataset['GyroscopeZ (rad/s)'])

        flattened_values["gyroX_min"] = np.min(new_dataset['GyroscopeX (rad/s)'])
        flattened_values["gyroY_min"] = np.min(new_dataset['GyroscopeY (rad/s)'])
        flattened_values["gyroZ_min"] = np.min(new_dataset['GyroscopeZ (rad/s)'])

        flattened_values["gyroX_max"] = np.max(new_dataset['GyroscopeX (rad/s)'])
        flattened_values["gyroY_max"] = np.max(new_dataset['GyroscopeY (rad/s)'])
        flattened_values["gyroZ_max"] = np.max(new_dataset['GyroscopeZ (rad/s)'])

        flattened_values["gyroX_sd"] = np.std(new_dataset['GyroscopeX (rad/s)'])
        flattened_values["gyroY_sd"] = np.std(new_dataset['GyroscopeY (rad/s)'])
        flattened_values["gyroZ_sd"] = np.std(new_dataset['GyroscopeZ (rad/s)'])

        flattened_values["accX_avg"] = np.mean(new_dataset['AccelerometerX (m/s^2)'])
        flattened_values["accY_avg"] = np.mean(new_dataset['AccelerometerY (m/s^2)'])
        flattened_values["accZ_avg"] = np.mean(new_dataset['AccelerometerZ (m/s^2)'])

        flattened_values["accX_min"] = np.min(new_dataset['AccelerometerX (m/s^2)'])
        flattened_values["accY_min"] = np.min(new_dataset['AccelerometerY (m/s^2)'])
        flattened_values["accZ_min"] = np.min(new_dataset['AccelerometerZ (m/s^2)'])

        flattened_values["accX_max"] = np.max(new_dataset['AccelerometerX (m/s^2)'])
        flattened_values["accY_max"] = np.max(new_dataset['AccelerometerY (m/s^2)'])
        flattened_values["accZ_max"] = np.max(new_dataset['AccelerometerZ (m/s^2)'])

        flattened_values["accX_sd"] = np.std(new_dataset['AccelerometerX (m/s^2)'])
        flattened_values["accY_sd"] = np.std(new_dataset['AccelerometerY (m/s^2)'])
        flattened_values["accZ_sd"] = np.std(new_dataset['AccelerometerZ (m/s^2)'])

        for columnname, values  in transformations:
            # flattened_values = np.append(flattened_values, transformation)

            flattened_values[columnname +"_argmax"] = (np.argmax(values))
            flattened_values[columnname +"_max"] = (np.max(values))
            #flattened_values = flattened_values.join(np.max(transformation))
            #flattened_values = np.append(flattened_values, np.argmax(transformation))
            #flattened_values = np.append(flattened_values, np.max(transformation))

        df = pd.DataFrame(data=flattened_values)
        df['class'] = str(label)

        frames.append(df)

result = pd.concat(frames)

result.columns = result.columns.astype(str)

# result = result.sample(frac=1)
prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(result, ['class'], 'unlike', 0.8,
                                                                               filter=True, temporal=False)

#number_training_samples = len(train_X)
#val_split = int(0.7 * number_training_samples)
#val_X = train_X[val_split:-1]
#val_y = train_y[val_split:-1]
#train_X = train_X[0:val_split - 1]
#train_y = train_y[0:val_split - 1]

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

print(len(train_X))
print(len(test_X))

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(train_X,
                                                                                                        train_y,
                                                                                                        test_X,
                                                                                                        hidden_layer_sizes=(
                                                                                                            250, 50,),
                                                                                                        # alpha=reg_param,
                                                                                                        max_iter=500,
                                                                                                        gridsearch=False)

performance_tr_nn = eval.accuracy(train_y, class_train_y)
performance_te_nn = eval.accuracy(test_y, class_test_y)

print(performance_te_nn)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
    train_X, train_y, test_X, gridsearch=True)
performance_tr_nn = eval.accuracy(train_y, class_train_y)
performance_te_nn = eval.accuracy(test_y, class_test_y)

print(performance_te_nn)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(train_X, train_y,
                                                                                           test_X,
                                                                                           gridsearch=True
                                                                                           ,print_model_details=True)
performance_tr_rf = eval.accuracy(train_y, class_train_y)
performance_te_rf = eval.accuracy(test_y, class_test_y)
print(performance_te_rf)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
    train_X, train_y, test_X, gridsearch=True)
performance_tr_svm = eval.accuracy(train_y, class_train_y)
performance_te_svm = eval.accuracy(test_y, class_test_y)

print(performance_te_svm)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(train_X,
                                                                                                train_y,
                                                                                                test_X,
                                                                                                gridsearch=True)
performance_tr_knn = eval.accuracy(train_y, class_train_y)
performance_te_knn = eval.accuracy(test_y, class_test_y)

print(performance_te_knn)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X, train_y,
                                                                                           test_X,
                                                                                           gridsearch=True, print_model_details = True, export_tree_path=my_path+ "\\..\\data\\")
performance_tr_dt = eval.accuracy(train_y, class_train_y)
performance_te_dt = eval.accuracy(test_y, class_test_y)

print(performance_te_dt)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(train_X, train_y,
                                                                                         test_X)
performance_tr_nb = eval.accuracy(train_y, class_train_y)
performance_te_nb = eval.accuracy(test_y, class_test_y)

print(performance_te_nb)
