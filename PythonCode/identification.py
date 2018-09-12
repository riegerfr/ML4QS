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

path = "C:\\Users\\riege\\OneDrive - University of Waterloo\\LaptopSicherung\\Desktop\\Olang_ML4QS\\daten-neu"
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

        cutofffrequency = 20
        LowPass = LowPassFilter()
        new_dataset = LowPass.low_pass_filter(dataSet.data_table, 'AccelerometerX (m/s^2)', 50, cutofffrequency,
                                              order=3, phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'AccelerometerY (m/s^2)', 50, cutofffrequency, order=3,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'AccelerometerZ (m/s^2)', 50, cutofffrequency, order=3,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'GyroscopeX (rad/s)', 50, cutofffrequency, order=3,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'GyroscopeY (rad/s)', 50, cutofffrequency, order=3,
                                              phase_shift=True)
        new_dataset = LowPass.low_pass_filter(new_dataset, 'GyroscopeZ (rad/s)', 50, cutofffrequency, order=3,
                                              phase_shift=True)

        FreqAbs = FourierTransformation()

        transformation = np.abs(np.fft.fft(new_dataset['GyroscopeZ (rad/s)'], 50))  # todo: other sensors

        # gyro_ft_real, gyro_ft_imag = FreqAbs.find_fft_transformation(new_dataset['GyroscopeZ (rad/s)'], 50)
        # data_table = FreqAbs.abstract_frequency(copy.deepcopy(new_dataset), ['GyroscopeZ (rad/s)'], 20, 50)
        # Get the frequencies from the columns....
        # frequencies = []
        # values = []
        # for col in data_table.columns:
        #    val = re.findall(r'freq_\d+\.\d+_Hz', col)
        #    if len(val) > 0:
        #        frequency = float((val[0])[5:len(val) - 4])
        #        frequencies.append(frequency)
        #        values.append(data_table.ix[data_table.index, col])
        # print new_dataset

        flattened_values = new_dataset.values.flatten()
        flattened_values = np.append(flattened_values, transformation)

        df = pd.DataFrame(data=flattened_values).T
        df['class'] = [str(label)]

        #  samples_dataframe = samples_dataframe.append(df)
        frames.append(df)

result = pd.concat(frames)

result.columns = result.columns.astype(str)

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(result, ['class'], 'unlike', 0.7,
                                                                               filter=True, temporal=False)

# acc_dataset = pd.read_csv(path + "Accelerometer.csv", skipinitialspace=True)
# mag_dataset = pd.read_csv(path + "Magnetometer.csv", skipinitialspace=True)

# time = acc_dataset["Time (s)"]
# acc_x = acc_dataset["Acceleration x (m/s^2)"]
# acc_y = acc_dataset["Acceleration y (m/s^2)"]
# acc_z = acc_dataset["Acceleration z (m/s^2)"]

# learner = ClassificationAlgorithms()

# prepare = PrepareDatasetForLearning()


learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(train_X,
                                                                                                        train_y,
                                                                                                        test_X,
                                                                                                        hidden_layer_sizes=(
                                                                                                            250,),
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
                                                                                           gridsearch=True)
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
                                                                                           gridsearch=True)
performance_tr_dt = eval.accuracy(train_y, class_train_y)
performance_te_dt = eval.accuracy(test_y, class_test_y)

print(performance_te_dt)


class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(train_X, train_y,
                                                                                         test_X)
performance_tr_nb = eval.accuracy(train_y, class_train_y)
performance_te_nb = eval.accuracy(test_y, class_test_y)

print(performance_te_nb)

