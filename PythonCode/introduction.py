import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


path = "C:\\Users\\riege\\OneDrive - University of Waterloo\\LaptopSicherung\\Desktop\\Olang_ML4QS\\phyphox 2018-09-03 11-37-59\\Accelerometer.csv"

dataset = pd.read_csv(path, skipinitialspace=True)

#columns = dataset.columns

time = dataset["Time (s)"]
acc_x = dataset["Acceleration x (m/s^2)"]
acc_y = dataset["Acceleration y (m/s^2)"]
acc_z = dataset["Acceleration z (m/s^2)"]

v_x = np.zeros_like(acc_x)
p_x = np.zeros_like(acc_x)
v_y = np.zeros_like(acc_x)
p_y = np.zeros_like(acc_x)
v_z = np.zeros_like(acc_x)
p_z = np.zeros_like(acc_x)

for i in range(len(time) - 1 ):
    delta_t = time[i + 1] - time[i]
    v_x[i+1] = v_x[i] + acc_x[i] * delta_t
    p_x[i+1] = p_x[i] + v_x[i] * delta_t
    v_y[i+1] = v_y[i] + acc_y[i] * delta_t
    p_y[i+1] = p_y[i] + v_y[i] * delta_t
    v_z[i+1] = v_z[i] + acc_z[i] * delta_t
    p_z[i+1] = p_z[i] + v_z[i] * delta_t


plot.hold(True)
plot.plot(time, p_x)
plot.plot(time, p_y)
#plot.plot(time, p_z)
ax = plot.axes()
plot.hold(False)
plot.show()

print(p_x)
print(p_y)
print(p_z)
