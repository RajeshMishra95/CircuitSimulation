import numpy as np
import matplotlib.pyplot as plt

# Data for 10^{-3} to 10^{-2}
x_data = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])
y_data3 = np.array([0.0012, 0.0053, 0.0095, 0.0138, 0.026, 0.0338, 0.042, 0.056, 0.0648, 0.0838])
y_data5 = np.array([0.0006, 0.0033, 0.0123, 0.0265, 0.035, 0.048, 0.096, 0.106, 0.128, 0.164])
y_data7 = np.array([0.00063, 0.005, 0.021, 0.032, 0.086, 0.126, 0.14, 0.2, 0.27])
x_data7 = np.array([0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

# Data for 5x10^{-4} to 10^{-3}
# x_data = np.array([0.0005, 0.00055, 0.00061, 0.00067, 0.00072, 0.00078, 0.00083, 0.00089, 0.00094, 0.001])
# y_data3 = np.array([0.00033, 0.00029, 0.00047, 0.00052, 0.0007, 0.00060, 0.0009, 0.0011, 0.0012, 0.001])

x_data_log = np.log10(x_data)
x_data_log7 = np.log10(x_data7)
y_data_log3 = np.log10(y_data3)
y_data_log5 = np.log10(y_data5)
y_data_log7 = np.log10(y_data7)
x0 = np.array([0.0, 0.0])
m3, c3 = np.polyfit(x_data_log, y_data_log3, 1)
m5, c5 = np.polyfit(x_data_log, y_data_log5, 1)
m7, c7 = np.polyfit(x_data_log7, y_data_log7, 1)
y_ls3 = (10**c3)*np.power(x_data, m3)
y_ls5 = (10**c5)*np.power(x_data, m5)
y_ls7 = (10**c7)*np.power(x_data7, m7)

# print(m)
# print(c)

plt.loglog(x_data, y_data3, 'ro')
plt.loglog(x_data, y_ls3, 'r--', label='Distance 3')
plt.loglog(x_data, y_data5, 'bo')
plt.loglog(x_data, y_ls5, 'b--', label='Distance 5')
plt.loglog(x_data7, y_data7, 'yo')
plt.loglog(x_data7, y_ls7, 'y--', label='Distance 7')
plt.legend()
plt.title('Logical Error vs Physical Error')
plt.grid(which='both')
plt.show()
