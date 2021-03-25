import numpy as np
import matplotlib.pyplot as plt

# Data for 10^{-3} to 10^{-2}
# x_data = np.array([0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055])
# y_data3 = np.array([0.0068, 0.0097, 0.0118, 0.0178, 0.0203, 0.0244, 0.0288])
# y_data5 = np.array([0.0066, 0.0085, 0.0156, 0.0192, 0.0244, 0.0352, 0.0469])
# y_data7 = np.array([0.0104, 0.021, 0.0294, 0.0387, 0.063])
# x_data7 = np.array([0.0035, 0.004, 0.0045, 0.005, 0.0055])

x_data = np.array([0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045])
y_data3 = np.array([0.0023, 0.0065, 0.0097, 0.012, 0.019, 0.027, 0.032, 0.042])
y_data5 = np.array([0.0015, 0.0044, 0.01, 0.02, 0.029, 0.039, 0.059, 0.076])
y_data7 = np.array([0.0004, 0.002, 0.007, 0.014, 0.031, 0.047, 0.068, 0.107])

x_data_log = np.log10(x_data)
# x_data_log7 = np.log10(x_data7)
y_data_log3 = np.log10(y_data3)
y_data_log5 = np.log10(y_data5)
y_data_log7 = np.log10(y_data7)
x0 = np.array([0.0, 0.0])
m3, c3 = np.polyfit(x_data_log, y_data_log3, 1)
m5, c5 = np.polyfit(x_data_log, y_data_log5, 1)
m7, c7 = np.polyfit(x_data_log, y_data_log7, 1)
y_ls3 = (10**c3)*np.power(x_data, m3)
y_ls5 = (10**c5)*np.power(x_data, m5)
y_ls7 = (10**c7)*np.power(x_data, m7)

# print(m)
# print(c)

plt.loglog(x_data, y_data3, 'ro')
plt.loglog(x_data, y_ls3, 'r--', label='Distance 3')
plt.loglog(x_data, y_data5, 'bo')
plt.loglog(x_data, y_ls5, 'b--', label='Distance 5')
plt.loglog(x_data, y_data7, 'yo')
plt.loglog(x_data, y_ls7, 'y--', label='Distance 7')
plt.legend()
plt.title('Logical Error vs Physical Error for Amp Damping Noise')
plt.grid(which='both')
plt.show()
