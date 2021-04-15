import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 5})

# Data for 10^{-3} to 10^{-2}
x_data = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])
y_data_log3 = np.array([-3.025730628, -2.340865155, -2.050497973, -1.805601902, -1.62680886, -1.492790766, -1.365618077, -1.272583015, -1.176770792, -1.092658067])
y_std_log3 = np.array([-3.363383583,-3.236830361,-3.011046786,-2.635920543,-2.837611921,-2.752922703,-2.796423565,-2.463140825,-2.34431806,-2.371857661])
y_data_log5 = np.array([-3.205451813, -2.442125535, -2.009993524, -1.682500631, -1.451677377, -1.236289302, -1.104290602, -0.967912821, -0.853720393, -0.765824289])
y_std_log5 = np.array([-3.483984267,-3.208842033,-3.009984279,-2.763847552,-2.812772927,-2.534189462,-2.513105499,-2.36407633,-2.477275725,-2.256491947])

y_data3 = 10**y_data_log3
y_data5 = 10**y_data_log5
y_std3 = 10**y_std_log3
y_std5 = 10**y_std_log5

x_data_log = np.log10(x_data)
x0 = np.array([0.0, 0.0])
m3, c3 = np.polyfit(x_data_log, y_data_log3, 1)
m5, c5 = np.polyfit(x_data_log, y_data_log5, 1)
y_ls3 = (10**c3)*np.power(x_data, m3)
y_ls5 = (10**c5)*np.power(x_data, m5)

print(m3)
print(m5)

plt.errorbar(x_data, y_data3, y_std3, capthick=1.5)
plt.plot(x_data, y_ls3, 'r--', label='Distance 3')
plt.errorbar(x_data, y_data5, y_std5, capthick=1.5)
plt.plot(x_data, y_ls5, 'b--', label='Distance 5')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('Logical Error vs Physical Error for Amp Damping Noise')
plt.grid(which='both')
plt.show()
