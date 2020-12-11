import matplotlib.pyplot as plt 

x = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]
distance_3 = []
f = open("distance_3.txt", "r")
for y in f:
    distance_3.append(eval(y))
f.close()

distance_5 = []
f = open("distance_5.txt", "r")
for y in f:
    distance_5.append(eval(y))
f.close()

plt.loglog(x, distance_3, label="Distance 3")
plt.loglog(x, distance_5, label="Distance 5")
plt.legend()
plt.show()
