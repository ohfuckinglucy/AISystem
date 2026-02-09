import numpy as np
from matplotlib import pyplot as plt

dt = np.dtype("f8, f8, f8, f8, U30")
data = np.genfromtxt("iris.data", delimiter=",", dtype=dt)

print(f'Data type: {type(data)}')
print(f'Data[0] type: {type(data[0])}')
print(f'Data[0][4] type: {type(data[0][4])}')
print(f'Data shape: {data.shape}')
print(data[:10])

sepal_length = []
sepal_width = []
petal_length = []
petal_width = []

for dot in data:
    sepal_length.append(dot[0])
    sepal_width.append(dot[1])
    petal_length.append(dot[2])
    petal_width.append(dot[3])

plt.figure(1)
plt.plot(sepal_length[:50], sepal_width[:50], 'rp', label="Setosa")
plt.plot(sepal_length[50:100], sepal_width[50:100], 'g^', label="Versicolor")
plt.plot(sepal_length[100:150], sepal_width[100:150], 'bs', label="Verginica")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()

plt.figure(2)
plt.plot(sepal_length[:50], petal_length[:50], 'rp', label="Setosa")
plt.plot(sepal_length[50:100], petal_length[50:100], 'g^', label="Versicolor")
plt.plot(sepal_length[100:150], petal_length[100:150], 'bs', label="Verginica")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()

plt.figure(3)
plt.plot(sepal_length[:50], petal_width[:50], 'rp', label="Setosa")
plt.plot(sepal_length[50:100], petal_width[50:100], 'g^', label="Versicolor")
plt.plot(sepal_length[100:150], petal_width[100:150], 'bs', label="Verginica")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()

plt.figure(4)
plt.plot(petal_width[:50], petal_width[:50], 'rp', label="Setosa")
plt.plot(petal_width[50:100], petal_width[50:100], 'g^', label="Versicolor")
plt.plot(petal_width[100:150], petal_width[100:150], 'bs', label="Verginica")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()

plt.figure(5)
plt.plot(petal_length[:50], petal_width[:50], 'rp', label="Setosa")
plt.plot(petal_length[50:100], petal_width[50:100], 'g^', label="Versicolor")
plt.plot(petal_length[100:150], petal_width[100:150], 'bs', label="Verginica")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()

plt.figure(6)
plt.plot(petal_length[:50], petal_length[:50], 'rp', label="Setosa")
plt.plot(petal_length[50:100], petal_length[50:100], 'g^', label="Versicolor")
plt.plot(petal_length[100:150], petal_length[100:150], 'bs', label="Verginica")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()

plt.show()