import numpy as np
from matplotlib import pyplot as plt

dt = np.dtype([
    ('diameter', 'f4'),
    ('mass', 'f4'),
    ('sweetness', 'f4'),
    ('firmness', 'f4'),
    ('fruit', 'U10')
])
data = np.genfromtxt("fruc.data", delimiter=",", dtype=dt, encoding='utf-8')

print(f'Data type: {type(data)}')
print(f'Data[0] type: {type(data[0])}')
print(f'Data[0][4] type: {type(data[0][4])}')
print(f'Data shape: {data.shape}')
print(data[:5])

plt.figure(1)
plt.plot(data['diameter'][data['fruit'] == 'apple'], data['mass'][data['fruit'] == 'apple'], 'ro', label='Apple')
plt.plot(data['diameter'][data['fruit'] == 'banana'], data['mass'][data['fruit'] == 'banana'], 'yo', label='Banana')
plt.plot(data['diameter'][data['fruit'] == 'grape'], data['mass'][data['fruit'] == 'grape'], 'bo', label='Grape')
plt.xlabel("Диаметр (см)")
plt.ylabel("Масса (граммы)")
plt.legend()
plt.grid(True)

plt.figure(2)
plt.plot(data['sweetness'][data['fruit'] == 'apple'], data['firmness'][data['fruit'] == 'apple'], 'ro', label='Apple')
plt.plot(data['sweetness'][data['fruit'] == 'banana'], data['firmness'][data['fruit'] == 'banana'], 'yo', label='Banana')
plt.plot(data['sweetness'][data['fruit'] == 'grape'], data['firmness'][data['fruit'] == 'grape'], 'bo', label='Grape')
plt.xlabel("Сладость 1–10")
plt.ylabel("Твёрдость 1–10")
plt.legend()
plt.grid(True)

plt.figure(3)
plt.plot(data['sweetness'][data['fruit'] == 'apple'], data['mass'][data['fruit'] == 'apple'], 'ro', label='Apple')
plt.plot(data['sweetness'][data['fruit'] == 'banana'], data['mass'][data['fruit'] == 'banana'], 'yo', label='Banana')
plt.plot(data['sweetness'][data['fruit'] == 'grape'], data['mass'][data['fruit'] == 'grape'], 'bo', label='Grape')
plt.xlabel("Сладость 1–10")
plt.ylabel("Масса (граммы)")
plt.legend()
plt.grid(True)

plt.figure(4)
plt.plot(data['diameter'][data['fruit'] == 'apple'], data['firmness'][data['fruit'] == 'apple'], 'ro', label='Apple')
plt.plot(data['diameter'][data['fruit'] == 'banana'], data['firmness'][data['fruit'] == 'banana'], 'yo', label='Banana')
plt.plot(data['diameter'][data['fruit'] == 'grape'], data['firmness'][data['fruit'] == 'grape'], 'bo', label='Grape')
plt.xlabel("Диаметр (см)")
plt.ylabel("Твёрдость 1–10")
plt.legend()
plt.grid(True)

plt.show()