import numpy as np
import seaborn as sb
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sb.set_palette("husl")

data_src = "name_gender_dataset.csv"
data = pd.read_csv(data_src, delimiter=',', header=None, skiprows=1, 
                   dtype={0: str, 1: str, 2: int, 3: float}, 
                   names=['Name', 'Gender', 'Count', 'Probability'])

# data = data[:10000].copy()

le = LabelEncoder()
data['GenderEncoded'] = le.fit_transform(data['Gender'])
data['NameLength'] = data['Name'].astype(str).apply(len)

vowels = "aeiouAEIOU"
data['VowelCount'] = data['Name'].astype(str).apply(lambda x: sum(1 for c in x if c in vowels))
data['VowelRatio'] = data['VowelCount'] / data['NameLength']
data['LogCount'] = np.log1p(data['Count'])
data['LastLetterPos'] = data['Name'].str.lower().apply(lambda x: ord(x[-1]) - ord('a') + 1)

# sb.pairplot(data, hue='GenderEncoded', palette='husl')

X = data[[
    'NameLength', 
    'VowelCount',
    'VowelRatio', 
    'LogCount', 
    'Probability',
    'LastLetterPos'
]]

Y = data['GenderEncoded'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=49)
knn.fit(X_train, Y_train)

predictions = knn.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f"Точность модели: {accuracy:.2%}")

sample = data.sample(5)
for idx, row in sample.iterrows():
    features = row[['NameLength', 'VowelCount', 'VowelRatio', 'LogCount', 'Probability', 'LastLetterPos']].values.reshape(1, -1)
    pred = knn.predict(features)[0]
    print(f"{row['Name']:12} Реальный: {row['Gender']} Предсказание: {le.inverse_transform([pred])[0]}")

k_list = list(range(1, 50, 2)) 

cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1-x for x in cv_scores]

plt.figure()
plt.plot(k_list, MSE)
plt.xlabel("Кол-во соседей (К)")
plt.ylabel("Ошибка классификации (MSE)")

k_min = min(MSE)

all_k_min = []
for i in range(len(MSE)):
    if MSE[i] <= k_min:
        all_k_min.append(k_list[i])

print(f'Оптимальные значения К: {all_k_min}')

plt.show()