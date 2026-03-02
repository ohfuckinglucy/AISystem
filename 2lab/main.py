import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

data = pd.read_csv("name_gender_dataset.csv")

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

plt.scatter(data['Count'], data['Probability'])
plt.xscale('log')
plt.title('Зависимость вероятности от количества')
plt.xlabel('Количество (Count)')
plt.ylabel('Вероятность (Probability)')
plt.tight_layout()
plt.show()

sns.countplot(data=data, x='Gender')
plt.title('Количество записей по полу')
plt.xlabel('Пол')
plt.ylabel('Число записей')
plt.tight_layout()
plt.show()

sns.boxplot(data=data, x='Gender', y='Count')
plt.title('Распределение количества по полу')
plt.xlabel('Пол')
plt.ylabel('Количество (Count)')
plt.tight_layout()
plt.show()

mean_count = data.groupby('Gender')['Count'].mean()
mean_count.plot(kind='bar')
plt.title('Среднее значение количества по полу')
plt.xlabel('Пол')
plt.ylabel('Среднее количество')
plt.tight_layout()
plt.show()

mean_prob = data.groupby('Gender')['Probability'].mean()
mean_prob.plot(kind='bar')
plt.title('Среднее значение вероятности по полу')
plt.xlabel('Пол')
plt.ylabel('Средняя вероятность')
plt.tight_layout()
plt.show()

corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Матрица корреляции')
plt.show()

top5_m = data[data['Gender'] == 'M'].nlargest(5, 'Count')
top5_m.plot.barh(x='Name', y='Count')
plt.title('Топ-5 мужских имён по количеству')
plt.xlabel('Количество')
plt.ylabel('Имя')
plt.tight_layout()
plt.show()

top5_f = data[data['Gender'] == 'F'].nlargest(5, 'Count')
top5_f.plot.barh(x='Name', y='Count')
plt.title('Топ-5 женских имён по количеству')
plt.xlabel('Количество')
plt.ylabel('Имя')
plt.tight_layout()
plt.show()

data['NameLength'] = data['Name'].apply(len)
sns.boxplot(data=data, x='Gender', y='NameLength')
plt.title('Длина имени по полу')
plt.xlabel('Пол')
plt.ylabel('Длина имени (символов)')
plt.tight_layout()
plt.show()

vowels = "aeiouAEIOU"
data['VowelCount'] = data['Name'].apply(lambda x: sum(1 for c in x if c in vowels))
data['VowelRatio'] = data['VowelCount'] / data['NameLength']
sns.boxplot(data=data, x='Gender', y='VowelRatio')
plt.title('Доля гласных букв в имени по полу')
plt.xlabel('Пол')
plt.ylabel('Доля гласных')
plt.tight_layout()
plt.show()

data['LastLetter'] = data['Name'].str[-1]
top_last = data.groupby(['LastLetter', 'Gender']).size().unstack()
top_last.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Распределение по последней букве имени и полу')
plt.xlabel('Последняя буква')
plt.ylabel('Число имён')
plt.legend(title='Пол')
plt.tight_layout()
plt.show()

corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Матрица корреляции')
plt.show()

le = LabelEncoder()
data['GenderEncoded'] = le.fit_transform(data['Gender'])
data['LogCount'] = np.log1p(data['Count'])
last_letter_freq = data['LastLetter'].value_counts(normalize=True)
data['LastLetterFreq'] = data['LastLetter'].map(last_letter_freq)

features = [
    'GenderEncoded',
    'NameLength',
    'VowelCount',
    'VowelRatio',
    'Count',
    'LogCount',
    'Probability',
    'LastLetterFreq'
]

corr = data[features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Расширенная матрица корреляции признаков")
plt.tight_layout()
plt.show()

target_corr = corr['GenderEncoded'].sort_values(ascending=False)
plt.figure(figsize=(6,6))
sns.barplot(x=target_corr.values, y=target_corr.index)
plt.title("Корреляция признаков с полом")
plt.xlabel("Коэффициент корреляции")
plt.ylabel("Признак")
plt.tight_layout()
plt.show() 