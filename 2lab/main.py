import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

data = pd.read_csv("name_gender_dataset.csv")

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

plt.scatter(data['Count'], data['Probability'])
plt.xscale('log')
plt.title('Зависимость Probability от Count')
plt.xlabel('Count')
plt.ylabel('Probability')
plt.tight_layout()
plt.show()

sns.countplot(data=data, x='Gender')
plt.title('Количество записей по полу')
plt.tight_layout()
plt.show()

sns.boxplot(data=data, x='Gender', y='Count')
plt.title('Boxplot Count по полу')
plt.tight_layout()
plt.show()

mean_count = data.groupby('Gender')['Count'].mean()

mean_count.plot(kind='bar')
plt.title('Среднее значение Count по полу')
plt.ylabel('Среднее значение')
plt.tight_layout()
plt.show()

mean_prob = data.groupby('Gender')['Probability'].mean()

mean_prob.plot(kind='bar')
plt.title('Среднее значение Probability по полу')
plt.ylabel('Среднее значение')
plt.tight_layout()
plt.show()

corr = data.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Матрица корреляции')
plt.show()

top5_m = data[data['Gender'] == 'M'].nlargest(5, 'Count')
top5_f = data[data['Gender'] == 'F'].nlargest(5, 'Count')

top5_m.plot.barh(x='Name', y='Count')
plt.title('Топ-5 мужских имён')
plt.tight_layout()
plt.show()

top5_f.plot.barh(x='Name', y='Count')
plt.title('Топ-5 женских имён')
plt.tight_layout()
plt.show()