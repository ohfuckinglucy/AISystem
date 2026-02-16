import numpy as np
from matplotlib import pyplot as plt

dt = np.dtype([
    ('name', 'U20'),
    ('gender', 'U1'),
    ('count', 'i4'),
    ('probability', 'f8')
])

data = np.genfromtxt("name_gender_dataset.csv", delimiter=",", dtype=dt, skip_header=1, encoding='utf-8')

print(f'Тип данных: {type(data)}')
print(f'Тип записи: {type(data[0])}')
print(f'Тип имени: {type(data[0][0])}')
print(f'Форма данных: {data.shape}')
print(f'\nПервые 5 записей:')
print(data[:5])

male_mask = data['gender'] == 'M'
female_mask = data['gender'] == 'F'

male_data = data[male_mask]
female_data = data[female_mask]

print(f'\nМужских имён: {len(male_data)}')
print(f'Женских имён: {len(female_data)}')

sorted_by_count = np.sort(data, order='count')[::-1]

plt.figure(1)
top_15 = sorted_by_count[:15]
colors = ['#1f77b4' if g == 'M' else '#ff7f0e' for g in top_15['gender']]
plt.barh(range(15), top_15['count'], color=colors)
plt.yticks(range(15), top_15['name'])
plt.xlabel('Количество носителей')
plt.title('Топ-15 самых популярных имён')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.figure(2)
male_total = male_data['count'].sum()
female_total = female_data['count'].sum()
total = male_total + female_total
plt.pie([male_total, female_total], labels=[f'Мужские ({male_total:,})', f'Женские ({female_total:,})'], colors=['#1f77b4', '#ff7f0e'])
plt.title('Доля имён по полу (общее количество носителей)')
plt.axis('equal')

plt.figure(3)
plt.scatter(male_data['count'], male_data['probability'], c='#1f77b4', label='Мужские')
plt.scatter(female_data['count'], female_data['probability'], c='#ff7f0e', label='Женские')
plt.xlabel('Количество носителей (Count)')
plt.ylabel('Вероятность (Probability)')
plt.title('Зависимость Probability от Count')
plt.legend()
plt.grid()

plt.figure(4)

male_sorted = np.sort(male_data, order='count')[::-1][:10]
plt.subplot(1, 2, 1)
plt.barh(range(10), male_sorted['count'], color='#1f77b4')
plt.yticks(range(10), male_sorted['name'])
plt.xlabel('Количество')
plt.title('Топ-10 мужских имён')
plt.gca().invert_yaxis()
plt.grid()

female_sorted = np.sort(female_data, order='count')[::-1][:10]
plt.subplot(1, 2, 2)
plt.barh(range(10), female_sorted['count'], color='#ff7f0e')
plt.yticks(range(10), female_sorted['name'])
plt.xlabel('Количество')
plt.title('Топ-10 женских имён')
plt.gca().invert_yaxis()
plt.grid()

plt.tight_layout()
plt.show()