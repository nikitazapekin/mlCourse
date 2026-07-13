# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загружаем датасет Iris из CSV файла
df = pd.read_csv('dataset/archive/Iris.csv')
df.drop(columns=['Id'], inplace=True)

feature_names = df.columns[:-1].tolist()
target_names = df['Species'].unique().tolist()

'''
print(feature_names)
['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
'''

'''
print(target_names )
['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
'''

X = df[feature_names].values
y = pd.Categorical(df['Species']).codes

'''
print(X)   -наши стебли 
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 [4.6 3.4 1.4 0.3]
 [5.  3.4 1.5 0.2]
 [4.4 2.9 1.4 0.2]
 [4.9 3.1 1.5 0.1]
 [5.4 3.7 1.5 0.2]
 [4.8 3.4 1.6 0.2]
 [4.8 3.  1.4 0.1]
 [4.3 3.  1.1 0.1]
'''

'''
pd.Categorical(df['Species']).codes преобразует строковые названия видов 
(Iris-setosa, Iris-versicolor, Iris-virginica) в числовые коды (0, 1, 2). 
Это нужно, потому что KNeighborsClassifier работает с числами, а не со строками.


print(y)
(.venv) nikita@nikita-HP-Laptop-17-cp2xxx:~/mll/models/supervised/classifications/KNearestNeighbours$ python3 realization.py 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''

print("Первые 5 строк данных:")

'''
Первые 5 строк данных:
   SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0            5.1           3.5            1.4           0.2  Iris-setosa
1            4.9           3.0            1.4           0.2  Iris-setosa
2            4.7           3.2            1.3           0.2  Iris-setosa
3            4.6           3.1            1.5           0.2  Iris-setosa
4            5.0           3.6            1.4           0.2  Iris-setosa
'''
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print(f"\nРазмер датасета: {df.shape}")
print(f"Классы: {target_names}")
print(f"Распределение классов:\n{df['Species'].value_counts()}")


'''
Первые 5 строк данных:
   SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0            5.1           3.5            1.4           0.2  Iris-setosa
1            4.9           3.0            1.4           0.2  Iris-setosa
2            4.7           3.2            1.3           0.2  Iris-setosa
3            4.6           3.1            1.5           0.2  Iris-setosa
4            5.0           3.6            1.4           0.2  Iris-setosa

Информация о данных:
<class 'pandas.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   SepalLengthCm  150 non-null    float64
 1   SepalWidthCm   150 non-null    float64
 2   PetalLengthCm  150 non-null    float64
 3   PetalWidthCm   150 non-null    float64
 4   Species        150 non-null    str    
dtypes: float64(4), str(1)
memory usage: 6.0 KB
None

Размер датасета: (150, 5)
Классы: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
Распределение классов:
Species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: count, dtype: int64
'''





'''
ШАГ 2 исследуем данные

'''


# Статистика по данным
print("\nСтатистическое описание:")
print(df.describe())

# Визуализация
plt.figure(figsize=(15, 5))

# 1. Распределение признаков по классам
plt.subplot(1, 3, 1)
for species in target_names:
    subset = df[df['Species'] == species]
    plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], 
                label=species, alpha=0.7)
plt.xlabel('Длина лепестка')
plt.ylabel('Ширина лепестка')
plt.title('Распределение по длине и ширине лепестка')
plt.legend()

# 2. Pairplot для визуализации всех признаков
plt.subplot(1, 3, 2)
# Используем pairplot из seaborn для всех признаков
sns.pairplot(df[['SepalLengthCm', 'SepalWidthCm', 
                 'PetalLengthCm', 'PetalWidthCm', 'Species']], 
             hue='Species')
plt.title('Pairplot всех признаков')

plt.tight_layout()
plt.savefig('visualization.png', dpi=150)
print("График сохранен в visualization.png")









# Разделение на признаки и целевую переменную
X = df[['SepalLengthCm', 'SepalWidthCm', 
         'PetalLengthCm', 'PetalWidthCm']]



'''
print('X', X)

 SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
0              5.1           3.5            1.4           0.2
1              4.9           3.0            1.4           0.2
2              4.7           3.2            1.3           0.2
3              4.6           3.1            1.5           0.2
4              5.0           3.6            1.4           0.2
..             ...           ...            ...           ...
145            6.7           3.0            5.2           2.3
146            6.3           2.5            5.0           1.9
147            6.5           3.0            5.2           2.0
148            6.2           3.4            5.4           2.3
149            5.9           3.0            5.1           1.8
'''

y = df['Species']

'''
print('Y', y)


Y 0         Iris-setosa
1         Iris-setosa
2         Iris-setosa
3         Iris-setosa
4         Iris-setosa
            ...      
145    Iris-virginica
146    Iris-virginica
147    Iris-virginica
148    Iris-virginica
149    Iris-virginica
'''



# Разделение на обучающую (70%) и тестовую (30%) выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение в обучающей:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Распределение в тестовой:\n{pd.Series(y_test).value_counts().sort_index()}")

# ВАЖНО: Масштабирование признаков для KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



'''
StandardScaler приводит все признаки к одному масштабу (среднее = 0, стандартное отклонение = 1).
Зачем нужно: KNN считает расстояния между точками. Если один признак в диапазоне 0-1000, а другой 0-1, то первый будет доминировать при вычислении расстояния, и KNN будет игнорировать второй признак. Масштабирование устраняет эту проблему.
- fit_transform(X_train) — вычисляет среднее и отклонение на train и применяет
- transform(X_test) — применяет те же значения (среднее/отклонение) к test, чтобы данные были в одном масштабе




PetalLengthCm имеет диапазон в ~3 раза шире. Без масштабирования:
1. KNN — расстояние между точками почти целиком определяется PetalLengthCm. SepalWidthCm practically игнорируется.
 Если ближайший сосед по PetalLengthCm — он и будет соседом, хотя SepalWidthCm может сказать обратное.
2. Градиентный спуск — функция ошибки вытянута в виде длинного узкого овала.
 Сходимость будет медленной, шаги будут «прыгать» вдоль длинной оси.
3. Регуляризация — штрафует большие веса. Без масштабирования веса для признаков с малым диапазоном будут 
искусственно большими, и регуляризация будет штрафовать их сильнее, чем нужно.
4. PCA — главные компоненты будут определяться признаками с большими значениями, а не теми, 
которые действительно несут больше информации.
В общем: если признаки в разных шкалах — модель будет считать, что «шумный» признак с большим диапазоном 
важнее, чем тихий признак с малым.
▣  Build · Big Pickle · 7.3s
Build·Big PickleOpenCode Zen



Пример: Если SepalLengthCm в train имеет mean=5.8 и std=0.8, то значение 5.1 превращается в (5.1 - 5.8) / 0.8 = -0.900 — это и есть первое число в первом ряду.
Зачем: Теперь все признаки в диапазоне примерно от -3 до +3. KNN будет считать расстояния корректно — ни один признак не будет доминировать из-за своей шкалы.



print('train', X_train_scaled)


train [[-0.90045861 -1.21813584 -0.44283471 -0.13515309]
 [ 0.38036614 -1.8819988   0.40257701  0.38088597]
 [-0.90045861  1.65860364 -1.28824644 -1.1672312 ]
 [ 1.07899781  0.33087773  1.19162795  1.41296408]
 [-0.20182693 -0.55427288  0.17713389  0.12286644]
 [ 0.9625592  -0.11169758  0.79710248  1.41296408]
 [-1.13333583  0.10959008 -1.28824644 -1.42525073]
 [-0.90045861  1.65860364 -1.23188566 -1.29624096]
 [-1.7155289  -0.33298523 -1.34460722 -1.29624096]
 [ 1.42831365 -0.11169758  1.19162795  1.15494455]
 [ 0.72968197  0.33087773  0.7407417   1.02593479]
'''


'''

print('test' , X_test_scaled )

test [[ 1.66119088 -0.33298523  1.41707108  0.76791526]
 [ 0.26392752 -0.33298523  0.51529857  0.25187621]
 [ 0.49680475 -0.55427288  0.7407417   0.38088597]
 [ 0.49680475  0.55216538  0.51529857  0.50989573]
 [ 0.26392752 -0.11169758  0.62802014  0.76791526]
 [ 0.9625592   0.10959008  1.02254561  1.54197385]
 [-0.43470415 -1.66071114  0.1207731   0.12286644]
 [-0.31826554 -1.21813584  0.06441232 -0.13515309]

'''

print("\nПример масштабированных данных (первые 5 строк):")
print(pd.DataFrame(X_train_scaled[:5], columns=feature_names).round(3))

'''

'''



# Создаем модель с начальным k=5
knn_basic = KNeighborsClassifier(n_neighbors=5)
knn_basic.fit(X_train_scaled, y_train)

# Предсказания на тестовой выборке
y_pred_basic = knn_basic.predict(X_test_scaled)

'''
print('pred', y_pred_basic) выводим предсказанные ответы

pred ['Iris-virginica' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
 'Iris-virginica' 'Iris-virginica' 'Iris-versicolor' 'Iris-versicolor'
 'Iris-setosa' 'Iris-virginica' 'Iris-setosa' 'Iris-setosa'
 'Iris-virginica' 'Iris-virginica' 'Iris-setosa' 'Iris-virginica'
 'Iris-versicolor' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
 'Iris-versicolor' 'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'
 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
 'Iris-versicolor' 'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'
 'Iris-versicolor' 'Iris-setosa' 'Iris-virginica' 'Iris-setosa'
 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-versicolor'
 'Iris-versicolor' 'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'
 'Iris-versicolor']
'''



# Оценка базовой модели
accuracy_basic = accuracy_score(y_test, y_pred_basic)
print(f"Точность базовой модели (k=5): {accuracy_basic:.4f}")
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred_basic, target_names=target_names))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred_basic)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (базовая модель)')
plt.savefig('confusion_matrix.png', dpi=150)
print("Матрица ошибок сохранена в confusion_matrix.png")





def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """
    Функция для предсказания вида ириса по новым данным
    """
    # Создаем массив с признаками
    new_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Масштабируем
    new_flower_scaled = scaler.transform(new_flower)
    
    # Предсказываем
    prediction = knn_basic.predict(new_flower_scaled)
    probability = knn_basic.predict_proba(new_flower_scaled)
    
    # Выводим результат
    species = prediction[0]
    probs = {target_names[i]: prob for i, prob in enumerate(probability[0])}
    
    print(f"=== Результат предсказания ===")
    print(f"Признаки: длина чашелистика={sepal_length}, ширина чашелистика={sepal_width}")
    print(f"         длина лепестка={petal_length}, ширина лепестка={petal_width}")
    print(f"\nПредсказанный вид: {species}")
    print(f"Вероятности:")
    for species_name, prob in probs.items():
        print(f"  {species_name}: {prob:.2%}")
    
    return prediction[0]

# Примеры предсказаний
print("\nПример 1: Цветок с типичными признаками setosa")
predict_iris(5.1, 3.5, 1.4, 0.2)

print("\nПример 2: Цветок с типичными признаками versicolor")
predict_iris(6.0, 2.8, 4.5, 1.3)

print("\nПример 3: Цветок с типичными признаками virginica")
predict_iris(7.0, 3.2, 6.0, 2.0)

print("\nПример 4: Неопределенный случай")
predict_iris(6.5, 3.0, 5.0, 1.8)


'''
=== Результат предсказания ===
Признаки: длина чашелистика=6.0, ширина чашелистика=2.8
         длина лепестка=4.5, ширина лепестка=1.3

Предсказанный вид: Iris-versicolor
Вероятности:
  Iris-setosa: 0.00%
  Iris-versicolor: 100.00%
  Iris-virginica: 0.00%

Пример 3: Цветок с типичными признаками virginica
/home/nikita/mll/backend/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
=== Результат предсказания ===
Признаки: длина чашелистика=7.0, ширина чашелистика=3.2
         длина лепестка=6.0, ширина лепестка=2.0

Предсказанный вид: Iris-virginica
Вероятности:
  Iris-setosa: 0.00%
  Iris-versicolor: 0.00%
  Iris-virginica: 100.00%

Пример 4: Неопределенный случай
/home/nikita/mll/backend/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
=== Результат предсказания ===
Признаки: длина чашелистика=6.5, ширина чашелистика=3.0
         длина лепестка=5.0, ширина лепестка=1.8

Предсказанный вид: Iris-virginica
Вероятности:
  Iris-setosa: 0.00%
  Iris-versicolor: 20.00%
  Iris-virginica: 80.00%
(.venv) nikita@nikita-HP-Laptop-17-cp2xxx:~/mll/models/supervised/classifications/KNearestNeighbours$ 
'''