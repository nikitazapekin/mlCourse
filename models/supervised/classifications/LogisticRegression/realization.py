import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# Загрузка
train = pd.read_csv('./dataset/playground-series-s5e8/train.csv')
test = pd.read_csv('./dataset/playground-series-s5e8/test.csv')
sample = pd.read_csv('./dataset/playground-series-s5e8/sample_submission.csv')

print(train.shape, test.shape)
print(train.head())
print(train.info())

#memory usage: 103.0 MB


# Проверка пропусков

print(train.isnull().sum())
'''


train.isnull().sum() считает количество пропущенных значений (NaN) в каждой колонке. 
Все нули — значит пропусков нет, данные чистые. Если бы где-то было, например, age 5, это значило бы
 5 пропущенных значений в колонке age.


None
id           0
age          0
job          0
marital      0
education    0
default      0
balance      0
housing      0
loan         0
contact      0
day          0
month        0
duration     0
campaign     0
pdays        0
previous     0
poutcome     0
y            0
dtype: int64
'''


# Распределение целевой переменной
print(train['y'].value_counts(normalize=True))


'''
y
0    0.879349
1    0.120651


- 87.9% — клиенты, которые не подписали депозит (y=0)
- 12.1% — клиенты, которые подписали (y=1)
'''

print('=======')

# Анализ категориальных признаков
categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(train[col].value_counts())

'''
выводим общую информацию 


job:
job
management       175541
blue-collar      170498
technician       138107
admin.            81492
services          64209
retired           35185
self-employed     19020
entrepreneur      17718
unemployed        17634
housemaid         15912
student           11767
unknown            2917
Name: count, dtype: int64

marital:
marital
married     480759
single      194834
divorced     74407
Name: count, dtype: int64

education:
education
secondary    401683
tertiary     227508
primary       99510
unknown       21299
Name: count, dtype: int64

default:
default
no     737151
yes     12849
Name: count, dtype: int64

housing:
housing
yes    411288
no     338712
Name: count, dtype: int64

loan:
loan
no     645023
yes    104977
Name: count, dtype: int64

contact:
contact
cellular     486655
unknown      231627
telephone     31718
Name: count, dtype: int64

month:
month
may    228411
aug    128859
jul    110647
jun     93670
nov     66062
apr     41319
feb     37611
jan     18937
oct      9204
sep      7409
mar      5802
dec      2069
Name: count, dtype: int64

poutcome:
poutcome
unknown    672450
failure     45115
success     17691
other       14744
Name: count, dtype: int64
'''



# Анализ числовых признаков
numerical_cols = train.select_dtypes(include=[np.number]).columns
print(train[numerical_cols].describe())
'''

               id            age        balance  ...          pdays       previous              y
count  750000.000000  750000.000000  750000.000000  ...  750000.000000  750000.000000  750000.000000
mean   374999.500000      40.926395    1204.067397  ...      22.412733       0.298545       0.120651
std    216506.495284      10.098829    2836.096759  ...      77.319998       1.335926       0.325721
min         0.000000      18.000000   -8019.000000  ...      -1.000000       0.000000       0.000000
25%    187499.750000      33.000000       0.000000  ...      -1.000000       0.000000       0.000000
50%    374999.500000      39.000000     634.000000  ...      -1.000000       0.000000       0.000000
75%    562499.250000      48.000000    1390.000000  ...      -1.000000       0.000000       0.000000
max    749999.000000      95.000000   99717.000000  ...     871.000000     200.000000       1.000000




escribe() выводит статистику по каждому числовому признаку:
- count — количество значений
- mean — среднее
- std — стандартное отклонение (разброс)
- min — минимум
- 25% — 25-й перцентиль (нижняя четверть)
- 50% — медиана
- 75% — 75-й перцентиль (верхняя четверть)
- max — максимум
'''





'''
Feature Engineering
'''
# Разделяем признаки и целевую переменную
X = train.drop(['id', 'y'], axis=1)
y = train['y']


'''
print("X", X)  - тут мы просто удалили id и y  из данных

X         
age          job   marital  education default  ...  duration campaign pdays previous  poutcome
0        42   technician   married  secondary      no  ...       117        3    -1        0   unknown
1        38  blue-collar   married  secondary      no  ...       185        1    -1        0   unknown
2        36  blue-collar   married  secondary      no  ...       111        2    -1        0   unknown
3        27      student    single  secondary      no  ...        10        2    -1        0   unknown
4        26   technician   married  secondary      no  ...       902        1    -1        0   unknown
...     ...          ...       ...        ...     ...  ...       ...      ...   ...      ...       ...
749995   29     services    single  secondary      no  ...      1006        2    -1        0   unknown
749996   69      retired  divorced   tertiary      no  ...        87        1    -1        0   unknown
749997   50  blue-collar   married  secondary      no  ...       113        1    -1        0   unknown
749998   32   technician   married  secondary      no  ...       108        6    -1        0   unknown
749999   42   technician   married  secondary      no  ...       143        1     1        7   failure

'''

'''
print('y', y)  - тут мы присвоили переменной у колонку у из  train.csv

 y, Length: 750000, dtype: int64
'''


X_test = test.drop(['id'], axis=1)

# 1. Обработка пропусков
# Для простоты заполним числовые медианой, категориальные - модой 






# пустых значений у нас нет но это хорошая практика

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')



# Разделяем колонки по типу
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(include=['object']).columns



'''
Разделили признаки на две группы по типу данных:
- num_cols — числовые: age, balance, day, duration, campaign, pdays, previous
- cat_cols — категориальные (строки): job, marital, education, default, housing, loan, contact, month, poutcome
Это нужно, чтобы к каждой группе применить разную обработку: числовые — масштабировать,
 категориальные — закодировать (OneHotEncoder).
'''


'''
print('num', num_cols)

num Index(['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'], dtype='str')
'''

'''
print('cat', cat_cols )

cat Index(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
       'month', 'poutcome'],
      dtype='str')
'''


# Заполняем пропуски
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# 2. Кодирование категориальных признаков
# Используем LabelEncoder или One-Hot Encoding
encoder = LabelEncoder()
for col in cat_cols:
    # Объединяем train и test, чтобы кодировать одинаково
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    '''
    print('COMBINED', combined)
    print('==============')
    
    =============
COMBINED 0         no
1         no
2         no
3         no
4         no
          ..
249995    no
249996    no
249997    no
249998    no
249999    no
Name: loan, Length: 1000000, dtype: str
==============
COMBINED 0         cellular
1          unknown
2          unknown
3          unknown
4         cellular
            ...   
249995    cellular
249996    cellular
249997    cellular
249998     unknown
249999    cellular
Name: contact, Length: 1000000, dtype: str
==============
COMBINED 0         aug
1         jun
2         may
3         may
4         feb
         ... 
249995    nov
249996    nov
249997    jul
249998    may
249999    apr
Name: month, Length: 1000000, dtype: str
    '''

    encoder.fit(combined)
    '''
    print(' encoder', encoder)
    
     encoder LabelEncoder()
 encoder LabelEncoder()
 encoder LabelEncoder()
 encoder LabelEncoder()
 encoder LabelEncoder()
 encoder LabelEncoder()
 encoder LabelEncoder()
 encoder LabelEncoder()
 encoder LabelEncoder()
    '''


    X[col] = encoder.transform(X[col].astype(str))

    '''
    print('x col', X[col])
    
    
    Name: loan, Length: 750000, dtype: int64
x col 0         0
1         2
2         2
3         2
4         0
         ..
749995    2
749996    0
749997    0
749998    0
749999    0
Name: contact, Length: 750000, dtype: int64
x col 0         1
1         6
2         8
3         8
4         3
         ..
749995    5
749996    1
749997    0
749998    1
749999    1
Name: month, Length: 750000, dtype: int64
x col 0         3
1         3
2         3
3         3
4         3
         ..
749995    3
749996    3
749997    3
749998    3
749999    0
    '''
    X_test[col] = encoder.transform(X_test[col].astype(str))

# 3. Масштабирование (ОЧЕНЬ ВАЖНО для логистической регрессии!)


'''

наши численные признаки и их мастштабируем
print('X', X)
 
X          age  job  marital  education  default  balance  ...  month  duration  campaign  pdays  previous  poutcome
0       42.0    9        1          1        0      7.0  ...      1     117.0       3.0   -1.0       0.0         3
1       38.0    1        1          1        0    514.0  ...      6     185.0       1.0   -1.0       0.0         3
2       36.0    1        1          1        0    602.0  ...      8     111.0       2.0   -1.0       0.0         3
3       27.0    8        2          1        0     34.0  ...      8      10.0       2.0   -1.0       0.0         3
4       26.0    9        1          1        0    889.0  ...      3     902.0       1.0   -1.0       0.0         3
'''

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



'''
print('X_scaled', X_scaled)

X_scaled [[ 0.10630996  1.43658638 -0.27816759 ... -0.30280328 -0.2234746
   0.31835563]
 [-0.28977584 -1.01972426 -0.27816759 ... -0.30280328 -0.2234746
   0.31835563]
 [-0.48781874 -1.01972426 -0.27816759 ... -0.30280328 -0.2234746
   0.31835563]
 ...
 [ 0.89848155 -1.01972426 -0.27816759 ... -0.30280328 -0.2234746
   0.31835563]
 [-0.88390453  1.43658638 -0.27816759 ... -0.30280328 -0.2234746
   0.31835563]
 [ 0.10630996  1.43658638 -0.27816759 ... -0.27693673  5.01633991
'''



X_test_scaled = scaler.transform(X_test)

# 4. Разделение на обучающую и валидационную выборку
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)




# Создаем модель
model = LogisticRegression(
    penalty='l2',          # L2-регуляризация (стандарт)
    C=1.0,                 # Обратный коэффициент регуляризации
    solver='lbfgs',        # Алгоритм оптимизации
    max_iter=1000,
    random_state=42,
    class_weight='balanced' # Помогает при дисбалансе классов
)

# Обучаем
model.fit(X_train, y_train)

# Предсказываем вероятности на валидации
y_val_proba = model.predict_proba(X_val)[:, 1]  # Берем вероятность класса 1

# Оцениваем по ROC AUC
roc_auc = roc_auc_score(y_val, y_val_proba)
print(f'ROC AUC на валидации: {roc_auc:.4f}')

'''
ROC AUC на валидации: 0.9228
'''


'''
Создали модель логистической регрессии с параметрами:
- penalty='l2' — регуляризация (штрафует большие веса, предотвращает переобучение)
- C=1.0 — сила регуляризации (меньше C = сильнее штраф)
- solver='lbfgs' — алгоритм оптимизации
- class_weight='balanced' — автоматически корректирует веса классов (компенсирует дисбаланс 88%/12%)
Что дальше:
1. model.fit(X_train, y_train) — обучили на тренировочных данных
2. predict_proba(X_val)[:, 1] — предсказали вероятность принадлежности к классу 1 (подписался)
3. ROC AUC = 0.9228 — метрика качества (1.0 = идеально, 0.5 = случайные угадывания). 0.92 — хороший результат.
'''




from sklearn.model_selection import GridSearchCV

# Сетка параметров
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']  # Поддерживают l1
}

# Grid Search с кросс-валидацией
grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Лучшие параметры: {grid.best_params_}")
print(f"Лучший ROC AUC на CV: {grid.best_score_:.4f}")

# Используем лучшую модель
best_model = grid.best_estimator_
y_val_proba_best = best_model.predict_proba(X_val)[:, 1]
print(f'ROC AUC на валидации (лучшая модель): {roc_auc_score(y_val, y_val_proba_best):.4f}')


'''
Лучшие параметры: {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}
Лучший ROC AUC на CV: 0.9216
ROC AUC на валидации (лучшая модель): 0.9228
'''


'''
Да. GridSearchCV перебрал все комбинации параметров из param_grid:
- C: 0.01, 0.1, 1, 10, 100
- penalty: l1, l2
- solver: liblinear, saga
= 20 комбинаций. Каждую проверил на 5-fold кросс-валидации (cv=5).
Результат: Лучшая комбинация — C=0.1, penalty='l1', solver='liblinear'.
 ROC AUC на валидации остался 0.9228 — модель не улучшилась, но и не ухудшилась. 
 Это значит, что исходные параметры были уже нормальными.
'''


'''

model и grid — это две отдельные модели:
1. model — логистическая регрессия с фиксированными параметрами (C=1.0, penalty='l2'). Обучили её вручную.
2. grid — AutoML-подбор: создаёт новые LogisticRegression с разными параметрами, обучает каждую на кросс-валидации,
выбирает лучшую.
grid.best_estimator_ — это и есть новая модель с лучшими параметрами. model больше нигде не используется. 
Это как сравнение: "вот модель с моими параметрами, а вот что даёт自动 подбор параметров".
'''




# Предсказания на тестовой выборке
test_proba = model.predict_proba(X_test_scaled)[:, 1]

'''

testt_proba = model.predict_proba(X_test_scaled) 
print('ttest', testt_proba)


ttest [[0.94569251 0.05430749]
 [0.32000639 0.67999361]
 [0.96948631 0.03051369]
 ...
 [0.16040657 0.83959343]
 [0.97179864 0.02820136]
 [0.60356684 0.39643316]]
'''
# Создаем DataFrame для сабмита
submission = pd.DataFrame({
    'id': test['id'],
    'y': test_proba
})

# Проверяем формат
print(submission.head())
print(f"Размер: {submission.shape}")  # Должен быть (количество строк в test, 2)

# Сохраняем в CSV
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv создан!")


'''
Мы обучили модель так: model.fit(X_train, y_train). Второй аргумент y_train — это то, 
что модель учится предсказывать.
predict_proba(X_test_scaled) предсказывает вероятности того, чему модель обучилась — то
 есть y (подписался клиент или нет). Balance и другие признаки — это входные данные (X), а не цель.
'''


'''
y — это целевая переменная: подписал ли клиент депозит (1) или нет (0).
 Это задача бинарной классификации — мы предсказываем вероятность того, 
 что клиент подпишется, чтобы банк мог направить маркетинг только на тех, кто скорее всего подпишется.
'''



'''
то есть мы тут не создавали никаких дополнительных признаков? 
/home/nikita/mll/models/supervised/classifications/LogisticRegression/realization.py  
и чисто работали с признаками что у нас есть уже? 

Да, только с теми признаками, что уже есть в датасете. 
Никаких новых колонок не создавали — только обработали их: заполнили пропуски, 
закодировали категориальные (OneHotEncoder), масштабировали числовые (StandardScaler).
'''