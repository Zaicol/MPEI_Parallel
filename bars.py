import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from analyze import BenchmarkResult  # Импортируйте вашу модель

# Настройка подключения к базе данных
DATABASE_URL = 'sqlite:///benchmark_results.db'  # Замените на ваш URL базы данных
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
nt = 0

# Получаем данные из базы данных
results = session.query(BenchmarkResult)
if nt:
    results = results.filter(BenchmarkResult.num_threads == nt)

results = results.all()


# Преобразуем данные в DataFrame
data = {
    'system_name': [result.system_name for result in results],
    'test_datetime': [result.test_datetime for result in results],
    'program_type': [result.program_type for result in results],
    'num_threads': [result.num_threads for result in results],
    'test_word': [result.test_word for result in results],
    'brute_force_time': [result.brute_force_time for result in results],
    'total_execution_time': [result.total_execution_time for result in results],
    'additional_param': [result.additional_param for result in results],
}

df = pd.DataFrame(data)
df = df[~((df['program_type'] == 'MPI') & (df['additional_param'] == 'old'))]
df = df[~(df['system_name'] == 'DESKTOP-4EV88DG')]
df = df[~(df['test_word'] == 'passw')]
df = df[~(df['test_word'] == '9999')]
# Группируем данные по типу программы, слову и количеству потоков
grouped = df.groupby(['system_name', 'test_word', 'num_threads'])
base_version = ('OpenMP', 'dynamic,0')


# Функция для расчета относительного времени выполнения
def calculate_relative_times(group):
    # Находим минимальное время выполнения в группе
    min_time = group['total_execution_time'].max()

    # Рассчитываем относительное время для всех версий
    group['relative_time'] = (group['total_execution_time'] / min_time) * 100
    return group


# Функция для расчета относительного времени выполнения
def calculate_relative_times2(group):
    # Находим время выполнения основной верс
    if (group['program_type'] == base_version[0]).any() and (group['additional_param'] == base_version[1]).any():
        base_time = group.loc[
            (group['program_type'] == base_version[0]) &
            (group['additional_param'] == base_version[1]),
            'total_execution_time'
        ].values[0]

        # Рассчитываем относительное время для всех версий
        group['relative_time'] = (group['total_execution_time'] / base_time) * 100
    else:
        group['relative_time'] = None  # Если основной версии нет, устанавливаем NaN
    return group


# Применяем функцию к каждой группе
relative_times_df = grouped.apply(calculate_relative_times2).reset_index(drop=True)

# Рассчитываем средний процент для каждой версии
average_relative_times = relative_times_df.groupby(['program_type', 'additional_param'])[
    'relative_time'].mean().reset_index()

# Сохраняем в CSV файл
average_relative_times.to_csv('bars/average_relative_times.csv', index=False, sep="\t")

# Строим столбчатую диаграмму
average_relative_times_sorted = average_relative_times.sort_values(by='relative_time')

plt.figure(figsize=(10, 6))
plt.barh(average_relative_times_sorted[
    'program_type'] + ' (' + average_relative_times_sorted['additional_param'].astype(str) + ')',
         average_relative_times_sorted['relative_time'], color='skyblue')
plt.xlabel('Среднее относительное время (%)')
plt.title('Сравнение относительного времени выполнения версий (NP = {})'.format(nt))
plt.grid(axis='x')
plt.tight_layout()

# Сохранение графика
plt.savefig(f'bars/mipt/comparison_chart_nt{nt}.png')

# Показ графика
plt.show()

# Закрываем сессию
session.close()
