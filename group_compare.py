import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from analyze import BenchmarkResult
import statistics

# Настройка подключения к базе данных
DATABASE_URL = 'sqlite:///benchmark_results.db'  # Замените на ваш URL базы данных
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Возможные значения для nt и базовых версий
nt_values = range(0, 13)  # nt от 0 до 12
base_versions = [
    ('MPI', '50k'),
    ('OpenMP', 'static,10k'),
    ('OpenMP', 'dynamic,200k'),
    ('MPI', '10k')
]

# Хранение позиций версий для всех комбинаций
version_positions = {}

# Цикл по nt и базовым версиям
for nt in nt_values:
    for base_version in base_versions:
        # Шаг 1: Получаем данные из базы данных
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

        # Шаг 2: Исключаем базовую версию из DataFrame
        df = df[~((df['program_type'] == 'MPI') & (df['additional_param'] == 'old'))]

        # Группируем данные по типу программы, слову и количеству потоков
        grouped = df.groupby(['system_name', 'test_word', 'num_threads'])


        # Функция для расчета относительного времени выполнения
        def calculate_relative_times(group):
            # Находим время выполнения основной версии
            if (group['program_type'] == base_version[0]).any() and (
                    group['additional_param'] == base_version[1]).any():
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
        relative_times_df = grouped.apply(calculate_relative_times).reset_index(drop=True)

        # Шаг 3: Рассчитываем средний процент для каждой версии
        average_relative_times = relative_times_df.groupby(['program_type', 'additional_param'])[
            'relative_time'].mean().reset_index()

        # Шаг 4: Сортируем версии по среднему относительному времени и присваиваем позиции
        average_relative_times_sorted = average_relative_times.sort_values(by='relative_time')
        average_relative_times_sorted['position'] = range(1, len(average_relative_times_sorted) + 1)

        # Сохранение позиций для каждой версии
        for _, row in average_relative_times_sorted.iterrows():
            version_key = (row['program_type'], row['additional_param'])
            if version_key not in version_positions:
                version_positions[version_key] = []
            version_positions[version_key].append(row['position'])

        # Шаг 5: Сохранение графика
        plt.figure(figsize=(10, 6))
        plt.barh(average_relative_times_sorted['additional_param'].astype(str) + ' (' + average_relative_times_sorted[
            'program_type'] + ')',
                 average_relative_times_sorted['relative_time'], color='skyblue')
        plt.xlabel('Среднее относительное время (%)')
        plt.title(f'Сравнение относительного времени выполнения версий (NP={nt}, относительно {base_version})')
        plt.grid(axis='x')
        plt.tight_layout()

        # Сохранение графика
        plt.savefig(f'bars/autobase_no_old/comparison_chart_np{nt}_base_{base_version[0]}_{base_version[1]}.png')

# Шаг 6: Статистика для каждой версии
statistics_df = []
for version, positions in version_positions.items():
    avg = sum(positions) / len(positions)
    max_pos = max(positions)
    min_pos = min(positions)
    mode_pos = statistics.mode(positions)

    statistics_df.append({
        'program_type': version[0],
        'additional_param': version[1],
        'average_position': avg,
        'max_position': max_pos,
        'min_position': min_pos,
        'mode_position': mode_pos
    })

# Преобразуем в DataFrame для удобного экспорта и анализа
statistics_df = pd.DataFrame(statistics_df)

# Красивый вывод
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Показ статистики
print(statistics_df)

# Сохранение в CSV
statistics_df.to_csv('version_statistics_no_old.csv', index=False)

# Закрываем сессию
session.close()
