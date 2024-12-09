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

# Получаем данные из базы данных
results = session.query(BenchmarkResult).all()

# Преобразуем данные в DataFrame
data = {
    'system_name': [result.system_name for result in results],
    'test_datetime': [result.test_datetime for result in results],
    'program_type': [result.program_type for result in results],
    'num_threads': [result.num_threads for result in results],
    'test_word': [result.test_word for result in results],
    'total_execution_time': [result.total_execution_time for result in results],
    'additional_param': [result.additional_param for result in results],
}

df = pd.DataFrame(data)

# Фильтрация данных
df = df[~((df['program_type'] == 'MPI') & (df['additional_param'] == 'old'))]
df = df[~(df['system_name'] == 'DESKTOP-4EV88DG')]
df = df[~(df['test_word'].isin(['passw', '9999']))]

# Группируем данные по system_name, test_word, num_threads и находим среднее время выполнения
grouped = df.groupby(['system_name', 'test_word', 'num_threads', 'program_type', 'additional_param'])[
    'total_execution_time'].mean().reset_index()

# Строим график зависимости времени выполнения от числа потоков для каждой комбинации system_name и test_word
for (system_name, test_word), group in grouped.groupby(['system_name', 'test_word']):
    plt.figure(figsize=(10, 6))

    for (program_type, additional_param), version_group in group.groupby(['program_type', 'additional_param']):
        plt.plot(version_group['num_threads'], version_group['total_execution_time'], marker='o',
                 label=f"{program_type} ({additional_param})")

    plt.xlabel('Количество потоков')
    plt.ylabel('Время выполнения (секунды)')
    plt.yscale('log')
    plt.title(f'Зависимость времени выполнения от числа потоков\nSystem: {system_name}, Test Word: {test_word}')
    plt.legend(title="Версия программы")
    plt.grid(True)
    plt.tight_layout()

    # Сохранение графика
    plt.savefig(f'images/bigcompare/log_execution_time_{system_name}_{test_word}.png')

    # Показ графика
    plt.show()

# Закрываем сессию
session.close()
