import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from analyze import BenchmarkResult
import platform


def convert_from_base(base62_str, base, alphabet=""):
    if alphabet == "":
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if base > len(alphabet):
        raise ValueError("Base exceeds the size of the alphabet")

    number = 0
    for i, char in enumerate(reversed(base62_str)):
        value = alphabet.index(char)
        number += value * (base ** i)

    return number


# Функция для получения данных из базы и построения графика
def plot_brute_force_times(word, db_url='sqlite:///benchmark_results.db'):
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

    # Загружаем данные в DataFrame с помощью pandas
    df = pd.DataFrame(data)
    df = df[~((df['program_type'] != 'Pthreads') & (df['additional_param'] == 'old'))]

    if df.empty:
        print(f"No data found for word: {word}")
        return

    # Группировка данных по версиям программы
    versions = ['dynamic,0', 'static,0', 'auto,0',
                'dynamic,10k', 'static,10k', 'auto,10k',
                'dynamic,200k', 'static,200k', 'auto,200k',
                '10k', '50k', 'old']
    labels = ['OpenMP dynamic,auto', 'OpenMP static,auto', 'OpenMP auto,auto',
              "OpenMP dynamic,10k", "OpenMP static,10k", "OpenMP auto,10k",
              "OpenMP dynamic,200k", "OpenMP static,200k", "OpenMP auto,200k",
              'MPI 10k', 'MPI 50k', 'MPI old']

    colors = ["#e60049", "#0bb4ff", "#50e991",
              "#e6d800", "#9b19f5", "#ffa300",
              "#dc0ab4", "#b3d4ff", "#00bfa0",
              "#7c1158", "#4421af", "#00b7c7"]

    plt.style.use('dark_background')

    plt.figure(figsize=(10, 6))

    # Построение графиков для каждой версии программы
    for version, label, color in zip(versions, labels, colors):
        # Фильтрация данных по версии программы
        version_df = df[df['additional_param'] == version]

        if not version_df.empty:
            # Сортировка данных по количеству потоков для корректного отображения
            version_df = version_df.sort_values('num_threads')

            # Построение линии на графике с указанным цветом
            plt.plot(version_df['num_threads'], version_df['brute_force_time'], marker='o', color=color, label=label)

    # Настройки графика
    plt.title(f"Brute-force Time vs Number of Processes for '{word}' on {platform.node()}")
    plt.xlabel("Number of Processes (Threads)")
    plt.ylabel("Brute-force Time (seconds)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 1, 1 - это верхний правый угол
    plt.grid(True)

    # Показ графика
    plt.savefig(f"bars/mipt/md5_{word}_{convert_from_base(word, 62)}.png", bbox_inches='tight')
    # plt.show()


# Пример вызова функции
# tw_list = ["999", "aaaa", "anaB", "anaC", "AAAA", "test", "9999", "passw"]
# for tw in tw_list[7:]:
#     plot_brute_force_times(tw)
plot_brute_force_times("anaB")
