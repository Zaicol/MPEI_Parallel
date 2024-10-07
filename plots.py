import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as sa
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
    # Создаем подключение к базе данных
    engine = sa.create_engine(db_url)

    # SQL запрос для выборки данных по заданному слову
    query = f"""
        SELECT program_type, num_threads, brute_force_time, additional_param 
        FROM benchmark_results 
        WHERE test_word = '{word}'
        AND (program_type = 'MPI' OR program_type = 'OpenMP')
        AND system_name = '{platform.node()}'
    """

    # Загружаем данные в DataFrame с помощью pandas
    df = pd.read_sql(query, engine)

    if df.empty:
        print(f"No data found for word: {word}")
        return

    # Заменим пустое значение в 'additional_param' на OpenMP для удобства
    df['additional_param'] = df['additional_param'].replace({'': 'dynamic,0', None: 'dynamic,0'})

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
    plt.savefig(f"images/bigcompare/md5_{word}_{convert_from_base(word, 62)}.png", bbox_inches='tight')
    # plt.show()


# Пример вызова функции
tw_list = ["999", "aaaa", "anaB", "anaC", "AAAA", "test", "9999", "passw"]
for tw in tw_list[7:]:
    plot_brute_force_times(tw)
