import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as sa


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
        AND (additional_param IN ('10k', '50k', 'old', '') OR additional_param IS NULL)
    """

    # Загружаем данные в DataFrame с помощью pandas
    df = pd.read_sql(query, engine)

    if df.empty:
        print(f"No data found for word: {word}")
        return

    # Заменим пустое значение в 'additional_param' на OpenMP для удобства
    df['additional_param'] = df['additional_param'].replace({'': 'OpenMP', None: 'OpenMP'})

    # Группировка данных по версиям программы
    versions = ['OpenMP', '10k', '50k', 'old']
    labels = ['OpenMP', 'MPI 10k', 'MPI 50k', 'MPI old']

    colors = ['yellow', 'lightcyan', 'skyblue', 'fuchsia']

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
    plt.title(f"Brute-force Time vs Number of Processes for '{word}'")
    plt.xlabel("Number of Processes (Threads)")
    plt.ylabel("Brute-force Time (seconds)")
    plt.legend()
    plt.grid(True)

    # Показ графика
    plt.savefig(f"images/md5_{word}_wold.png")
    plt.show()


# Пример вызова функции
tw_list = ["999", "aaaa", "anaB", "anaC", "AAAA", "test", "9999"]
for tw in tw_list:
    plot_brute_force_times(tw)
