import os
import platform
import subprocess
import datetime
import argparse
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base

# Настройка базы данных и SQLAlchemy
Base = declarative_base()

# Модель для таблицы результатов
class BenchmarkResult(Base):
    __tablename__ = 'benchmark_results'

    id = sa.Column(sa.Integer, primary_key=True)
    system_name = sa.Column(sa.String)
    test_datetime = sa.Column(sa.DateTime)
    program_type = sa.Column(sa.String)  # MPI или OpenMP
    num_threads = sa.Column(sa.Integer)
    test_word = sa.Column(sa.String)
    brute_force_time = sa.Column(sa.Float)
    total_execution_time = sa.Column(sa.Float)
    additional_param = sa.Column(sa.String)  # версия программы

# Создание базы данных (если не существует)
engine = sa.create_engine('sqlite:///benchmark_results.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Функция для запуска программы и сбора времени выполнения
def run_program(command):
    try:
        # Запуск программы
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Получаем вывод программы
        output = result.stdout
        print(output)  # для отладки

        # Извлечение времени из вывода программы
        brute_force_time = float([line.split(": ")[1].split(" ")[0] for line in output.splitlines() if "Brute-force time" in line][0])
        total_time = float([line.split(": ")[1].split(" ")[0] for line in output.splitlines() if "Total execution time" in line][0])

        return brute_force_time, total_time
    except subprocess.CalledProcessError as e:
        print(f"Error running program: {e}")
        return None, None

# Функция для запуска теста и записи результатов в БД
def run_benchmark(program_type, version, test_word, num_threads):
    system_name = platform.node()  # Получаем имя системы (например, DESKTOP-12345678)
    test_datetime = datetime.datetime.now()  # Текущая дата и время

    # Если версия указана как "0", то она приравнивается к пустой строке
    version_suffix = "" if version == "0" else f"_{version}"

    if program_type == "m":
        program_name = f"../MPI/brute_force_md5{version_suffix}"
        command = ["mpirun", "-np", str(num_threads), program_name, test_word]
    elif program_type == "o":
        program_name = f"./md5_bf_openmp{version_suffix}"
        # Задаем количество потоков для OpenMP
        # subprocess.run(command, text=True, check=True)
        command = ["bash", "-c", f"OMP_NUM_THREADS={num_threads} {program_name} {test_word}"]
    else:
        raise ValueError("Invalid program type. Use 'm' for MPI or 'o' for OpenMP.")

    print(f"Running {program_type.upper()} test with command: {' '.join(command)}")

    brute_force_time, total_time = run_program(command)

    if brute_force_time is not None and total_time is not None:
        # Создаем запись для базы данных
        result = BenchmarkResult(
            system_name=system_name,
            test_datetime=test_datetime,
            program_type="MPI" if program_type == "m" else "OpenMP",
            num_threads=num_threads,
            test_word=test_word,
            brute_force_time=brute_force_time,
            total_execution_time=total_time,
            additional_param=version_suffix.replace('_', '')
        )

        # Добавляем и сохраняем запись
        session.add(result)
        session.commit()
        print(f"Test result saved for {program_type.upper()}")

# Настройка аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Run brute-force tests with MPI or OpenMP and log the results.")
    parser.add_argument("program_type", choices=['m', 'o'], help="Program type: 'm' for MPI, 'o' for OpenMP")
    parser.add_argument("version", help="Version of the program (_10k, _50k, etc.). Use 0 for no version suffix.")
    parser.add_argument("test_word", help="Test word for the brute-force attack")
    parser.add_argument("num_threads", type=int, help="Number of threads (for both MPI and OpenMP)")

    return parser.parse_args()

# Запуск всех тестов
if __name__ == "__main__":
    # args = parse_args()
    # run_benchmark(args.program_type, args.version, args.test_word, args.num_threads)
    tw_list = ["999", "aaaa", "anaB", "anaC", "AAAA", "test", "9999", "passw"]
    for tw in [tw_list[7]]:
        for i in range(3, 5):
            run_benchmark('o', '0', tw, i + 1)
            if i < 6:
                run_benchmark('m', '10k', tw, i + 1)
                run_benchmark('m', '50k', tw, i + 1)
                # run_benchmark('m', 'old', tw, i + 1)
