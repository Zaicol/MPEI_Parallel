import os
import platform
import subprocess
import datetime
import argparse
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from time import sleep

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
last_db_update = datetime.datetime.now()
OPENMP_V = {
    "d": "dynamic",
    "s": "static",
    "a": "auto",
    "g": "guided"
}


def number_shortener(number):
    if number < 1000:
        return str(number)
    elif number < 1000000:
        divided = number / 1000
        if int(divided) == divided:
            return f"{int(divided)}k"
        return f"{divided}k"
    else:
        divided = number / 1000000
        if int(divided) == divided:
            return f"{int(divided)}k"
        return f"{divided}k"


# Функция для запуска программы и сбора времени выполнения
def run_program(command):
    try:
        # Запуск программы
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(' '.join(command))

        # Получаем вывод программы
        output = result.stdout
        print(output)  # для отладки

        # Извлечение времени из вывода программы
        total_time = float(
            [line.split(": ")[1].split(" ")[0] for line in output.splitlines() if "Total execution time" in line][0])
        try:
            brute_force_time = float(
                [line.split(": ")[1].split(" ")[0] for line in output.splitlines() if "Brute-force time" in line][0])
        except Exception:
            brute_force_time = total_time

        return brute_force_time, total_time
    except subprocess.CalledProcessError as e:
        print(f"Error running program: {e}")
        return None, None


# Функция для запуска теста и записи результатов в БД
def run_benchmark(program_type, version, test_word, num_threads, chunk_size=0):
    global last_db_update
    system_name = platform.node()  # Получаем имя системы
    test_datetime = datetime.datetime.now()  # Текущая дата и время
    add_param = version

    # Если версия указана как "0", то она приравнивается к пустой строке

    if program_type == "m":
        program_type = "MPI"
        version_suffix = "" if version == "0" else f"_{version}"
        program_name = f"./md5_mpi{version_suffix}"
        command = ["mpirun", "-np", str(num_threads), program_name, test_word]
    elif program_type == "o":
        program_type = "OpenMP"
        program_name = f"./md5_openmp"
        cs_formatted = number_shortener(chunk_size)
        add_param = f"{OPENMP_V[version]},{cs_formatted}"
        # Задаем количество потоков для OpenMP
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["OMP_SCHEDULE"] = f"{OPENMP_V[version]}{(',' + str(chunk_size)) if chunk_size > 0 else ''}"
        command = [program_name, test_word]
    elif program_type == "p":
        program_type = "Pthreads"
        program_name = f"./md5_pthreads"
        add_param = f"{version},{number_shortener(chunk_size)}"
        command = [program_name, test_word, f"--nt={num_threads}", f"--ch={chunk_size}"]
    elif program_type == "t":
        program_type = "Thread"
        program_name = f"./md5_thread"
        add_param = f"{version},{number_shortener(chunk_size)}"
        command = [program_name, test_word, f"--nt={num_threads}", f"--ch={chunk_size}"]
    else:
        raise ValueError("Invalid program type. Use 'm' for MPI or 'o' for OpenMP.")
    print(f"Running {program_type} with word: {test_word}, "
          f"version: {version}, threads: {num_threads}, chunk: {chunk_size}")

    brute_force_time, total_time = run_program(command)

    if brute_force_time is not None and total_time is not None:
        # Создаем запись для базы данных

        result = BenchmarkResult(
            system_name=system_name,
            test_datetime=test_datetime,
            program_type=program_type,
            num_threads=num_threads,
            test_word=test_word,
            brute_force_time=brute_force_time,
            total_execution_time=total_time,
            additional_param=add_param
        )

        # Добавляем и сохраняем запись
        session.add(result)
        print(f"Test result saved for {program_type} {num_threads}t {add_param}")


# Настройка аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Run brute-force tests with MPI or OpenMP and log the results.")
    parser.add_argument("program_type", choices=['m', 'o'], help="Program type: 'm' for MPI, 'o' for OpenMP")
    parser.add_argument("version", help="Version of the program (_10k, _50k, etc.). Use 0 for no version suffix.")
    parser.add_argument("test_word", help="Test word for the brute-force attack")
    parser.add_argument("num_threads", type=int, help="Number of threads")

    return parser.parse_args()


# Запуск всех тестов
if __name__ == "__main__":
    # args = parse_args()
    # run_benchmark(args.program_type, args.version, args.test_word, args.num_threads)
    tw_list = ["999", "aaaa", "anaB", "anaC", "AAAA", "test", "9999"]  # , "passw"]
    start_n = 1
    end_n = 12
    for testword in tw_list[:6]:

        run_benchmark("p", "default", testword, 6, 10000)
        run_benchmark("p", "default", testword, 6, 50000)

        # for i in range(start_n, end_n + 1):
        #     run_benchmark("o", "a", testword, i, 50000)
        #     run_benchmark("o", "s", testword, i, 50000)
        #     run_benchmark("o", "d", testword, i, 50000)
        #     run_benchmark("o", "g", testword, i, 50000)

        if datetime.datetime.now() - last_db_update < datetime.timedelta(milliseconds=7000):
            sleep(0.7)

        last_db_update = datetime.datetime.now()
        session.commit()
        print(f"Test results were committed to the DB")
