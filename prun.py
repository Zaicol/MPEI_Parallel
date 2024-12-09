import argparse
import os
import platform
import datetime
import subprocess


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



def run_benchmark(program_type, version, test_word, num_threads, chunk_size=0):
    system_name = platform.node()  # Получаем имя системы
    test_datetime = datetime.datetime.now()  # Текущая дата и время
    add_param = version

    if program_type == "m":
        program_type = "MPI"
        version_suffix = "_50k" if version == "0" else f"_{version}"
        program_name = f"./code/mpi/md5_mpi{version_suffix}"
        command = ["mpirun", "-np", str(num_threads), program_name, test_word]
    elif program_type == "o":
        program_type = "OpenMP"
        program_name = f"./code/openmp/md5_openmp"
        cs_formatted = number_shortener(chunk_size)
        add_param = f"{version},{cs_formatted}"
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["OMP_SCHEDULE"] = f"{version}{(',' + str(chunk_size)) if chunk_size > 0 else ''}"
        command = [program_name, test_word]
    elif program_type == "p":
        program_type = "Pthreads"
        program_name = f"./code/pthreads/md5_pthreads"
        add_param = f"{version},{number_shortener(chunk_size)}"
        command = [program_name, test_word, f"--nt={num_threads}", f"--ch={chunk_size}"]
    elif program_type == "t":
        program_type = "Thread"
        program_name = f"./code/thread/md5_thread"
        add_param = f"{version},{number_shortener(chunk_size)}"
        command = [program_name, test_word, f"--nt={num_threads}", f"--ch={chunk_size}"]
    elif program_type == "a":
        program_type = "AVX/SSE"
        version_suffix = "" if version == "" else f"_{version}"
        program_name = f"./code/avxsse/md5_avxsse{version_suffix}"
        add_param = f"{version}"
        command = [program_name, test_word]
        num_threads = 8
        chunk_size = 0
    else:
        raise ValueError("Invalid program type. Use 'm', 'o', 'p', 't', or 'a'.")

    print(f"Running {program_type} with word: {test_word}, "
          f"version: {version}, threads: {num_threads}, chunk: {chunk_size}")

    brute_force_time, total_time = run_program(command)

    if brute_force_time is not None and total_time is not None:
        print(f"Test result: {program_type} {num_threads} threads, version: {version}, "
              f"brute_force_time: {brute_force_time}s, total_time: {total_time}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD5 benchmark.")
    parser.add_argument("program_type", choices=["m", "o", "p", "t", "a"],
                        help="Type of the program: m (MPI), o (OpenMP), p (Pthreads), t (Thread), a (AVX/SSE)")
    parser.add_argument("-w", default="test", help="Word to test MD5 hash")
    parser.add_argument("-v", default="", help="Program version")
    parser.add_argument("-nt", type=int, default=12, help="Number of threads")
    parser.add_argument("-ch", type=int, default=1, help="Chunk size for parallel processing")

    args = parser.parse_args()

    run_benchmark(args.program_type, args.v, args.w, args.nt, args.ch)