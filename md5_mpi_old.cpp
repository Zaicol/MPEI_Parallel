#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <openssl/md5.h>
#include <sstream>
#include <string>
#include <vector>

// Функция для преобразования числа в строку по указанному основанию
std::string convertToBase(int number, int base, const std::string &alphabet, int set_size) {
    if (base > alphabet.size()) {
        throw std::invalid_argument("Base exceeds the size of the alphabet");
    }

    std::string result;
    int quotient = number;

    do {
        int remainder = quotient % base;
        result = alphabet[remainder] + result;
        quotient /= base;
    } while (quotient > 0);

    if (result.size() < set_size) {
        result = std::string(set_size - result.size(), alphabet[0]) + result;
    }

    return result;
}

// Функция для вычисления MD5-хеша строки
std::string md5_hash(const std::string &input) {
    unsigned char hash[MD5_DIGEST_LENGTH];

    MD5_CTX md5;
    MD5_Init(&md5);
    MD5_Update(&md5, input.c_str(), input.size());
    MD5_Final(hash, &md5);

    std::stringstream ss;

    for (unsigned char i: hash) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(i);
    }
    return ss.str();
}

// Генерация комбинаций для брутфорса в указанном диапазоне
bool brute_force_md5_range(int start_index, int end_index, const std::string &characters, int password_length,
                           const std::string &target_hash, std::string &found_password) {
    for (int index = start_index; index < end_index; ++index) {
        // Генерация текущей комбинации
        std::string candidate = convertToBase(index, static_cast<int>(characters.size()), characters, password_length);

        // Проверка хеша
        if (md5_hash(candidate) == target_hash) {
            found_password = candidate;
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Проверяем, передан ли пароль в командной строке
    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <target_password>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Получаем целевой пароль из аргументов командной строки
    std::string target_password = argv[1];
    std::string target_md5_hash = md5_hash(target_password);

    // Символы, используемые для брутфорса
    const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const int max_length = static_cast<int>(target_password.size()); // Максимальная длина пароля

    // Замер времени начала выполнения всей программы
    double start_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Starting brute-force attack with " << size << " processes..." << std::endl;
    }

    std::string found_password;
    bool found_flag = false;

    // Барьер перед началом брутфорса, чтобы процессы синхронизировались
    MPI_Barrier(MPI_COMM_WORLD);
    double brute_force_start_time = MPI_Wtime();  // Время начала брутфорс-атаки

    for (int password_length = max_length; password_length <= max_length; ++password_length) {
        constexpr int subchunk_size = 50000;
        const int total_combinations = static_cast<int>(std::pow(characters.size(), password_length));
        const int chunk_size = total_combinations / size;
        if (rank == 0) {
            std::cout << "chunk size: " << chunk_size << "" << std::endl;
        }
        const int start_index = rank * chunk_size;
        const int end_index = (rank == size - 1) ? total_combinations : start_index + chunk_size;

        // Each process brute-forces its assigned range
        if (brute_force_md5_range(start_index, end_index, characters,
                                  password_length, target_md5_hash, found_password)
        ) {
            found_flag = true;
        }

        // Reduce to check if any process found the password
        int global_found_flag;
        MPI_Allreduce(&found_flag, &global_found_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // If any process found the password, stop the search
        if (global_found_flag > 0) {
            break;
        }
    }

    // Барьер после окончания брутфорс-атаки для синхронизации
    MPI_Barrier(MPI_COMM_WORLD);
    double brute_force_end_time = MPI_Wtime();  // Время окончания брутфорс-атаки

    // Определяем, какой процесс нашёл пароль
    int found_rank = found_flag ? rank : -1;
    MPI_Allreduce(MPI_IN_PLACE, &found_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Передаём найденный пароль (если он был найден)
    char password_buffer[256] = {0};
    if (found_flag) {
        strncpy(password_buffer, found_password.c_str(), sizeof(password_buffer));
    }
    MPI_Bcast(password_buffer, sizeof(password_buffer), MPI_CHAR, found_rank, MPI_COMM_WORLD);

    // Барьер перед выводом результатов, чтобы все процессы синхронизировались
    MPI_Barrier(MPI_COMM_WORLD);

    // Замер времени окончания выполнения всей программы
    double end_time = MPI_Wtime();

    // Вывод результата
    if (rank == 0) {
        if (strlen(password_buffer) > 0) {
            std::cout << "Password found: " << password_buffer << std::endl;
        } else {
            std::cout << "Password not found." << std::endl;
        }

        // Вывод времени выполнения
        std::cout << "Brute-force time: " << brute_force_end_time - brute_force_start_time << " seconds." << std::endl;
        std::cout << "Total execution time: " << end_time - start_time << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
