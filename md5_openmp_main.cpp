#include <iostream>
#include <string>
#include <openssl/md5.h> // Для MD5, нужно установить библиотеку OpenSSL
#include <omp.h>
#include <chrono> // Для измерения времени
#include <cmath>

// Функция для получения MD5-хэша
std::string md5_hash(const std::string& str) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(str.c_str()), str.size(), digest);

    char md5_string[33];
    for (int i = 0; i < 16; i++) {
        sprintf(&md5_string[i * 2], "%02x", digest[i]);
    }
    return std::string(md5_string);
}

// Функция для перевода числа в строку на основе указанного алфавита (система счисления)
std::string convertToBase(int number, int base, const std::string& alphabet, int set_size) {
    if (base > static_cast<int>(alphabet.size())) {
        throw std::invalid_argument("Base exceeds the size of the alphabet");
    }

    std::string result;
    int quotient = number;

    do {
        int remainder = quotient % base;
        result = alphabet[remainder] + result;
        quotient /= base;
    } while (quotient > 0);

    // Дополняем строку, если она короче ожидаемого размера
    if (result.size() < set_size) {
        result = std::string(set_size - result.size(), alphabet[0]) + result;
    }

    return result;
}

// Функция для брутфорса с параллелизацией OpenMP
void brute_force_md5(const std::string& target_md5, const std::string& characters, int password_length) {
    int found = 0;
    const int base = static_cast<int>(characters.size());
    const int total_combinations = static_cast<int>(pow(base, password_length));

#pragma omp parallel for schedule(dynamic) shared(found)
    for (int i = 0; i < total_combinations; ++i) {
        if (found) continue; // Останавливаем поток, если пароль найден

        std::string candidate = convertToBase(i, base, characters, password_length);

        // Проверяем текущий кандидат
        if (md5_hash(candidate) == target_md5) {
#pragma omp critical
            {
                if (!found) {
                    std::cout << "Password found: " << candidate << std::endl;
                    found = 1;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <target_password>" << std::endl;
        return 1;
    }

    // Получаем целевой пароль из аргументов командной строки
    const std::string target_password = argv[1];
    const std::string target_md5_hash = md5_hash(target_password);

    const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const int password_length = static_cast<int>(target_password.size()); // Известная длина пароля

    std::cout << "Target MD5 hash: " << target_md5_hash << std::endl;

    // Измерение общего времени выполнения
    auto total_start = std::chrono::high_resolution_clock::now();

    // Измерение времени брутфорса
    auto brute_force_start = std::chrono::high_resolution_clock::now();

    // Запускаем брутфорс для поиска пароля
    brute_force_md5(target_md5_hash, characters, password_length);

    auto brute_force_end = std::chrono::high_resolution_clock::now();

    // Вычисление времени брутфорса
    std::chrono::duration<double> brute_force_duration = brute_force_end - brute_force_start;
    std::cout << "Brute-force time: " << brute_force_duration.count() << " seconds." << std::endl;

    // Завершаем измерение общего времени
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    std::cout << "Total execution time: " << total_duration.count() << " seconds." << std::endl;

    return 0;
}
