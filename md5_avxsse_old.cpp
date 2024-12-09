#include <iostream>
#include <string>
#include <openssl/md5.h>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstring>
#include <atomic>

// Global
std::string target_md5_hash;
std::string characters;
int password_length;
std::atomic<bool> found(false);
int chunk_size = 10000;
int num_threads = 8;

// MD5 хеш
std::string md5_hash(const std::string& str) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(str.c_str()), str.size(), digest);

    char md5_string[33];
    for (int i = 0; i < 16; i++) {
        sprintf(&md5_string[i * 2], "%02x", digest[i]);
    }
    return std::string(md5_string);
}

// Преобразование числа в строку с использованием алфавита
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

    if (result.size() < set_size) {
        result = std::string(set_size - result.size(), alphabet[0]) + result;
    }

    return result;
}

void brute_force_avx(int start_index, int total_combinations, int base) {
    unsigned char result[MD5_DIGEST_LENGTH];

    for (int i = start_index; i < total_combinations; i += num_threads) {
        if (found.load()) {
            break;
        }

        std::string candidate = convertToBase(i, base, characters, password_length);

        MD5(reinterpret_cast<const unsigned char*>(candidate.c_str()), candidate.size(), result);

        std::string candidate_hash = md5_hash(candidate);
        if (candidate_hash == target_md5_hash) {
            std::cout << "Password found: " << candidate << std::endl;
            found.store(true);
            break;
        }
    }
}

void brute_force_md5_avx() {
    int total_combinations = static_cast<int>(pow(characters.size(), password_length));

    // Для AVX обработаем несколько паролей одновременно
    for (int i = 0; i < num_threads; ++i) {
        brute_force_avx(i, total_combinations, characters.size());
        if (found.load()) {
            break;  // Если флаг найден, выходим из цикла
        }
    }
}

void parse_arguments(int argc, char* argv[]) {
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--nt=") == 0) {
            num_threads = std::stoi(arg.substr(5));
        } else if (arg.find("--ch=") == 0) {
            chunk_size = std::stoi(arg.substr(5));
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            exit(1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <target_password> [--nt=<num>] [--ch=<size>]" << std::endl;
        return 1;
    }

    const std::string target_password = argv[1];
    password_length = static_cast<int>(target_password.size());

    target_md5_hash = md5_hash(target_password);
    std::cout << "Target MD5 hash: " << target_md5_hash << std::endl;

    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    auto total_start = std::chrono::high_resolution_clock::now();

    parse_arguments(argc, argv);

    brute_force_md5_avx(); // Использование AVX для брутфорса

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    std::cout << "Total execution time: " << total_duration.count() << " seconds." << std::endl;

    return 0;
}
