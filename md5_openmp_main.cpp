#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <string>
#include <openssl/md5.h>


using namespace std;
string target_md5_hash;
string characters;
int password_length;

// Функция для получения MD5-хэша
string md5_hash(const string& str) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(str.c_str()), str.size(), digest);

    char md5_string[33];
    for (int i = 0; i < 16; i++) {
        sprintf(&md5_string[i * 2], "%02x", digest[i]);
    }
    return string(md5_string);
}

// Функция для перевода числа в строку на основе указанного алфавита (система счисления)
string convertToBase(int number, int base, const string& alphabet, int set_size) {
    if (base > static_cast<int>(alphabet.size())) {
        throw invalid_argument("Base exceeds the size of the alphabet");
    }

    string result;
    int quotient = number;

    do {
        int remainder = quotient % base;
        result = alphabet[remainder] + result;
        quotient /= base;
    } while (quotient > 0);

    // Дополняем строку, если она короче ожидаемого размера
    if (result.size() < set_size) {
        result = string(set_size - result.size(), alphabet[0]) + result;
    }

    return result;
}

// Функция для брутфорса с параллелизацией OpenMP
void brute_force_md5(const string& target_md5, const string& characters, int password_length) {
    int found = 0;
    const int base = static_cast<int>(characters.size());
    const int total_combinations = static_cast<int>(pow(base, password_length));

#pragma omp parallel for schedule(runtime) shared(found)
    for (int i = 0; i < total_combinations; ++i) {
        if (found) break; // Останавливаем поток, если пароль найден

        string candidate = convertToBase(i, base, characters, password_length);

        // Проверяем текущий кандидат
        if (md5_hash(candidate) == target_md5) {
#pragma omp critical
            {
                if (!found) {
                    cout << "Password found: " << candidate << endl;
                    found = 1;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <target_password>" << endl;
        return 1;
    }

    const string target_password = argv[1];
    password_length = static_cast<int>(target_password.size());

    target_md5_hash = md5_hash(target_password);
    cout << "Target MD5 hash: " << target_md5_hash << endl;

    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    auto total_start = chrono::high_resolution_clock::now();

    brute_force_md5(target_md5_hash, characters, password_length);

    auto total_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_duration = total_end - total_start;
    cout << "Total execution time: " << total_duration.count() << " seconds." << endl;

    return 0;
}
