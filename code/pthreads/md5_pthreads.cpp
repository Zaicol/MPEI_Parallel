#include <iostream>
#include <string>
#include <openssl/md5.h>
#include <pthread.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <unistd.h> // sysconf

// Global
std::string target_md5_hash;
std::string characters;
int password_length;
bool found = false;
pthread_mutex_t found_mutex;
int chunk_size = 10000;
int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
int thread_chunk_size;

std::string md5_hash(const std::string& str) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(str.c_str()), str.size(), digest);

    char md5_string[33];
    for (int i = 0; i < 16; i++) {
        sprintf(&md5_string[i * 2], "%02x", digest[i]);
    }
    return std::string(md5_string);
}

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

// One thread bruteforce
void* brute_force_thread(void* arg) {
    int thread_id = *(int*)arg;
    delete (int*)arg;
    int base = static_cast<int>(characters.size());
    int total_combinations = static_cast<int>(pow(base, password_length));
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);

    for (int i = thread_id; i < total_combinations; i += num_threads) {
        if ((i - thread_id) % thread_chunk_size == 0) {
            pthread_mutex_lock(&found_mutex);
            if (found) {
                pthread_mutex_unlock(&found_mutex);
                return nullptr;
            }
            pthread_mutex_unlock(&found_mutex);
        }
        std::string candidate = convertToBase(i, base, characters, password_length);

        if (md5_hash(candidate) == target_md5_hash) {
            pthread_mutex_lock(&found_mutex);
            if (!found) {
                std::cout << "Password found: " << candidate << std::endl;
                found = true;
            }
            pthread_mutex_unlock(&found_mutex);
            return nullptr;
        }
    }
    return nullptr;
}

// Threads managing
void brute_force_md5(int num_threads) {
    std::vector<pthread_t> threads(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        int thread_id = i;
        pthread_create(&threads[i], nullptr, brute_force_thread, new int(thread_id));
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
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

    pthread_mutex_init(&found_mutex, nullptr);
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    auto total_start = std::chrono::high_resolution_clock::now();

    parse_arguments(argc, argv);
    thread_chunk_size = chunk_size * num_threads;

    brute_force_md5(num_threads);

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    std::cout << "Total execution time: " << total_duration.count() << " seconds." << std::endl;

    pthread_mutex_destroy(&found_mutex);

    return 0;
}
