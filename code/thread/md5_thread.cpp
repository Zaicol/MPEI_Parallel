#include <iostream>
#include <string>
#include <openssl/md5.h>
#include <thread>
#include <vector>
#include <cmath>
#include <mutex>
#include <unistd.h>
#include <chrono>
#include <sstream>

using namespace std;

// Global
string target_md5_hash;
string characters;
int password_length;
bool found = false;
mutex found_mutex;
int chunk_size = 10000;
int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
int thread_chunk_size;

string md5_hash(const string &str) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char *>(str.c_str()), str.size(), digest);

    char md5_string[33];
    for (int i = 0; i < 16; i++) {
        sprintf(&md5_string[i * 2], "%02x", digest[i]);
    }
    return string(md5_string);
}

string convertToBase(int number, int base, const string &alphabet, int set_size) {
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

// One thread bruteforce
void brute_force_thread(int thread_id, int num_threads) {
    int base = static_cast<int>(characters.size());
    int total_combinations = static_cast<int>(pow(base, password_length));

    for (int i = thread_id; i < total_combinations; i += num_threads) {
        if ((i - thread_id) % thread_chunk_size == 0) {
            lock_guard <mutex> lock(found_mutex);
            if (found) {
                return;
            }
        }

        string candidate = convertToBase(i, base, characters, password_length);

        if (md5_hash(candidate) == target_md5_hash) {
            lock_guard <mutex> lock(found_mutex);
            if (!found) {
                cout << "Password found: " << candidate << endl;
                found = true;
            }
            return;
        }
    }
}

// Threads managing
void brute_force_md5(int num_threads) {
    vector <thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(brute_force_thread, i, num_threads);
    }

    for (auto &t: threads) {
        t.join();
    }
}

void parse_arguments(int argc, char* argv[]) {
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg.find("--nt=") == 0) {
            num_threads = stoi(arg.substr(5));
        } else if (arg.find("--ch=") == 0) {
            chunk_size = stoi(arg.substr(5));
        } else {
            cerr << "Unknown argument: " << arg << endl;
            exit(1);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <target_password> [--num_threads=<num>] [--chunk_size=<size>]" << endl;
        return 1;
    }

    const string target_password = argv[1];
    password_length = static_cast<int>(target_password.size());

    target_md5_hash = md5_hash(target_password);
    cout << "Target MD5 hash: " << target_md5_hash << endl;

    parse_arguments(argc, argv);
    thread_chunk_size = chunk_size * num_threads;

    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    auto total_start = chrono::high_resolution_clock::now();

    brute_force_md5(num_threads);

    auto total_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_duration = total_end - total_start;
    cout << "Total execution time: " << total_duration.count() << " seconds." << endl;

    return 0;
}
