// ReSharper disable CppTooWideScopeInitStatement
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>
#include <openssl/md5.h>


using namespace std;
int thread_rank, n_threads;
static string target_md5_hash, characters;
int password_length;

string convertToBase(int number, int base, const string &alphabet, int set_size) {
    if (base > alphabet.size()) {
        throw invalid_argument("Base exceeds the size of the alphabet");
    }

    string result;
    int quotient = number;

    do {
        int remainder = quotient % base;
        result = alphabet[remainder] + result;
        quotient /= base;
    } while (quotient > 0);

    if (result.size() < set_size) {
        result = string(set_size - result.size(), alphabet[0]) + result;
    }

    return result;
}

// Function to calculate the MD5 hash of a string
string md5_hash(const string &input) {
    unsigned char hash[MD5_DIGEST_LENGTH];

    MD5_CTX md5;
    MD5_Init(&md5);
    MD5_Update(&md5, input.c_str(), input.size());
    MD5_Final(hash, &md5);

    stringstream ss;

    for (unsigned char i: hash) {
        ss << hex << setw(2) << setfill('0') << static_cast<int>(i);
    }
    return ss.str();
}

bool brute_force_md5_range(const int start_index, const int end_index, const string &characters,
                           const string &target_hash, string &found_password, int subchunk_size,
                           int password_length) {
    for (int index = start_index; index < end_index; index += subchunk_size) {
        // cout << "Processing range [" << index << ", " << min(end_index, index + subchunk_size) << ")" <<
                //endl;
        for (
            int sub_index = index;
            sub_index < min(end_index, index + subchunk_size);
            ++sub_index
        ) {
            if (
                string candidate = convertToBase(sub_index, static_cast<int>(characters.size()), characters, password_length);
                md5_hash(candidate) == target_hash
            ) {
                found_password = candidate;
                cout << "Password found: " << found_password << "\tindex: " << sub_index << endl;
                break;
            }
        }
        int local_found = found_password.empty() ? 0 : 1;
        int global_found = 0;
        MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_found > 0) {
            if (local_found) {
                return true;
            }
            return false; // Stop search if password was found by any process
        }
    }
    return false;
}

bool brute_force_md5(string &found_password) {
    const int total_combinations = static_cast<int>(pow(characters.size(), password_length));
    const int chunk_size = total_combinations / n_threads;
    const int start_index = thread_rank * chunk_size;
    const int end_index = (thread_rank == n_threads - 1) ? total_combinations : start_index + chunk_size;

    constexpr int subchunk_size = 10000;
    bool found_flag = brute_force_md5_range(start_index, end_index, characters, target_md5_hash, found_password, subchunk_size, password_length);

    int found_rank = found_flag ? thread_rank : -1;
    MPI_Allreduce(MPI_IN_PLACE, &found_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    char password_buffer[256] = {0};
    if (found_flag) {
        strncpy(password_buffer, found_password.c_str(), sizeof(password_buffer));
    }
    MPI_Bcast(password_buffer, sizeof(password_buffer), MPI_CHAR, found_rank, MPI_COMM_WORLD);

    if (found_flag) {
        found_password = password_buffer;
    }
    return found_flag;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &thread_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_threads);

    if (argc != 2) {
        if (thread_rank == 0) {
            cerr << "Usage: " << argv[0] << " <target_password>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    const string target_password = argv[1];
    password_length = static_cast<int>(target_password.size());

    target_md5_hash = md5_hash(target_password);

    if (thread_rank == 0) {
        cout << "Target MD5 hash: " << target_md5_hash << endl;
        cout << "Starting brute-force attack with " << n_threads << " processes..." << endl;
    }

    string found_password;
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    MPI_Barrier(MPI_COMM_WORLD);
    const double total_start = MPI_Wtime();

    bool found = brute_force_md5(found_password);

    MPI_Barrier(MPI_COMM_WORLD);
    const double total_end = MPI_Wtime();

    if (thread_rank == 0) {
        if (found) {
            cout << "Password found: " << found_password << endl;
        } else {
            cout << "Password not found." << endl;
        }

        cout << "Total execution time: " << total_end - total_start << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}
