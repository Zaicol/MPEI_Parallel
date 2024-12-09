// Рассчёты места
// Device 0: NVIDIA GeForce RTX 4070
// Max threads per block: 1024
// Max grid dimensions: (2147483647, 65535, 65535)
// Max shared memory per block: 49152 bytes
// Max threads per SM: 1536
// Total number of SM: 46
// Max registers per SM: 65536

// В локальной памяти каждого потока нужно хранить следующие данные:
// A, B, C, D, F, G - переменные MD5, uint32_t = 1 регистр, соответственно всего 6 регистров
// M[16] - 16 слов по uint32_t, соответственно всего 16 регистров
// Итого 16 + 6 = 22 регистра на поток

// Всего доступно 1536 потоков на один SM
// Соответственно, 1536 * 22 = 33664 регистров на SM. 33664 < 65526 => все данные уместятся

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>

using namespace std;

// Определяем константы MD5
__constant__ uint32_t K[64];
__constant__ uint32_t s[64];
__constant__ uint32_t A_init;
__constant__ uint32_t B_init;
__constant__ uint32_t C_init;
__constant__ uint32_t D_init;

// Общая переменная для целевого MD5 хэша
__constant__ uint32_t target_md5_hash_bytes[4];

// Размеры сетки и блоков
constexpr int threadsPerBlock = 512;
constexpr int blocksPerGrid = 3 * 46;
constexpr int base = 62;

string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
int password_length;

// Прототип функции для получения строки по индексу
string getString(int index)
{
    string result;
    int quotient = index;

    do
    {
        int remainder = quotient % base;
        result = alphabet[remainder] + result;
        quotient /= base;
    } while (quotient > 0);

    // Дополняем строку, если она короче ожидаемого размера
    if (result.size() < password_length)
    {
        result = string(password_length - result.size(), alphabet[0]) + result;
    }
    return result;
}

// Функция для преобразования строки MD5 в массив байтов
void md5StringToBytes(const string &hashString, uint32_t *hashBytes)
{
    if (hashString.length() != 32)
    {
        throw invalid_argument("It is not MD5 hash, MD5 hash has 32 characters.");
    }

    for (size_t i = 0; i < 4; ++i)
    {
        hashBytes[i] = static_cast<uint32_t>(
            stoul(hashString.substr(i * 8, 8), nullptr, 16));
    }
}

// Определяем функции F, G, H, I согласно MD5 спецификации
__device__ uint32_t F(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) | (~x & z);
}

__device__ uint32_t G(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & z) | (y & ~z);
}

__device__ uint32_t H(uint32_t x, uint32_t y, uint32_t z)
{
    return x ^ y ^ z;
}

__device__ uint32_t I(uint32_t x, uint32_t y, uint32_t z)
{
    return y ^ (x | ~z);
}

// MD5 ядро для обработки входящих данных
__global__ void computeManyMD5(const char *inputStrings, int numStrings, int password_length, int *foundIndex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numStrings)
        return;

    uint32_t A = A_init;
    uint32_t B = B_init;
    uint32_t C = C_init;
    uint32_t D = D_init;

    uint32_t M[16] = {0};

    const char *currentString = &inputStrings[idx * password_length];

    for (int i = 0; i < password_length; ++i)
    {
        M[i / 4] |= static_cast<uint32_t>(currentString[i]) << ((i % 4) * 8);
    }

    M[password_length / 4] |= (1U << ((password_length % 4) * 8));
    uint64_t bitLength = static_cast<uint64_t>(password_length) * 8;
    M[14] = static_cast<uint32_t>(bitLength);
    M[15] = static_cast<uint32_t>(bitLength >> 32);

    for (int i = 0; i < 64; ++i)
    {
        uint32_t F_val, g;

        if (i < 16)
        {
            F_val = F(B, C, D);
            g = i;
        }
        else if (i < 32)
        {
            F_val = G(B, C, D);
            g = (5 * i + 1) % 16;
        }
        else if (i < 48)
        {
            F_val = H(B, C, D);
            g = (3 * i + 5) % 16;
        }
        else
        {
            F_val = I(B, C, D);
            g = (7 * i) % 16;
        }

        uint32_t temp = D;
        D = C;
        C = B;
        B = B + ((A + F_val + K[i] + M[g]) << s[i]);
        A = temp;
    }

    A += A_init;
    B += B_init;
    C += C_init;
    D += D_init;

    bool isEqual = (A == target_md5_hash_bytes[0]) &&
                   (B == target_md5_hash_bytes[1]) &&
                   (C == target_md5_hash_bytes[2]) &&
                   (D == target_md5_hash_bytes[3]);

    if (isEqual)
    {
        printf("\n\nFound password\nThread %d ( %d  %d ): Final A: %u, B: %u, C: %u, D: %u\n\n", idx, blockIdx.x, threadIdx.x, A, B, C, D);
        *foundIndex = idx;
    }
    // else
    // {
    //     if (threadIdx.x < 3 && blockIdx.x < 120)
    //     {
    //         printf("Thread %d ( %d\t%d ): Final A: %u, B: %u, C: %u, D: %u\ntarget A: %u, target B: %u, target C: %u, target D: %u\n",
    //                idx, blockIdx.x, threadIdx.x, A, B, C, D,
    //                target_md5_hash_bytes[0], target_md5_hash_bytes[1], target_md5_hash_bytes[2], target_md5_hash_bytes[3]);
    //     }
    // }
}

__global__ void computeOneMD5(const char *inputString, int password_length, uint32_t *result)
{
    uint32_t A = A_init;
    uint32_t B = B_init;
    uint32_t C = C_init;
    uint32_t D = D_init;

    uint32_t M[16] = {0};

    // Подготовка данных
    for (int i = 0; i < password_length; ++i)
    {
        M[i / 4] |= static_cast<uint32_t>(inputString[i]) << ((i % 4) * 8);
    }

    M[password_length / 4] |= (1U << ((password_length % 4) * 8));
    uint64_t bitLength = static_cast<uint64_t>(password_length) * 8;
    M[14] = static_cast<uint32_t>(bitLength);
    M[15] = static_cast<uint32_t>(bitLength >> 32);

    // Основной MD5 цикл
    for (int i = 0; i < 64; ++i)
    {
        uint32_t F_val, g;

        if (i < 16)
        {
            F_val = F(B, C, D);
            g = i;
        }
        else if (i < 32)
        {
            F_val = G(B, C, D);
            g = (5 * i + 1) % 16;
        }
        else if (i < 48)
        {
            F_val = H(B, C, D);
            g = (3 * i + 5) % 16;
        }
        else
        {
            F_val = I(B, C, D);
            g = (7 * i) % 16;
        }

        uint32_t temp = D;
        D = C;
        C = B;
        B = B + ((A + F_val + K[i] + M[g]) << s[i]);
        A = temp;
    }

    // Финальные вычисления
    A += A_init;
    B += B_init;
    C += C_init;
    D += D_init;

    // Сохранение результата
    result[0] = A;
    result[1] = B;
    result[2] = C;
    result[3] = D;
}

void md5_hash(const string &target_password, uint32_t *output)
{
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t) * 4);

    char *d_inputString;
    cudaMalloc(&d_inputString, target_password.size());
    cudaMemcpy(d_inputString, target_password.c_str(), target_password.size(), cudaMemcpyHostToDevice);

    // Запуск ядра
    computeOneMD5<<<1, 1>>>(d_inputString, target_password.size(), d_result);

    // Переносим вычисленный результат обратно
    cudaMemcpy(output, d_result, sizeof(uint32_t) * 4, cudaMemcpyDeviceToHost);

    cudaFree(d_inputString);
    cudaFree(d_result);
}

// CUDA Error check
void checkCudaError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("\nCUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// Парсинг аргументов
bool parseArguments(int argc, char **argv, string &target_password)
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <target_password> [password_length]" << endl;
        return false;
    }

    target_password = argv[1];
    if (argc > 2)
    {
        password_length = atoi(argv[2]);
    }
    else
    {
        password_length = target_password.size();
    }
    cout << "Password length: " << password_length << endl;
    return true;
}

// Подготовка данных MD5 и копирование в глобальные переменные
void prepareMD5Hash(const string &target_password)
{
    uint32_t host_hash[4] = {0};
    md5_hash(target_password, host_hash);
    for (int i = 0; i < 4; ++i)
    {
        cout << host_hash[i] << " ";
    }
    cout << endl;
    cudaMemcpyToSymbol(target_md5_hash_bytes, host_hash, sizeof(uint32_t) * 4);
}

// Подготовка данных для ядра
void prepareInputData(vector<char> &flatInputStrings, vector<string> &inputStrings, int totalThreads, int iter_i)
{
    for (int i = 0; i < min(totalThreads, (int)pow(62, password_length) - totalThreads * iter_i); ++i)
    {
        inputStrings[i] = getString(i + totalThreads * iter_i);
        memcpy(&flatInputStrings[i * password_length], inputStrings[i].c_str(), password_length);
    }
}

// Настройка памяти и копирование данных на устройство
void setupDeviceMemory(char **d_inputStrings, int totalChars, vector<char> &flatInputStrings)
{
    cudaMalloc(d_inputStrings, totalChars * sizeof(char));
    cudaMemcpy(*d_inputStrings, flatInputStrings.data(), totalChars * sizeof(char), cudaMemcpyHostToDevice);
}

// Основной запуск CUDA
void launchKernel(char *d_inputStrings, int totalThreads, int password_length, int *d_foundIndex)
{
    computeManyMD5<<<blocksPerGrid, threadsPerBlock>>>(d_inputStrings, totalThreads, password_length, d_foundIndex);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    string target_password;

    // Парсинг аргументов
    if (!parseArguments(argc, argv, target_password))
    {
        return 1;
    }

    cout << "Target password: " << target_password << endl;

    // Подготовка данных и MD5-хэша
    prepareMD5Hash(target_password);

    // Подготовка данных для запуска CUDA
    auto total_start = chrono::high_resolution_clock::now();

    const int total_combinations = static_cast<int>(pow(62, password_length));
    const int totalThreads = threadsPerBlock * blocksPerGrid; // 70656
    vector<string> inputStrings(totalThreads);
    vector<char> flatInputStrings(totalThreads * password_length);
    char *d_inputStrings;
    int *d_foundIndex, h_foundIndex;
    int totalChars = totalThreads * password_length;
    cudaMalloc(&d_inputStrings, totalChars * sizeof(char));
    cudaMalloc(&d_foundIndex, sizeof(int));
    cudaMemset(d_foundIndex, -1, sizeof(int));
    cout << "\nStarting computation..." << endl;

    for (int iter_i = 0; iter_i < (total_combinations + totalThreads - 1) / totalThreads; ++iter_i)
    {

        prepareInputData(flatInputStrings, inputStrings, totalThreads, iter_i);
        checkCudaError(cudaGetLastError());
        cudaMemcpy(d_inputStrings, flatInputStrings.data(), totalChars * sizeof(char), cudaMemcpyHostToDevice);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("\nCUDA error: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        // Запуск вычислений
        launchKernel(d_inputStrings, totalThreads, password_length, d_foundIndex);
        checkCudaError(cudaGetLastError());

        // Обработка результата
        cudaMemcpy(&h_foundIndex, d_foundIndex, sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaError(cudaGetLastError());
        if (h_foundIndex != -1)
        {
            cout << "Password found: " << getString(h_foundIndex + totalThreads * iter_i) << endl;
            // Выход из цикла
            break;
        }
        else
        {
            // cout << "Password not found in iteration " << iter_i + 1 << "." << endl;
        }
    }
    cout << "Iterations: " << total_combinations / totalThreads << endl;
    cudaFree(d_inputStrings);
    cudaFree(d_foundIndex);
    auto total_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_duration = total_end - total_start;
    cout << "\nTotal execution time: " << total_duration.count() << " seconds." << endl;

    return 0;
}
