#include <iostream>
#include <string>
#include <openssl/md5.h>
#include <thread>
#include <vector>
#include <cmath>
#include <mutex>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <sstream>

#include <immintrin.h> // Для AVX и SSE
#include <cstdint>     // Для uint32_t, uint8_t
#include <cstring>     // Для memcpy

using namespace std;

// Global
string target_md5_hash;
string characters;
int password_length;
string found_password;
string target_password_g;

#define mmp _mm256_add_epi32

// Таблица синусов MD5
const __m256i K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, // K[ 0.. 3]
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, // K[ 4.. 7]
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, // K[ 8..11]
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, // K[12..15]
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, // K[16..19]
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, // K[20..23]
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, // K[24..27]
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, // K[28..31]
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, // K[32..35]
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, // K[36..39]
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, // K[40..43]
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, // K[44..47]
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, // K[48..51]
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, // K[52..55]
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, // K[56..59]
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391  // K[60..63]
};

const int s[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

inline __m256i leftRotate(__m256i x, int n)
{
    __m256i left = _mm256_slli_epi32(x, n);
    __m256i right = _mm256_srli_epi32(x, 32 - n);
    return _mm256_or_si256(left, right);
}

// Обработка одного блока (512 бит)
void processBlock(__m256i &A, __m256i &B, __m256i &C, __m256i &D, const uint8_t block[8][64], bool same_data = false)
{
    __m256i M[16];
    if (!same_data)
    {
        for (int i = 0; i < 16; ++i)
        {
            uint32_t word[8]; // Массив для одного слова из 8 блоков

            // Собираем слово из каждого из 8 блоков
            for (int j = 0; j < 8; ++j)
            {
                word[j] = *reinterpret_cast<const uint32_t *>(&block[j][i * 4]); // 4 байта
            }

            // Загружаем векторное значение
            M[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(word));
        }
    }
    else
    {
        for (int i = 0; i < 8; ++i)
        {
            std::cout << "Block " << i << ": ";
            for (int j = 0; j < 64; ++j)
            {
                std::cout << std::hex << (int)block[i][j] << " ";
            }
            std::cout << std::endl;
        }
        for (int i = 0; i < 16; ++i)
        {
            M[i] = _mm256_set1_epi32(*reinterpret_cast<const uint32_t *>(&block[0][i * 4]));
        }
        for (int i = 0; i < 16; ++i)
        {
            uint32_t value = *reinterpret_cast<const uint32_t *>(&block[0][i * 4]);
            std::cout << "M[" << i << "] = " << std::hex << value << std::endl;
            M[i] = _mm256_set1_epi32(value);
        }
    }

    __m256i a = A, b = B, c = C, d = D;

    // 4 этапа по 16 раундов
    for (int i = 0; i < 64; ++i)
    {
        __m256i f;
        uint32_t g;
        if (i < 16)
        {
            f = _mm256_or_si256(
                _mm256_and_si256(b, c),
                _mm256_andnot_si256(b, d));
            g = i;
        }
        else if (i < 32)
        {
            f = _mm256_or_si256(
                _mm256_and_si256(d, b),
                _mm256_andnot_si256(d, c));
            g = (5 * i + 1) % 16;
        }
        else if (i < 48)
        {
            f = _mm256_xor_si256(
                _mm256_xor_si256(b, c), d);
            g = (3 * i + 5) % 16;
        }
        else
        {
            f = _mm256_xor_si256(c,
                                 _mm256_or_si256(b, _mm256_xor_si256(d, _mm256_set1_epi32(0xFFFFFFFF))));
            g = (7 * i) % 16;
        }

        __m256i temp_a = a;
        a = d;
        d = c;
        c = b;
        b = mmp(leftRotate(mmp(mmp(mmp(temp_a, f), M[g]), K[i]), s[i]), b);
    }

    // Переносится на следующий блок
    A = _mm256_add_epi32(A, a);
    B = _mm256_add_epi32(B, b);
    C = _mm256_add_epi32(C, c);
    D = _mm256_add_epi32(D, d);
    alignas(32) uint32_t result_a[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_a), A);

    for (int i = 0; i < 8; ++i)
    {
        std::cout << "Lane " << i << " result: " << std::hex << result_a[i] << std::endl;
    }
}

string hashToString(const uint8_t hash[16])
{
    ostringstream oss;
    for (int i = 0; i < 16; ++i)
    {
        oss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    return oss.str();
}

void extractHashes(const __m256i &A, const __m256i &B, const __m256i &C, const __m256i &D, uint8_t hashes[8][16])
{
    alignas(32) uint32_t a[8], b[8], c[8], d[8];

    // Извлекаем значения из векторных регистров в массивы
    _mm256_store_si256(reinterpret_cast<__m256i *>(a), A);
    _mm256_store_si256(reinterpret_cast<__m256i *>(b), B);
    _mm256_store_si256(reinterpret_cast<__m256i *>(c), C);
    _mm256_store_si256(reinterpret_cast<__m256i *>(d), D);
    // Сохраняем состояния в формате MD5
    for (int i = 0; i < 8; ++i)
    {
        // Копируем 4 байта из каждого состояния
        memcpy(hashes[i], &a[i], 4);      // Первые 4 байта (A)
        memcpy(hashes[i] + 4, &b[i], 4);  // Следующие 4 байта (B)
        memcpy(hashes[i] + 8, &c[i], 4);  // Следующие 4 байта (C)
        memcpy(hashes[i] + 12, &d[i], 4); // Последние 4 байта (D)
    }
}

// Подготовка данных (padding)
void padData(const uint8_t *input, size_t size, uint8_t *output)
{
    // Сначала копируем всю входную строку в выходную
    memcpy(output, input, size);
    // Добавляем бит 1 и устанавливаем длину
    output[size] = 0x80;
    size_t padded = size + 1;

    // Используя проверку размера, добавляем дополнительные биты
    while ((padded % 64) != 56)
    {
        output[padded++] = 0;
    }

    // Длина входной строки в битах (size - размер в байтах)
    // Надеюсь тут всё правильно
    uint64_t bitLength = size * 8;
    memcpy(output + padded, &bitLength, sizeof(bitLength));
}

// Основная функция MD5
void md5(const uint8_t *data, size_t size, uint8_t *hash)
{
    uint8_t *paddedData = new uint8_t[(size + 72 + 63) & ~63];
    padData(data, size, paddedData);

    // Начальные значения MD5
    __m256i A = _mm256_set1_epi32(0x67452301);
    __m256i B = _mm256_set1_epi32(0xEFCDAB89);
    __m256i C = _mm256_set1_epi32(0x98BADCFE);
    __m256i D = _mm256_set1_epi32(0x10325476);

    // processBlock(A, B, C, D, paddedData);

    // Запись результата в hash
    memcpy(hash, &A, sizeof(A));
    memcpy(hash + 4, &B, sizeof(B));
    memcpy(hash + 8, &C, sizeof(C));
    memcpy(hash + 12, &D, sizeof(D));
}

string md5_hash(const string &str)
{
    uint8_t padded_strings[8][64];
    for (int j = 0; j < 8; j++)
    {
        string converted = str;
        padData(reinterpret_cast<const uint8_t *>(converted.c_str()), converted.length(), padded_strings[j]);
    }
    // Начальные значения MD5
    __m256i A = _mm256_set1_epi32(0x67452301);
    __m256i B = _mm256_set1_epi32(0xEFCDAB89);
    __m256i C = _mm256_set1_epi32(0x98BADCFE);
    __m256i D = _mm256_set1_epi32(0x10325476);

    // Обработка блоков
    processBlock(A, B, C, D, padded_strings, true);

    uint8_t hashes[8][16];

    // Извлекаем значения из векторных регистров в массивы
    extractHashes(A, B, C, D, hashes);

    for (int i = 0; i < 8; i++)
    {
        cout << "Hash " << i + 1 << ": " << hashToString(hashes[i]) << endl;
    }

    return hashToString(hashes[0]);
}

string convertToBase(int number, int base, const string &alphabet, int set_size)
{
    if (base > static_cast<int>(alphabet.size()))
    {
        throw invalid_argument("Base exceeds the size of the alphabet");
    }

    string result;
    int quotient = number;

    do
    {
        int remainder = quotient % base;
        result = alphabet[remainder] + result;
        quotient /= base;
    } while (quotient > 0);

    // Дополняем строку, если она короче ожидаемого размера
    if (result.size() < set_size)
    {
        result = string(set_size - result.size(), alphabet[0]) + result;
    }

    return result;
}

// One thread bruteforce
void brute_force_md5()
{
    int base = static_cast<int>(characters.size());
    int total_combinations = static_cast<int>(pow(base, password_length));

    for (int i = 0; i < total_combinations; i += 8)
    {
        // Check 8 passwords at once to increase parallelism
        uint8_t padded_strings[8][64];
        string passwords[8];
        for (int j = 0; j < 8; j++)
        {
            passwords[j] = convertToBase(i + j, base, characters, password_length);
            padData(reinterpret_cast<const uint8_t *>(passwords[j].c_str()), passwords[j].length(), padded_strings[j]);
        }
        // Начальные значения MD5
        __m256i A = _mm256_set1_epi32(0x67452301);
        __m256i B = _mm256_set1_epi32(0xEFCDAB89);
        __m256i C = _mm256_set1_epi32(0x98BADCFE);
        __m256i D = _mm256_set1_epi32(0x10325476);

        // Обработка блоков
        processBlock(A, B, C, D, padded_strings);

        uint8_t hashes[8][16];

        // Извлекаем значения из векторных регистров в массивы
        extractHashes(A, B, C, D, hashes);
        for (int h = 0; h < 8; ++h)
        {
            std::cout << "Hash " << i + h << ": " << hashToString(hashes[h]) << std::endl;
            if (hashToString(hashes[h]) == target_md5_hash || passwords[h] == target_password_g)
            {
                cout << "Password found: " << passwords[h] << "\tindex: " << i + h << endl;
                return;
            }
        }
    }
}

int main(int argc, char *argv[])
{

    const string target_password = argv[1];
    target_password_g = target_password;
    password_length = static_cast<int>(target_password.size());

    target_md5_hash = md5_hash(target_password);
    cout << "Target MD5 hash: " << target_md5_hash << endl;

    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    auto total_start = chrono::high_resolution_clock::now();

    brute_force_md5();

    auto total_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_duration = total_end - total_start;
    cout << "Total execution time: " << total_duration.count() << " seconds." << endl;

    return 0;
}
