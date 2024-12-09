#include <iostream>
#include <sstream>
#include <string>

using namespace std;

string target_md5_hash;
uint8_t target_md5_hash_bytes[16];

void md5StringToBytes(const string &hashString, uint8_t *hashBytes)
{
    if (hashString.length() != 32)
    {
        throw invalid_argument("MD5 hash string must be exactly 32 characters.");
    }

    for (size_t i = 0; i < 16; ++i)
    {
        string byteString = hashString.substr(i * 2, 2);
        hashBytes[i] = static_cast<uint8_t>(stoi(byteString, nullptr, 16));
    }
}

int main()
{
    target_md5_hash = "c8229d750070c723b99ebcdd145a19db";
    cout << target_md5_hash_bytes[0] << endl;
    md5StringToBytes(target_md5_hash, target_md5_hash_bytes);
    cout << static_cast<int>(target_md5_hash_bytes[0]) << endl;
    istringstream converter(target_md5_hash);
    unsigned int value;
    converter >> hex >> value;
    cout << value;
    return 0;
}
