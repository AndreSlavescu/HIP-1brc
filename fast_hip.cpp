#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <cstring>

#define MAX_CITY_BYTE 100  
#define MAX_THREADS_PER_BLOCK 1024

struct Part {
    long long offset; long long length;
};

struct Stat {
    char city[MAX_CITY_BYTE];
    float min = INFINITY; float max = -INFINITY; float sum = 0;
    int count = 0;
    Stat() {}
    Stat(const std::string& init_city) {
        strncpy(city, init_city.c_str(), init_city.size());
        city[init_city.size()] = '\0';
    }
};

__device__ static float atomicMin(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
        __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float hip_atof(char* str) {
    float result = 0.0f;
    int sign = 1; int decimal = 0; int digits = 0;

    if (*str == '-') {
        sign = -1;
        ++str;
    }

    while (*str >= '0' && *str <= '9') {
        result = result * 10.0f + (*str - '0');
        ++str;
        ++digits;
    }

    if (*str == '.') {
        ++str;
        while (*str >= '0' && *str <= '9') {
            result = result * 10.0f + (*str - '0');
            ++str;
            ++digits;
            ++decimal;
        }
    }
    result *= sign;

    while (decimal > 0) {
        result /= 10.0f;
        --decimal;
    }
    return result;
}

__device__ int hip_strcmp(const char* p1, const char* p2) {
    const unsigned char *s1 = (const unsigned char *) p1;
    const unsigned char *s2 = (const unsigned char *) p2;
    unsigned char c1, c2;
    do {
        c1 = (unsigned char) *s1++;
        c2 = (unsigned char) *s2++;
        if (c1 == '\0')
            return c1 - c2;
    } while (c1 == c2);
    return c1 - c2;
}

__device__ int get_index(char* cities, char* city_target, int n_city) {
    int left = 0;
    int right = n_city - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        const char* city_query = cities + mid * MAX_CITY_BYTE;

        int cmp = hip_strcmp(city_query, city_target);
        if (cmp == 0)
            return mid;
        else if (cmp < 0)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

__global__ void process_buffer(char* buffer, Part* parts, Stat* stats, char* cities, int n_city, long long buffer_offset, int part_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x * blockDim.x + tx;

    if (bx >= part_size)  
        return;

    int index = 0;
    bool parsing_city = true;

    char city[MAX_CITY_BYTE];
    char floatstr[5]; 

    for (int i = 0; i < parts[bx].length; i++) {
        char c = buffer[parts[bx].offset-buffer_offset + i];
        if (parsing_city) {  
            if (c == ';') {
                city[index] = '\0';
                index = 0;
                parsing_city = false;
            } else {
                city[index] = c;
                index++;
            }
        } else {  
            if (c == '\n') {
                floatstr[index] = '\0';

                int stat_index = get_index(cities, city, n_city);
                float temp = hip_atof(floatstr);

                atomicMin(&stats[stat_index].min, temp);
                atomicMax(&stats[stat_index].max, temp);
                atomicAdd(&stats[stat_index].sum, temp);
                atomicAdd(&stats[stat_index].count, 1);

                parsing_city = true;
                index = 0;
                floatstr[0] = '\0'; city[0] = '\0';
            } else {
                floatstr[index] = c;
                index++;
            }
        }
    }
}

std::vector<Part> split_file(std::string input_path, int num_parts) {
    std::ifstream file(input_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    long long split_size = size / num_parts;

    std::cout << "Total file size: " << size << ", split size: " << split_size << std::endl;

    long long offset = 0;
    std::vector<Part> parts;
    while (offset < size) {
        long long seek_offset = std::max(offset + split_size - MAX_CITY_BYTE, 0LL);
        if (seek_offset > size) {
            parts.back().length += size-offset;
            break;
        }
        file.seekg(seek_offset, std::ios::beg);
        char buf[MAX_CITY_BYTE];
        file.read(buf, MAX_CITY_BYTE);

        std::streamsize n = file.gcount();
        std::streamsize newline = -1;
        for (int i = n - 1; i >= 0; --i) {
            if (buf[i] == '\n') {
                newline = i;
                break;
            }
        }
        int remaining = n - newline - 1;
        long long next_offset = seek_offset + n - remaining;
        parts.push_back({offset, next_offset-offset});
        offset = next_offset;
    }
    file.close();
    return parts;
}

std::set<std::string> get_cities() {
    std::ifstream weather_file("data/weather_stations.csv");
    std::string line;
    std::set<std::string> all_cities;

    while (getline(weather_file, line)) {
        std::istringstream iss(line);
        if (line[0] == '#')
            continue;
        std::string station;
        std::getline(iss, station, ';');
        all_cities.insert(station);
    }
    weather_file.close();
    return all_cities;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <file path> <num parts> <batch size>" << std::endl;
        return 1;
    }

    std::set<std::string> all_cities = get_cities();

    int n_city = all_cities.size();
    Stat* stats = new Stat[n_city];
    int index = 0;
    char cities[MAX_CITY_BYTE * n_city] = {'\0'};

    for (const auto& city : all_cities) {
        stats[index] = Stat(city);
        strcpy(cities + (index * MAX_CITY_BYTE), city.c_str());
        index++;
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::string input_path = argv[1];
    int num_parts = atoi(argv[2]);
    int batch_size = atoi(argv[3]);

    std::vector<Part> parts = split_file(input_path, num_parts);
    num_parts = parts.size();

    std::cout << "Required GPU RAM Size (GB): " <<  parts[0].length * batch_size / 1'000'000'000.0 << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken finding parts: " << elapsed.count() << " seconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    Stat* d_stats; 
    hipMalloc(&d_stats, n_city * sizeof(Stat));
    hipMemcpy(d_stats, stats, n_city * sizeof(Stat), hipMemcpyHostToDevice);

    char* d_buffer;  
    hipMalloc((void**) &d_buffer, 10'000'000'000 * sizeof(char));

    Part* d_parts;  
    hipMalloc(&d_parts, parts.size() * sizeof(Part));

    char* d_cities;  
    hipMalloc(&d_cities, MAX_CITY_BYTE * n_city * sizeof(char));
    hipMemcpy(d_cities, cities, MAX_CITY_BYTE * n_city * sizeof(char), hipMemcpyHostToDevice);

    std::ifstream file(input_path, std::ios::binary);
    for (int b = 0; b < num_parts; b += batch_size) {
        long long batch_file_size = 0;
        for (int bi = b; bi < std::min(b + batch_size, num_parts); bi++)
            batch_file_size += parts[bi].length;

        file.seekg(parts[b].offset, std::ios::beg);

        char* buffer = new char[batch_file_size];
        file.read(buffer, batch_file_size);

        hipMemcpy(d_buffer, buffer, batch_file_size * sizeof(char), hipMemcpyHostToDevice);

        int part_size = batch_size;
        if (b + batch_size > num_parts)
            part_size = num_parts - b;
        hipMemcpy(d_parts, parts.data() + b, part_size * sizeof(Part), hipMemcpyHostToDevice);

        int grid_blocks = std::ceil((float) part_size / MAX_THREADS_PER_BLOCK);

        hipLaunchKernelGGL(process_buffer, dim3(grid_blocks), dim3(MAX_THREADS_PER_BLOCK), 0, 0, d_buffer, d_parts, d_stats, d_cities, n_city, parts[b].offset, part_size);
        hipError_t error = hipGetLastError();
        if (error != hipSuccess)
            std::cerr << "Error: " << hipGetErrorString(error) << std::endl;

        delete[] buffer;
    }

    hipDeviceSynchronize();  
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken in HIP kernel: " << elapsed.count() << " seconds" << std::endl;

    hipMemcpy(stats, d_stats, n_city * sizeof(Stat), hipMemcpyDeviceToHost);
    std::ofstream measurements("hip_measurements.out");
    for (int i = 0; i < n_city; i++) {
        if (stats[i].count != 0) {
            float mean = stats[i].sum / stats[i].count;
            measurements << stats[i].city << "=" << stats[i].min << "/";
            measurements << std::fixed << std::setprecision(1) << mean << "/";
            measurements << stats[i].max << std::endl;
        }
    }

    return 0;
}
