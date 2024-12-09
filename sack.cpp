
#include <iostream>
#include <vector>
#include <pthread.h>
#include <algorithm>
#include <ctime>


using namespace std;

struct KnapsackData {
    int start;                  
    int end;                    
    int capacity;               
    vector<int> weights;        
    vector<int> values;         
    int max_value;              
};

void* knapsackThread(void* arg) {
    KnapsackData* data = (KnapsackData*)arg;
    int local_max_value = 0;

    
    for (int i = data->start; i < data->end; ++i) {
        int current_weight = 0;
        int current_value = 0;

        
        for (int j = 0; j < data->weights.size(); ++j) {
            if (i & (1 << j)) {
                current_weight += data->weights[j];
                current_value += data->values[j];
            }
        }

        
        if (current_weight <= data->capacity) {
            local_max_value = max(local_max_value, current_value);
           
        }
    }

    data->max_value = local_max_value;
    return nullptr;
}

int main() {
    
    int start_time = clock();

    int capacity = 150; 
    vector<int> weights = { 10, 20, 30, 40, 50, 10, 10, 20, 30, 40, 50, 10, 10, 20, 30, 40, 50, 10, 10, 20, 30, 40, 50, 10};
    vector<int> values = { 60, 100, 120, 70, 20, 100,  60, 100, 120, 70, 20, 100,  60, 100, 120, 70, 20, 100,  60, 100, 120, 70, 20, 100, };

    int n = weights.size(); 

    const int num_threads = 4; 
    pthread_t threads[num_threads];
    KnapsackData data[num_threads];
    int total_combinations = (1 << n);

    
    for (int i = 0; i < num_threads; ++i) {
        data[i].start = i * (total_combinations / num_threads);
        data[i].end = (i + 1) * (total_combinations / num_threads);
        data[i].capacity = capacity;
        data[i].weights = weights;
        data[i].values = values;
        data[i].max_value = 0;

        if (i == num_threads - 1) {
            
            data[i].end = total_combinations;
        }

        pthread_create(&threads[i], nullptr, knapsackThread, &data[i]);
    }

    
    for (int i = 0; i < num_threads; ++i) {
        printf("%d\n", i);
        pthread_join(threads[i], nullptr);
    }

    
    int global_max_value = 0;
    for (int i = 0; i < num_threads; ++i) {
        global_max_value = max(global_max_value, data[i].max_value);
    }
    unsigned int end_time = clock();

    printf("time: %d\n", end_time - start_time);

    cout << "Max cost: " << global_max_value << endl;

    return 0;
}
