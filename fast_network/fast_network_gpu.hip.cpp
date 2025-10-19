#include "common.h"
#include <vector>
#include <iostream>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#include <algorithm>
#include <cstring>
#include <numeric> // For std::accumulate (if needed, though atomics handle sum now)
// No pybind includes needed here

// --- Host-side permutation table ---
const uint8_t PERMUTATIONS_W2_HOST[720][6] = {
    #include "permutations_6.h"
};

// --- Global Device Pointers ---
uint8_t *d_global_images = nullptr;
uint8_t *d_global_labels = nullptr;
int g_num_images = 0;
uint8_t *d_global_validation_images = nullptr;
uint8_t *d_global_validation_labels = nullptr;
int g_num_validation_images = 0;


// --- KERNEL 1: Population Evaluation (with weights) ---
__global__ void evaluate_hybrid_kernel(
    const uint8_t* all_chromosomes,
    const uint8_t* all_images,
    const uint8_t* all_labels,
    float* all_fitness_scores,
    int population_size,
    int num_images,
    int mini_batch_size,
    unsigned long long seed,
    const float* class_weights) // <-- Class weights array (device pointer)
{
    // Shared memory for counts AND total weights
    __shared__ int block_correct_counts[256]; // Scaled by 1000 if using weights
    __shared__ float block_total_weights[256]; // Track total weight per thread

    int local_thread_id = hipThreadIdx_x;
    int block_id = hipBlockIdx_x;
    int images_per_thread = (mini_batch_size + hipBlockDim_x - 1) / hipBlockDim_x;

    // Initialize shared memory
    if (local_thread_id < 256) {
        block_correct_counts[local_thread_id] = 0;
        block_total_weights[local_thread_id] = 0.0f;
    }

    if (block_id < population_size) {
        Transformer transformer;
        const uint8_t* chromosome = all_chromosomes + (block_id * CHROMOSOME_LENGTH);
        hiprandStateXORWOW_t rng_state;
        hiprand_init(seed, block_id * hipBlockDim_x + local_thread_id, 0, &rng_state);

        for (int i = 0; i < images_per_thread; ++i) {
            int current_image_index_in_batch = local_thread_id + i * hipBlockDim_x;
            if (current_image_index_in_batch >= mini_batch_size) continue;

            // Sample image index
            unsigned int rand_idx = hiprand(&rng_state) % num_images;
            const uint8_t* image = all_images + (rand_idx * IMAGE_SIZE);
            uint8_t label = all_labels[rand_idx];

            // Get weight for this image's class
            float weight = (class_weights == nullptr) ? 1.0f : class_weights[label]; // Default to 1

            // Accumulate total weight processed by this thread
            atomicAdd(&block_total_weights[local_thread_id], weight);

            // Network processing logic
            uint8_t state[STATE_SIZE];
            for(int k=0; k<IMAGE_SIZE; ++k) state[k] = image[k];
            for(int k=IMAGE_SIZE; k<STATE_SIZE; ++k) state[k] = 0;
            for (int l = 0; l < LAYERS; ++l) {
                uint8_t next_state[STATE_SIZE];
                for (int t = 0; t < TRANSFORMERS_PER_LAYER; ++t) {
                    transformer.process(state + (t * 4), chromosome + ((l * TRANSFORMERS_PER_LAYER + t) * BITS_PER_TRANSFORMER), next_state + (t * 4));
                }
                for(int k=0; k<STATE_SIZE; ++k) state[k] = next_state[k];
                if (l < LAYERS - 1) {
                    transformer.perfect_shuffle(state, next_state);
                    for(int k=0; k<STATE_SIZE; ++k) state[k] = next_state[k];
                }
            }

            // Scoring logic
            int bin_scores[10] = {0};
            for (int k = 0; k < 40; ++k) {
                if (state[k*20 + 0]) bin_scores[0]++; if (state[k*20 + 2]) bin_scores[1]++;
                if (state[k*20 + 4]) bin_scores[2]++; if (state[k*20 + 6]) bin_scores[3]++;
                if (state[k*20 + 8]) bin_scores[4]++; if (state[k*20 + 10]) bin_scores[5]++;
                if (state[k*20 + 12]) bin_scores[6]++; if (state[k*20 + 14]) bin_scores[7]++;
                if (state[k*20 + 16]) bin_scores[8]++; if (state[k*20 + 18]) bin_scores[9]++;
            }
            int max_score = -1, winner_count = 0; int winners[10];
            for(int k=0; k<10; ++k) {
                 if(bin_scores[k] > max_score) { max_score = bin_scores[k]; winner_count = 1; winners[0] = k; }
                 else if (bin_scores[k] == max_score) { winners[winner_count++] = k; }
            }
            int predicted_digit = winners[hiprand(&rng_state) % winner_count];

            // Apply weight if correct
            if (predicted_digit == label) {
                // Accumulate weighted score (scaled by 1000)
                atomicAdd(&block_correct_counts[local_thread_id], static_cast<int>(weight * 1000));
            }
        }
    }

    __syncthreads();

    // Reduce results within the block
    if (local_thread_id == 0 && block_id < population_size) {
        long long total_weighted_correct_scaled = 0;
        float total_weight_processed = 0.0f;
        for (int i = 0; i < hipBlockDim_x; ++i) {
            total_weighted_correct_scaled += block_correct_counts[i];
            total_weight_processed += block_total_weights[i];
        }

        // Calculate final weighted fitness
        if (total_weight_processed > 0.0f) {
             all_fitness_scores[block_id] = (static_cast<float>(total_weighted_correct_scaled) / 1000.0f) / total_weight_processed;
        } else {
             all_fitness_scores[block_id] = 0.0f;
        }
    }
}


// --- KERNEL 2: Champion Evaluation (Grid-Stride) ---
__global__ void evaluate_champion_kernel(
    const uint8_t* chromosome,
    const uint8_t* validation_images,
    const uint8_t* validation_labels,
    int num_validation_images,
    int* confusion_matrix_flat)
{
    int global_thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int total_threads = hipGridDim_x * hipBlockDim_x;

    Transformer transformer;
    hiprandStateXORWOW_t rng_state;
    hiprand_init(12345, global_thread_id, 0, &rng_state);

    for (int image_idx = global_thread_id; image_idx < num_validation_images; image_idx += total_threads) {
        const uint8_t* image = validation_images + (image_idx * IMAGE_SIZE);
        uint8_t label = validation_labels[image_idx];

        uint8_t state[STATE_SIZE];
        for(int k=0; k<IMAGE_SIZE; ++k) state[k] = image[k];
        for(int k=IMAGE_SIZE; k<STATE_SIZE; ++k) state[k] = 0;

        for (int l = 0; l < LAYERS; ++l) {
            uint8_t next_state[STATE_SIZE];
            for (int t = 0; t < TRANSFORMERS_PER_LAYER; ++t) {
                transformer.process(state + (t * 4), chromosome + ((l * TRANSFORMERS_PER_LAYER + t) * BITS_PER_TRANSFORMER), next_state + (t * 4));
            }
            for(int k=0; k<STATE_SIZE; ++k) state[k] = next_state[k];
            if (l < LAYERS - 1) {
                transformer.perfect_shuffle(state, next_state);
                for(int k=0; k<STATE_SIZE; ++k) state[k] = next_state[k];
            }
        }

        int bin_scores[10] = {0};
        for (int k = 0; k < 40; ++k) {
            if (state[k*20 + 0]) bin_scores[0]++; if (state[k*20 + 2]) bin_scores[1]++;
            if (state[k*20 + 4]) bin_scores[2]++; if (state[k*20 + 6]) bin_scores[3]++;
            if (state[k*20 + 8]) bin_scores[4]++; if (state[k*20 + 10]) bin_scores[5]++;
            if (state[k*20 + 12]) bin_scores[6]++; if (state[k*20 + 14]) bin_scores[7]++;
            if (state[k*20 + 16]) bin_scores[8]++; if (state[k*20 + 18]) bin_scores[9]++;
        }

        int max_score = -1, winner_count = 0; int winners[10];
        for(int k=0; k<10; ++k) {
             if(bin_scores[k] > max_score) { max_score = bin_scores[k]; winner_count = 1; winners[0] = k; }
             else if (bin_scores[k] == max_score) { winners[winner_count++] = k; }
        }
        int predicted_digit = winners[hiprand(&rng_state) % winner_count];

        atomicAdd(&confusion_matrix_flat[label * 10 + predicted_digit], 1);
    }
}


// --- HIP_CHECK macro ---
#define HIP_CHECK(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s (%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// --- HOST FUNCTION 1: init_gpu ---
void init_gpu(
    const std::vector<uint8_t>& flat_images,
    const std::vector<uint8_t>& labels,
    const std::vector<uint8_t>& flat_validation_images,
    const std::vector<uint8_t>& validation_labels)
{
    // Copy permutation tables
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_p_w2), PERMUTATIONS_W2_HOST, sizeof(PERMUTATIONS_W2_HOST)));
    const uint8_t PERMUTATIONS_W1_W3_HOST[24][4] = {
        {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
        {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
        {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
        {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}
    };
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_p_w1_w3), PERMUTATIONS_W1_W3_HOST, sizeof(PERMUTATIONS_W1_W3_HOST)));
    uint8_t states_w1[]={1,2,4,8}, states_w2[]={3,5,6,9,10,12}, states_w3[]={7,11,13,14};
    uint8_t rev_w1[16]={0}, rev_w2[16]={0}, rev_w3[16]={0};
    for(int i=0;i<4;++i)rev_w1[states_w1[i]]=i; for(int i=0;i<6;++i)rev_w2[states_w2[i]]=i; for(int i=0;i<4;++i)rev_w3[states_w3[i]]=i;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_rev_w1), rev_w1, sizeof(rev_w1)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_rev_w2), rev_w2, sizeof(rev_w2)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_rev_w3), rev_w3, sizeof(rev_w3)));

    // Free old data
    if (d_global_images) HIP_CHECK(hipFree(d_global_images));
    if (d_global_labels) HIP_CHECK(hipFree(d_global_labels));
    if (d_global_validation_images) HIP_CHECK(hipFree(d_global_validation_images));
    if (d_global_validation_labels) HIP_CHECK(hipFree(d_global_validation_labels));

    // Malloc and Memcpy EVOLUTION data
    g_num_images = labels.size();
    HIP_CHECK(hipMalloc(&d_global_images, flat_images.size()));
    HIP_CHECK(hipMalloc(&d_global_labels, labels.size()));
    HIP_CHECK(hipMemcpy(d_global_images, flat_images.data(), flat_images.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_global_labels, labels.data(), labels.size(), hipMemcpyHostToDevice));

    // Malloc and Memcpy VALIDATION data
    g_num_validation_images = validation_labels.size();
    HIP_CHECK(hipMalloc(&d_global_validation_images, flat_validation_images.size()));
    HIP_CHECK(hipMalloc(&d_global_validation_labels, validation_labels.size()));
    HIP_CHECK(hipMemcpy(d_global_validation_images, flat_validation_images.data(), flat_validation_images.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_global_validation_labels, validation_labels.data(), validation_labels.size(), hipMemcpyHostToDevice));

    std::cout << "--- GPU Initialized with " << g_num_images << " evolution images and "
              << g_num_validation_images << " validation images. ---" << std::endl;
}


// --- HOST FUNCTION 2: evaluate_population_gpu (with weights) ---
std::vector<float> evaluate_population_gpu(
    const uint8_t* flat_population_data,
    int population_size,
    int mini_batch_size,
    const std::vector<float>& host_class_weights) // Accept weights from host
{
    if (d_global_images == nullptr) {
        throw std::runtime_error("GPU not initialized. Call init_gpu() first.");
    }
    uint8_t *d_population;
    float *d_fitness_scores;
    float *d_class_weights = nullptr; // Device pointer for weights

    size_t population_bytes = population_size * CHROMOSOME_LENGTH * sizeof(uint8_t);
    HIP_CHECK(hipMalloc(&d_population, population_bytes));
    HIP_CHECK(hipMalloc(&d_fitness_scores, population_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_population, flat_population_data, population_bytes, hipMemcpyHostToDevice));

    // Allocate and copy class weights to GPU if provided
    bool use_weights = !host_class_weights.empty();
    if (use_weights) {
        if (host_class_weights.size() != 10) {
             throw std::runtime_error("Class weights vector must have size 10.");
        }
        size_t weights_bytes = 10 * sizeof(float);
        HIP_CHECK(hipMalloc(&d_class_weights, weights_bytes));
        HIP_CHECK(hipMemcpy(d_class_weights, host_class_weights.data(), weights_bytes, hipMemcpyHostToDevice));
    }

    int threads_per_block = 256;
    int blocks_per_grid = population_size;
    unsigned long long seed = time(0);
    // Increased shared memory for weights array
    size_t shared_mem_size = threads_per_block * (sizeof(int) + sizeof(float));

    // Launch kernel, passing the device pointer for weights (or nullptr)
    hipLaunchKernelGGL(evaluate_hybrid_kernel, dim3(blocks_per_grid), dim3(threads_per_block), shared_mem_size, 0,
        d_population, d_global_images, d_global_labels,
        d_fitness_scores, population_size, g_num_images, mini_batch_size, seed,
        d_class_weights); // Pass weights pointer

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> host_fitness_scores(population_size);
    HIP_CHECK(hipMemcpy(host_fitness_scores.data(), d_fitness_scores, population_size * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_population));
    HIP_CHECK(hipFree(d_fitness_scores));
    if (d_class_weights) { // Free weights memory if it was allocated
        HIP_CHECK(hipFree(d_class_weights));
    }

    return host_fitness_scores;
}


// --- HOST FUNCTION 3: evaluate_champion_gpu (RESTORED FULL BODY) ---
std::vector<int> evaluate_champion_gpu(const std::vector<uint8_t>& champion_chromosome)
{
    if (d_global_validation_images == nullptr) {
        throw std::runtime_error("GPU not initialized. Call init_gpu() first.");
    }

    uint8_t* d_champion_chromosome;
    int* d_confusion_matrix;

    HIP_CHECK(hipMalloc(&d_champion_chromosome, champion_chromosome.size()));
    HIP_CHECK(hipMemcpy(d_champion_chromosome, champion_chromosome.data(), champion_chromosome.size(), hipMemcpyHostToDevice));

    size_t matrix_bytes = 10 * 10 * sizeof(int);
    HIP_CHECK(hipMalloc(&d_confusion_matrix, matrix_bytes));
    HIP_CHECK(hipMemset(d_confusion_matrix, 0, matrix_bytes));

    int threads_per_block = 256;
    int blocks_per_grid = 64; // Saturate the GPU

    hipLaunchKernelGGL(evaluate_champion_kernel, dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
        d_champion_chromosome,
        d_global_validation_images,
        d_global_validation_labels,
        g_num_validation_images,
        d_confusion_matrix
    );

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> host_confusion_matrix(100);
    HIP_CHECK(hipMemcpy(host_confusion_matrix.data(), d_confusion_matrix, matrix_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_champion_chromosome));
    HIP_CHECK(hipFree(d_confusion_matrix));

    // --- Return statement fixes the warning ---
    return host_confusion_matrix;
}
