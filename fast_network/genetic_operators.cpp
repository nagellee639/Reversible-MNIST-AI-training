#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h> // Include OpenMP
#include <chrono> // For seeding RNG

#include "genetic_operators.h" // Include the header we just defined

namespace py = pybind11;

// --- Tournament Selection ---
// Selects the index of a winning parent
int tournament_selection(
    const float* fitness_scores, 
    int population_size, 
    int tournament_size, 
    std::mt19937& rng) 
{
    std::uniform_int_distribution<int> dist(0, population_size - 1);
    
    int best_index = -1;
    float best_fitness = -1.0f;

    for (int i = 0; i < tournament_size; ++i) {
        int idx = dist(rng);
        if (fitness_scores[idx] > best_fitness) {
            best_fitness = fitness_scores[idx];
            best_index = idx;
        }
    }
    return best_index;
}

// --- Block Crossover ---
void block_crossover(
    const uint8_t* parent1, 
    const uint8_t* parent2, 
    uint8_t* child, 
    int chromosome_length, 
    int num_transformers, 
    int bits_per_transformer, 
    std::mt19937& rng) 
{
    std::uniform_int_distribution<int> dist(0, 1);
    for (int i = 0; i < num_transformers; ++i) {
        const uint8_t* source_parent = (dist(rng) == 0) ? parent1 : parent2;
        int start_idx = i * bits_per_transformer;
        // Use memcpy for efficient block copying
        memcpy(child + start_idx, source_parent + start_idx, bits_per_transformer);
    }
}

// --- Block Mutation ---
void block_mutate(
    uint8_t* chromosome, 
    int chromosome_length, 
    int num_transformers, 
    int bits_per_transformer, 
    float mutation_rate, 
    std::mt19937& rng) 
{
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> bit_dist(0, 1);

    for (int i = 0; i < num_transformers; ++i) {
        if (prob_dist(rng) < mutation_rate) {
            int start_idx = i * bits_per_transformer;
            for (int j = 0; j < bits_per_transformer; ++j) {
                chromosome[start_idx + j] = bit_dist(rng);
            }
        }
    }
}

// --- Main Breeding Function (Parallelized with OpenMP) ---
np_array_uint8 breed_new_generation(
    np_array_uint8 population_np, 
    np_array_float fitness_scores_np, 
    np_array_uint8 current_champion_np, 
    int elitism_count, 
    int tournament_size, 
    float mutation_rate,
    int num_transformers, 
    int bits_per_transformer,
    int chromosome_length)
{
    // Get pointers and sizes from NumPy arrays
    py::buffer_info pop_buf = population_np.request();
    py::buffer_info fit_buf = fitness_scores_np.request();
    py::buffer_info champ_buf = current_champion_np.request();

    uint8_t* population = static_cast<uint8_t*>(pop_buf.ptr);
    float* fitness_scores = static_cast<float*>(fit_buf.ptr);
    uint8_t* current_champion = static_cast<uint8_t*>(champ_buf.ptr);
    int population_size = pop_buf.shape[0];

    // Allocate space for the new population (NumPy handles memory)
    auto new_population_np = np_array_uint8({population_size, chromosome_length});
    py::buffer_info new_pop_buf = new_population_np.request();
    uint8_t* new_population = static_cast<uint8_t*>(new_pop_buf.ptr);

    // --- Elitism ---
    // Find elite indices (simple sort on CPU for small elitism_count)
    std::vector<int> indices(population_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + elitism_count, indices.end(),
                      [&](int a, int b) { return fitness_scores[a] > fitness_scores[b]; });

    // Copy elites
    for (int i = 0; i < elitism_count; ++i) {
        memcpy(new_population + i * chromosome_length, 
               population + indices[i] * chromosome_length, 
               chromosome_length);
    }

    // Ensure champion is included (replace worst elite if needed)
    bool champion_in_elites = false;
    for (int i = 0; i < elitism_count; ++i) {
        if (memcmp(new_population + i * chromosome_length, current_champion, chromosome_length) == 0) {
            champion_in_elites = true;
            break;
        }
    }
    if (!champion_in_elites) {
        memcpy(new_population, current_champion, chromosome_length); // Overwrite index 0 (worst elite)
    }

    // --- Parallel Reproduction using OpenMP ---
    #pragma omp parallel
    {
        // Each thread gets its own random number generator, seeded uniquely
        unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count() + omp_get_thread_num();
        std::mt19937 rng(seed);

        // Each thread handles a portion of the population
        #pragma omp for
        for (int i = elitism_count; i < population_size; ++i) {
            // Select parents
            int p1_idx = tournament_selection(fitness_scores, population_size, tournament_size, rng);
            int p2_idx = tournament_selection(fitness_scores, population_size, tournament_size, rng);
            const uint8_t* parent1 = population + p1_idx * chromosome_length;
            const uint8_t* parent2 = population + p2_idx * chromosome_length;

            // Crossover
            uint8_t* child = new_population + i * chromosome_length; // Pointer to the child's location
            block_crossover(parent1, parent2, child, chromosome_length, num_transformers, bits_per_transformer, rng);

            // Mutate
            block_mutate(child, chromosome_length, num_transformers, bits_per_transformer, mutation_rate, rng);
        }
    } // End of parallel region

    return new_population_np; // Return the new NumPy array
}
