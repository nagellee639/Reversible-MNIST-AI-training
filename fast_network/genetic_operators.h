#ifndef GENETIC_OPERATORS_H
#define GENETIC_OPERATORS_H

#include <vector>
#include <cstdint>
#include <random>

// Define NumPy array type alias for convenience
namespace py = pybind11;
using np_array_uint8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using np_array_float = py::array_t<float, py::array::c_style | py::array::forcecast>;


// Function declarations
int tournament_selection(
    const float* fitness_scores, 
    int population_size, 
    int tournament_size, 
    std::mt19937& rng);

void block_crossover(
    const uint8_t* parent1, 
    const uint8_t* parent2, 
    uint8_t* child, 
    int chromosome_length, 
    int num_transformers, 
    int bits_per_transformer, 
    std::mt19937& rng);

void block_mutate(
    uint8_t* chromosome, 
    int chromosome_length, 
    int num_transformers, 
    int bits_per_transformer, 
    float mutation_rate, 
    std::mt19937& rng);

// The main function that orchestrates breeding using OpenMP
np_array_uint8 breed_new_generation(
    np_array_uint8 population_np, 
    np_array_float fitness_scores_np, 
    np_array_uint8 current_champion_np, 
    int elitism_count, 
    int tournament_size, 
    float mutation_rate,
    int num_transformers, 
    int bits_per_transformer,
    int chromosome_length);

#endif // GENETIC_OPERATORS_H
