#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "common.h"
#include <vector>
#include <cstdint>
#include "genetic_operators.h" // Include the new header

namespace py = pybind11;

// --- Forward declarations ---
void init_gpu(const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<uint8_t>&);
std::vector<float> evaluate_population_gpu(
    const uint8_t* flat_population_data,
    int population_size,
    int mini_batch_size,
    const std::vector<float>& host_class_weights); // <-- Added weights
std::vector<int> evaluate_champion_gpu(const std::vector<uint8_t>&);

// --- Wrapper for evaluate_population_gpu ---
std::vector<float> evaluate_population_gpu_wrapper(
    np_array_uint8 population_array, // Use alias
    int mini_batch_size,
    py::object class_weights_obj) // <-- Accept Python object (list, NumPy array, or None)
{
    py::buffer_info buf = population_array.request();
    const uint8_t* flat_population_data = static_cast<uint8_t*>(buf.ptr);
    int population_size = buf.shape[0];

    // --- Convert Python weights object to std::vector<float> ---
    std::vector<float> host_class_weights;
    if (!class_weights_obj.is_none()) {
        try {
            // Attempt conversion from list or NumPy array
             host_class_weights = class_weights_obj.cast<std::vector<float>>();
             if (host_class_weights.size() != 10) {
                 throw py::value_error("Class weights must be a list/array of size 10 or None.");
             }
        } catch (const py::cast_error& e) {
             throw py::type_error("Class weights must be a list/array of size 10 or None.");
        }
    }
    // If weights_obj was None, host_class_weights remains empty, which is handled by C++

    // Call the real C++ function, passing the (possibly empty) weights vector
    return evaluate_population_gpu(
        flat_population_data,
        population_size,
        mini_batch_size,
        host_class_weights);
}


// --- pybind11 Module Definition ---
PYBIND11_MODULE(fast_network_gpu, m) {
    m.doc() = "GPU-accelerated reversible network simulator";

    // init_gpu binding
    m.def("init_gpu", &init_gpu,
          "Initializes the GPU with constant image and label data",
          py::arg("flat_images"), py::arg("labels"),
          py::arg("flat_validation_images"), py::arg("validation_labels"));

    // evaluate_population_gpu binding (using wrapper)
    m.def("evaluate_population_gpu", &evaluate_population_gpu_wrapper,
          "Evaluates the entire population's fitness on the GPU, optionally using class weights.",
          py::arg("population"),
          py::arg("mini_batch_size"),
          py::arg("class_weights") = py::none()); // Add weights arg, default to None

    // evaluate_champion_gpu binding
    m.def("evaluate_champion_gpu", &evaluate_champion_gpu,
          "Evaluates a single chromosome on the entire validation set",
          py::arg("champion_chromosome"));

    // breed_new_generation binding
    m.def("breed_new_generation", &breed_new_generation,
          "Creates the next generation using CPU parallelism (OpenMP)",
          py::arg("population"),
          py::arg("fitness_scores"),
          py::arg("current_champion"),
          py::arg("elitism_count"),
          py::arg("tournament_size"),
          py::arg("mutation_rate"),
          py::arg("num_transformers"),
          py::arg("bits_per_transformer"),
          py::arg("chromosome_length")); // Corrected definition

    // Exposed constants
    m.attr("CHROMOSOME_LENGTH") = CHROMOSOME_LENGTH;
    m.attr("IMAGE_SIZE") = IMAGE_SIZE;
    m.attr("BITS_PER_TRANSFORMER") = BITS_PER_TRANSFORMER;
    m.attr("NUM_TRANSFORMERS") = CHROMOSOME_LENGTH / BITS_PER_TRANSFORMER;
}
