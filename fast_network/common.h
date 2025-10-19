#ifndef COMMON_H
#define COMMON_H

#include <cstdint> // Needed by both C++ and HIP

// --- Architectural Constants ---
// ... (unchanged) ...
constexpr int STATE_SIZE = 800;
constexpr int IMAGE_SIZE = 784;
constexpr int LAYERS = 10;
constexpr int TRANSFORMERS_PER_LAYER = STATE_SIZE / 4;
constexpr int BITS_PER_TRANSFORMER = 20;
constexpr int CHROMOSOME_LENGTH = TRANSFORMERS_PER_LAYER * LAYERS * BITS_PER_TRANSFORMER;


// --- HIP-SPECIFIC CODE ---
#ifdef __HIP__

#include <hip/hip_runtime.h>

// --- (Constant tables are unchanged) ---
__constant__ uint8_t d_p_w1_w3[24][4];
__constant__ uint8_t d_p_w2[720][6];
__constant__ uint8_t d_rev_w1[16];
__constant__ uint8_t d_rev_w2[16];
__constant__ uint8_t d_rev_w3[16];

// --- GPU-Optimized Transformer Class (unchanged) ---
class Transformer {
public:
    // --- (bits_to_int functions are unchanged) ---
    __device__ __forceinline__ unsigned int bits_to_int_4(const uint8_t* bits) const {
        return (bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3];
    }
    __device__ __forceinline__ unsigned int bits_to_int_5(const uint8_t* bits) const {
        return (bits[0] << 4) | (bits[1] << 3) | (bits[2] << 2) | (bits[3] << 1) | bits[4];
    }
    __device__ __forceinline__ unsigned int bits_to_int_10(const uint8_t* bits) const {
        return (bits[0] << 9) | (bits[1] << 8) | (bits[2] << 7) | (bits[3] << 6) |
               (bits[4] << 5) | (bits[5] << 4) | (bits[6] << 3) | (bits[7] << 2) |
               (bits[8] << 1) | bits[9];
    }

    __device__ void process(const uint8_t* input_4_bits, const uint8_t* control_20_bits, uint8_t* output_4_bits) const {
        unsigned int input_int = bits_to_int_4(input_4_bits);
        
        // --- OPTIMIZATION: Unroll weight calculation ---
        int weight = input_4_bits[0] + input_4_bits[1] + input_4_bits[2] + input_4_bits[3];

        if (weight == 0 || weight == 4) {
            for(int i=0; i<4; ++i) output_4_bits[i] = input_4_bits[i];
            return;
        }

        // --- (Rest of function is unchanged) ---
        unsigned int perm_index;
        int input_idx;
        const uint8_t* permutation;
        unsigned int output_int;
        uint8_t states_w1[] = {1, 2, 4, 8};
        uint8_t states_w2[] = {3, 5, 6, 9, 10, 12};
        uint8_t states_w3[] = {7, 11, 13, 14};

        if (weight == 1) {
            perm_index = bits_to_int_5(control_20_bits) % 24;
            input_idx = d_rev_w1[input_int];
            permutation = d_p_w1_w3[perm_index];
            output_int = states_w1[permutation[input_idx]];
        } else if (weight == 2) {
            perm_index = bits_to_int_10(control_20_bits + 5) % 720;
            input_idx = d_rev_w2[input_int];
            permutation = d_p_w2[perm_index];
            output_int = states_w2[permutation[input_idx]];
        } else {
            perm_index = bits_to_int_5(control_20_bits + 15) % 24;
            input_idx = d_rev_w3[input_int];
            permutation = d_p_w1_w3[perm_index];
            output_int = states_w3[permutation[input_idx]];
        }
        
        for(int i=0; i<4; ++i) { output_4_bits[i] = (output_int >> (3-i)) & 1; }
    }

    // --- (perfect_shuffle is unchanged) ---
    __device__ void perfect_shuffle(const uint8_t* state_in, uint8_t* state_out) const {
        int mid = STATE_SIZE / 2;
        for (int i = 0; i < mid; ++i) {
            state_out[i * 2] = state_in[i];
            state_out[i * 2 + 1] = state_in[mid + i];
        }
    }
};

#endif // __HIP__

#endif // COMMON_H
