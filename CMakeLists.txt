cmake_minimum_required(VERSION 3.21)
project(fast_network_gpu CXX HIP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Ofast")

find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)
# ... (hiprand find logic unchanged) ...
find_path(HIPRAND_INCLUDE_DIR hiprand/hiprand.h PATHS /opt/rocm/include NO_DEFAULT_PATH)
find_library(HIPRAND_LIBRARY hiprand PATHS /opt/rocm/lib NO_DEFAULT_PATH)
if(HIPRAND_INCLUDE_DIR AND HIPRAND_LIBRARY)
    add_library(hiprand::hiprand UNKNOWN IMPORTED)
    set_target_properties(hiprand::hiprand PROPERTIES
        IMPORTED_LOCATION "${HIPRAND_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HIPRAND_INCLUDE_DIR}"
    )
    message(STATUS "Found hiprand library: ${HIPRAND_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find the hiprand library or headers in /opt/rocm/")
endif()




#include(CheckIPOSupported)
#check_ipo_supported(RESULT LTO_SUPPORTED)
#if(LTO_SUPPORTED)
#	message(STATUS "Link Time Optimization (LTO) is supported. Enabling...")
#    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
#else()
#   message(WARNING "Link Time Optimization (LTO) is not supported by this compiler.")
#endif()





# --- STAGE 1: Static HIP Library (Unchanged) ---
add_library(network_core STATIC fast_network/fast_network_gpu.hip.cpp)
set_source_files_properties(fast_network/fast_network_gpu.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(network_core PROPERTIES
    HIP_ARCHITECTURES "gfx900"
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(network_core PRIVATE hiprand::hiprand)

# --- STAGE 2: Python Module (ADD SOURCE FILE) ---
add_library(fast_network_gpu MODULE
    fast_network/bindings_gpu.cpp
    fast_network/genetic_operators.cpp # <-- Add this line
)
pybind11_extension(fast_network_gpu) # Applies Python/pybind11 includes to this target

# --- STAGE 3: Linking (Unchanged) ---
target_link_libraries(fast_network_gpu PRIVATE
    network_core
    pybind11::module # Provides Python includes to fast_network_gpu target
    hiprand::hiprand
    OpenMP::OpenMP_CXX
)

# --- Install (Unchanged) ---
install(TARGETS fast_network_gpu DESTINATION fast_network)
