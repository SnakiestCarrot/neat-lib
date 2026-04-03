cmake_minimum_required(VERSION 3.20)
project(NEAT_HighPerf VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Core NEAT Library Target
# -----------------------------------------------------------------------------
# Add all your .cpp files here
add_library(neat_core
    src/genome.cpp
    src/innovation.cpp
    src/network.cpp
    src/population.cpp
    src/random.cpp
    src/config.cpp
)


# Tell CMake where the public headers are
target_include_directories(neat_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# -----------------------------------------------------------------------------
# Optional: Intel TBB for parallel STL (std::execution::par_unseq)
# -----------------------------------------------------------------------------
find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "TBB found — parallel evaluation enabled")
    target_link_libraries(neat_core PRIVATE TBB::tbb)
    target_compile_definitions(neat_core PUBLIC NEAT_PARALLEL_EVAL=1)
else()
    message(STATUS "TBB not found — parallel_eval will fall back to sequential")
endif()

# -----------------------------------------------------------------------------
# High-Performance Compiler Flags
# -----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(neat_core PRIVATE
        -Wall -Wextra -Wpedantic # Strict warnings for memory safety
        -O3                      # Maximum optimization
        -march=native            # Optimize for the specific CPU architecture (vectorization)
    )
    
    # Note on Determinism: 
    # Do NOT use -ffast-math here. While it speeds up execution, it breaks 
    # IEEE 754 floating-point compliance, which will ruin the bit-for-bit 
    # determinism required for your reproducibility goals.
endif()

# -----------------------------------------------------------------------------
# Executables (Benchmark Environments)
# -----------------------------------------------------------------------------
# Snake Environment
add_executable(neat_snake envs/snake/main.cpp)
target_link_libraries(neat_snake PRIVATE neat_core)

# Double Pendulum Environment
add_executable(neat_pendulum envs/pendulum/main.cpp)
target_link_libraries(neat_pendulum PRIVATE neat_core)

add_subdirectory(envs/cartpole)

# Testing
enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
