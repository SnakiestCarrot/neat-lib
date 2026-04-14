include(FetchContent)

# Fetch Google Benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
# Disable benchmark's own internal tests for speed
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# NOTE: Use "--benchmark_filter=" to run specific benchmarks

# Gather all benchmark source files
file(GLOB BENCHMARK_SOURCES "benchmark_*.cpp")

# Create the executable
add_executable(run_benchmarks ${BENCHMARK_SOURCES})

# Link the library and the benchmark_main target 
# (benchmark_main provides the main() function automatically)
target_link_libraries(run_benchmarks PRIVATE neat_core benchmark::benchmark_main)
