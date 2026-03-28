# 1. Automatically fetch GoogleTest from GitHub
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include the GoogleTest CMake module for test discovery
include(GoogleTest)

# 2. Config Tests
add_executable(test_config test_config.cpp)
# Link our library AND gtest_main
target_link_libraries(test_config PRIVATE neat_core GTest::gtest_main)
gtest_discover_tests(test_config DISCOVERY_MODE PRE_TEST)

# 3. Random Determinism Tests
add_executable(test_random test_random.cpp)
target_link_libraries(test_random PRIVATE neat_core GTest::gtest_main)
gtest_discover_tests(test_random DISCOVERY_MODE PRE_TEST)

# 4. Network Tests
add_executable(test_network test_network.cpp)
target_link_libraries(test_network PRIVATE neat_core GTest::gtest_main)
gtest_discover_tests(test_network DISCOVERY_MODE PRE_TEST)

# 5. Innovation Tracker Tests
add_executable(test_innovation test_innovation.cpp)
target_link_libraries(test_innovation PRIVATE neat_core GTest::gtest_main)
gtest_discover_tests(test_innovation DISCOVERY_MODE PRE_TEST)

# 6. Genome Tests
add_executable(test_genome test_genome.cpp)
target_link_libraries(test_genome PRIVATE neat_core GTest::gtest_main)
gtest_discover_tests(test_genome DISCOVERY_MODE PRE_TEST)