# neat-lib

A C++ library implementing the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. This repository contains the source code, test suites, and benchmarking utilities developed for a bachelor's thesis in computer engineering.

## Project Structure

- `src/`: Core library implementation files.
- `include/neat/`: Public header files.
- `envs/`: Evaluation environments for the neural networks.
- `tests/`: Unit tests and validation code.
- `benchmarks/`: Performance measurement utilities.

## Prerequisites

- C++ Compiler
- CMake 

## Build Instructions

1. Clone the repository:
   ```bash
   git clone [https://github.com/SnakiestCarrot/neat-lib.git](https://github.com/SnakiestCarrot/neat-lib.git)
   ```
2. Navigate to the project root and create a build directory:
   ```bash
   cd neat-lib
   mkdir build
   cd build
   ```
3. Generate the build files and compile:
   ```bash
   cmake ..
   make
   ```

## Cleaning the Build

To remove compiled binaries and intermediate object files, navigate to your `build` directory and run:
```bash
make clean
```

For a complete clean (useful if you change CMake configurations), you can safely remove and recreate the entire build directory from the project root:
```bash
rm -rf build
mkdir build
```

## Running Tests and Benchmarks (TODO)

After building the project, you can run the test or benchmark suite to verify the library's functionality and performance:
```bash
./tests/neat_tests
```

And:
```bash
./benchmarks/neat_benchmarks
```
