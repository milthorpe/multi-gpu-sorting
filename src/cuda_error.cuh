#pragma once

#include <iostream>

#ifdef DEBUG
#define CheckCudaError(instruction) \
  { AssertNoCudaError((instruction), __FILE__, __LINE__); }
#else
#define CheckCudaError(instruction) instruction
#endif

inline void AssertNoCudaError(hipError_t error_code, const char* file, int line) {
  if (error_code != hipSuccess) {
    std::cout << "Error: " << hipGetErrorString(error_code) << " " << file << " " << line << "\n";

    exit(error_code);
  }
}
