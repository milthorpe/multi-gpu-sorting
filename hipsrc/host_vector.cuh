#pragma once

#include <thrust/host_vector.h>

#ifdef __CUDACC__
#include <thrust/system/cuda/memory_resource.h>
using mr = thrust::system::cuda::universal_host_pinned_memory_resource;
#else
#include <thrust/mr/memory_resource.h>
using mr = thrust::universal_host_pinned_memory_resource;
#endif
template <typename T>
using HostVector = thrust::mr::stateless_resource_allocator<T, mr >;
