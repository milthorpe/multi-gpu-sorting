#pragma once

#include <thrust/host_vector.h>

#include <thrust/system/cuda/memory_resource.h>
using mr = thrust::system::cuda::universal_host_pinned_memory_resource;
template <typename T>
using HostVector = thrust::mr::stateless_resource_allocator<T, mr >;
