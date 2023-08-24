#include "hip/hip_runtime.h"
#pragma once
#ifndef _TANASIC_SORT
#define _TANASIC_SORT

#include <omp.h>
#include <thrust/merge.h>
#include <thrust/sort.h>

#include <array>
#include <iostream>
#include <vector>

#include "device_buffers.cuh"
#include "host_vector.cuh"
#include "time_durations.cuh"

template <typename T>
inline __device__ void GetValueFromVirtualPartition(size_t partition_size, T** virtual_partition, size_t index,
                                                    T* value) {
  *value = virtual_partition[index / partition_size][index % partition_size];
}

template <typename T>
__global__ void SelectPivot(size_t partition_size, size_t num_partitions, T** local_virtual_partition,
                            T** remote_virtual_partition, size_t* result_pivot) {
  size_t low = 0;
  size_t high = partition_size * num_partitions;

  while (low < high) {
    const size_t mid = high - (high - low) / 2;

    T a;
    GetValueFromVirtualPartition<T>(partition_size, local_virtual_partition, partition_size * num_partitions - mid, &a);
    T b;
    GetValueFromVirtualPartition<T>(partition_size, remote_virtual_partition, mid - 1, &b);

    if (a <= b) {
      high = mid - 1;
    } else {
      low = mid;
    }
  }

  *result_pivot = low;
}

template <typename T>
size_t FindPivot(DeviceBuffers<T>* device_buffers, const std::vector<int>& devices) {
  CheckCudaError(hipSetDevice(devices[0]));

  const size_t num_partitions = devices.size() / 2;

  std::vector<T*> local_partitions(num_partitions);
  std::vector<T*> remote_partitions(num_partitions);

  for (size_t i = 0; i < num_partitions; ++i) {
    local_partitions[i] = thrust::raw_pointer_cast(device_buffers->AtPrimary(devices[i])->data());
    remote_partitions[i] = thrust::raw_pointer_cast(device_buffers->AtPrimary(devices[i + num_partitions])->data());
  }

  T** local_virtual_partition =
      reinterpret_cast<T**>(device_buffers->GetDeviceAllocator(devices[0])->allocate(sizeof(T*) * num_partitions));
  T** remote_virtual_partition =
      reinterpret_cast<T**>(device_buffers->GetDeviceAllocator(devices[0])->allocate(sizeof(T*) * num_partitions));

  CheckCudaError(hipMemcpy(local_virtual_partition, local_partitions.data(), sizeof(T*) * num_partitions,
                            hipMemcpyHostToDevice));

  CheckCudaError(hipMemcpy(remote_virtual_partition, remote_partitions.data(), sizeof(T*) * num_partitions,
                            hipMemcpyHostToDevice));

  size_t* pivot = reinterpret_cast<size_t*>(device_buffers->GetHostAllocator(devices[0])->allocate(sizeof(size_t)));

  SelectPivot<T><<<1, 1>>>(device_buffers->GetPartitionSize(), num_partitions, local_virtual_partition,
                           remote_virtual_partition, pivot);

  CheckCudaError(hipDeviceSynchronize());

  size_t result_pivot = *pivot;

  device_buffers->GetDeviceAllocator(devices[0])
      ->deallocate(reinterpret_cast<uint8_t*>(local_virtual_partition), sizeof(T*) * num_partitions);
  device_buffers->GetDeviceAllocator(devices[0])
      ->deallocate(reinterpret_cast<uint8_t*>(remote_virtual_partition), sizeof(T*) * num_partitions);

  device_buffers->GetHostAllocator(devices[0])->deallocate(reinterpret_cast<uint8_t*>(pivot), sizeof(size_t));

  return result_pivot;
}

template <typename T>
std::array<int, 2> SwapPartitions(DeviceBuffers<T>* device_buffers, size_t pivot, const std::vector<int>& devices) {
  std::array<int, 2> devices_to_merge;

  const size_t partition_size = device_buffers->GetPartitionSize();
  size_t devices_to_swap = pivot / partition_size;

  if (pivot == partition_size * (devices.size() / 2)) {
    --devices_to_swap;
    pivot = partition_size;
  } else {
    pivot %= partition_size;
  }

#pragma omp parallel for
  for (size_t i = 0; i <= devices_to_swap; ++i) {
    const int left_device = devices[devices.size() / 2 - i - 1];
    const int right_device = devices[devices.size() / 2 + i];
    const size_t num_elements = (i == devices_to_swap) ? pivot : partition_size;

    CheckCudaError(hipSetDevice(left_device));

    CheckCudaError(hipMemcpyAsync(
        thrust::raw_pointer_cast(device_buffers->AtSecondary(left_device)->data() + partition_size - num_elements),
        thrust::raw_pointer_cast(device_buffers->AtPrimary(right_device)->data()), sizeof(T) * num_elements,
        hipMemcpyDeviceToDevice, *device_buffers->GetPrimaryStream(left_device)));

    CheckCudaError(hipSetDevice(right_device));

    CheckCudaError(hipMemcpyAsync(
        thrust::raw_pointer_cast(device_buffers->AtSecondary(right_device)->data()),
        thrust::raw_pointer_cast(device_buffers->AtPrimary(left_device)->data() + partition_size - num_elements),
        sizeof(T) * num_elements, hipMemcpyDeviceToDevice, *device_buffers->GetPrimaryStream(right_device)));

    if (i == devices_to_swap) {
      devices_to_merge[0] = left_device;
      devices_to_merge[1] = right_device;

#pragma omp parallel for
      for (size_t j = 0; j < devices_to_merge.size(); ++j) {
        CheckCudaError(hipSetDevice(devices_to_merge[j]));

        CheckCudaError(
            hipMemcpy(thrust::raw_pointer_cast(device_buffers->AtSecondary(devices_to_merge[j])->data() + (j * pivot)),
                       thrust::raw_pointer_cast(device_buffers->AtPrimary(devices_to_merge[j])->data() + (j * pivot)),
                       sizeof(T) * (partition_size - pivot), hipMemcpyDeviceToDevice));
      }
    }

    CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(left_device)));
    CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(right_device)));

    device_buffers->Flip(left_device);
    device_buffers->Flip(right_device);
  }

  return devices_to_merge;
}

template <typename T>
void MergeLocalPartitions(DeviceBuffers<T>* device_buffers, size_t pivot,
                          const std::array<int, 2>& devices_to_merge, const std::vector<int>& devices,
                          size_t num_fillers) {
  const size_t partition_size = device_buffers->GetPartitionSize();
  pivot %= partition_size;

#pragma omp parallel for
  for (size_t i = 0; i < devices.size(); ++i) {
    const int device = devices[i];
    const size_t offset = i >= devices.size() / 2 ? pivot : partition_size - pivot;

    CheckCudaError(hipSetDevice(device));

    if (device == devices_to_merge[0] || device == devices_to_merge[1]) {
      thrust::merge(
          thrust::hip::par(*device_buffers->GetDeviceAllocator(device)).on(*device_buffers->GetPrimaryStream(device)),
          device_buffers->AtPrimary(device)->begin(), device_buffers->AtPrimary(device)->begin() + offset,
          device_buffers->AtPrimary(device)->begin() + offset, device_buffers->AtPrimary(device)->end(),
          device_buffers->AtSecondary(device)->begin());

      device_buffers->Flip(device);

      CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(device)));
    }
  }
}

template <typename T>
void MergePartitions(DeviceBuffers<T>* device_buffers, const std::vector<int>& devices,
                     size_t num_fillers) {
  if (devices.size() > 2) {
#pragma omp parallel for
    for (size_t i = 0; i < 2; ++i) {
      MergePartitions(
          device_buffers,
          {devices.begin() + (i * (devices.size() / 2)), devices.begin() + ((i + 1) * (devices.size() / 2))},
          num_fillers);
    }
  }

  const size_t pivot = FindPivot<T>(device_buffers, devices);
  if (pivot > 0) {
    const std::array<int, 2> devices_to_merge = SwapPartitions<T>(device_buffers, pivot, devices);
    MergeLocalPartitions<T>(device_buffers, pivot, devices_to_merge, devices, num_fillers);
  }

  if (devices.size() > 2) {
#pragma omp parallel for
    for (size_t i = 0; i < 2; ++i) {
      MergePartitions(
          device_buffers,
          {devices.begin() + (i * (devices.size() / 2)), devices.begin() + ((i + 1) * (devices.size() / 2))},
          num_fillers);
    }
  }
}

void SetupTanasicSort(const std::vector<int>& gpus) {
  omp_set_schedule(omp_sched_static, 1);
  omp_set_nested(1);
  omp_set_num_threads(gpus.size());

#pragma omp parallel for
  for (size_t i = 0; i < gpus.size(); ++i) {
    CheckCudaError(hipSetDevice(gpus[i]));

    for (size_t j = 0; j < gpus.size(); ++j) {
      if (i != j) {
        CheckCudaError(hipDeviceEnablePeerAccess(gpus[j], 0));
      }
    }
  }
}

template <typename T>
void TanasicSort(HostVector<T>* elements, std::vector<int> gpus) {
  if (elements->empty()) {
    return;
  }

  TimeDurations::Get()->Tic("memory_allocation");

  size_t num_fillers = (elements->size() % gpus.size() != 0) ? (gpus.size() - elements->size() % gpus.size()) : 0;
  size_t buffer_size = (elements->size() + num_fillers) / gpus.size();

  while (buffer_size < num_fillers) {
    gpus.resize(gpus.size() / 2);
    num_fillers = (elements->size() % gpus.size() != 0) ? (gpus.size() - elements->size() % gpus.size()) : 0;
    buffer_size = (elements->size() + num_fillers) / gpus.size();
  }

  DeviceBuffers<T>* device_buffers = new DeviceBuffers<T>(gpus, buffer_size, 2);

  TimeDurations::Get()->Toc("memory_allocation");

  TimeDurations::Get()->Tic("sort_phase");

#pragma omp parallel for
  for (size_t i = 0; i < gpus.size(); ++i) {
    CheckCudaError(hipSetDevice(gpus[i]));

    CheckCudaError(hipMemcpyAsync(thrust::raw_pointer_cast(device_buffers->AtPrimary(gpus[i])->data()),
                                   thrust::raw_pointer_cast(elements->data() + (buffer_size * i)),
                                   (sizeof(T) * buffer_size) - (i == gpus.size() - 1 ? sizeof(T) * num_fillers : 0),
                                   hipMemcpyHostToDevice, *device_buffers->GetPrimaryStream(gpus[i])));

    if (i == gpus.size() - 1) {
      thrust::fill(thrust::hip::par(*device_buffers->GetDeviceAllocator(gpus[i]))
                       .on(*device_buffers->GetPrimaryStream(gpus[i])),
                   device_buffers->AtPrimary(gpus[i])->end() - num_fillers, device_buffers->AtPrimary(gpus[i])->end(),
                   std::numeric_limits<T>::max());
    }

    CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(gpus[i])));

    thrust::sort(
        thrust::hip::par(*device_buffers->GetDeviceAllocator(gpus[i])).on(*device_buffers->GetPrimaryStream(gpus[i])),
        device_buffers->AtPrimary(gpus[i])->begin(), device_buffers->AtPrimary(gpus[i])->end());

    CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(gpus[i])));
  }

#pragma omp parallel for
  for (size_t i = 0; i < gpus.size(); ++i) {
    device_buffers->GetDeviceAllocator(gpus[i])->SetOffset(buffer_size * sizeof(T));
  }

  TimeDurations::Get()->Tic("merge_phase");
  if (gpus.size() > 1) {
    MergePartitions(device_buffers, gpus, num_fillers);
  }
  TimeDurations::Get()->Toc("merge_phase");

#pragma omp parallel for
  for (size_t i = 0; i < gpus.size(); ++i) {
    CheckCudaError(hipSetDevice(gpus[i]));

    CheckCudaError(hipMemcpyAsync(thrust::raw_pointer_cast(elements->data() + (buffer_size * i)),
                                   thrust::raw_pointer_cast(device_buffers->AtPrimary(gpus[i])->data()),
                                   (sizeof(T) * buffer_size) - (i == gpus.size() - 1 ? sizeof(T) * num_fillers : 0),
                                   hipMemcpyDeviceToHost, *device_buffers->GetPrimaryStream(gpus[i])));

    CheckCudaError(hipStreamSynchronize(*device_buffers->GetPrimaryStream(gpus[i])));
  }

  TimeDurations::Get()->Toc("sort_phase");

  TimeDurations::Get()->Tic("memory_deallocation");
  delete device_buffers;
  TimeDurations::Get()->Toc("memory_deallocation");
}

#endif // _TANASIC_SORT
