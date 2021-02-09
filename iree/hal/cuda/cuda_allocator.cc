// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations ufnder the License.

#include "iree/hal/cuda/cuda_allocator.h"
#include "iree/hal/cuda/cuda_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/cuda/status_util.h"

using namespace iree::hal::cuda;

typedef struct iree_hal_cuda_allocator_s {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  
  iree::hal::cuda::DynamicSymbols* syms;
  CUcontext context;
} iree_hal_cuda_allocator_t;

extern const iree_hal_allocator_vtable_t iree_hal_cuda_allocator_vtable;

static iree_hal_cuda_allocator_t* iree_hal_cuda_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_allocator_vtable);
  return (iree_hal_cuda_allocator_t*)base_value;
}

iree_status_t iree_hal_cuda_allocator_create(
    CUcontext context, iree::hal::cuda::DynamicSymbols* syms,
    iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_allocator_t* allocator = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*allocator), (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_allocator_vtable,
                                 &allocator->resource);
    allocator->context = context;
    allocator->syms = syms;
    allocator->host_allocator = host_allocator;
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_cuda_allocator_t* allocator =
      iree_hal_cuda_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_cuda_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_cuda_allocator_t* allocator =
      (iree_hal_cuda_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_hal_buffer_compatibility_t
iree_hal_cuda_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  // TODO(benvanik): check to ensure the allocator can serve the memory type.

  // Disallow usage not permitted by the buffer itself. Since we then use this
  // to determine compatibility below we'll naturally set the right compat flags
  // based on what's both allowed and intended.
  intended_usage &= allowed_usage;

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  return compatibility;
}

static iree_status_t iree_hal_cuda_allocator_make_compatible(
    iree_hal_memory_type_t* memory_type,
    iree_hal_memory_access_t* allowed_access,
    iree_hal_buffer_usage_t* allowed_usage) {
  // TODO(benvanik): remove this entirely!
  // Host currently uses mapping to copy buffers, which is done a lot.
  // We could probably remove this mutation by preventing copies in those cases
  // or issuing small copy command buffers.
  *allowed_usage |=
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_allocator_allocate_internal(
    iree_hal_cuda_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_memory_access_t allowed_access, size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;

  void* host_ptr = NULL;
  CUdeviceptr device_ptr = NULL;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    auto flags = CU_MEMHOSTALLOC_DEVICEMAP;
    if (!iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
      flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    CUDA_RETURN_IF_ERROR(allocator->syms,
                         cuMemHostAlloc(&host_ptr, allocation_size, flags),
                         "cuMemHostAlloc");
    CUDA_RETURN_IF_ERROR(
        allocator->syms,
        cuMemHostGetDevicePointer(&device_ptr, host_ptr, /*flags=*/0),
        "cuMemHostGetDevicePointer");
  } 
  else {
    CUDA_RETURN_IF_ERROR(allocator->syms,
                         cuMemAlloc(&device_ptr, allocation_size),
                         "cuMemAlloc");
  }

  return iree_hal_cuda_buffer_wrap(
      (iree_hal_allocator_t*)allocator, memory_type, allowed_access,
      allowed_usage, allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/allocation_size, device_ptr, host_ptr, out_buffer);
}

static iree_status_t iree_hal_cuda_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_cuda_allocator_t* allocator =
      iree_hal_cuda_allocator_cast(base_allocator);

  // Coerce options into those required for use by VMA.
  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_ALL;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_allocator_make_compatible(
      &memory_type, &allowed_access, &allowed_usage));

  return iree_hal_cuda_allocator_allocate_internal(
      allocator, memory_type, allowed_usage, allowed_access, allocation_size,
      out_buffer);
}

void iree_hal_cuda_allocator_free(iree_hal_allocator_t* base_allocator,
                                  CUdeviceptr device_ptr, void* host_ptr,
                                  iree_hal_memory_type_t memory_type) {
  iree_hal_cuda_allocator_t* allocator =
      iree_hal_cuda_allocator_cast(base_allocator);
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    CUDA_CHECK_OK(allocator->syms, cuMemFreeHost(host_ptr));
  } else {
    CUDA_CHECK_OK(allocator->syms, cuMemFree(device_ptr));
  }
}

static iree_status_t iree_hal_cuda_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wrapping of external buffers not supported");
}

const iree_hal_allocator_vtable_t iree_hal_cuda_allocator_vtable = {
    /*.destroy=*/iree_hal_cuda_allocator_destroy,
    /*.host_allocator=*/iree_hal_cuda_allocator_host_allocator,
    /*.query_buffer_compatibility = */
    iree_hal_cuda_allocator_query_buffer_compatibility,
    /*.allocate_buffer=*/iree_hal_cuda_allocator_allocate_buffer,
    /*.wrap_buffer=*/iree_hal_cuda_allocator_wrap_buffer,
};
