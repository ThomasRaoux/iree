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
// limitations under the License.

#include "iree/hal/cuda/cuda_device.h"

#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/math.h"
#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/api.h"
#include "iree/hal/cuda/cuda_allocator.h"
#include "iree/hal/cuda/executable_layout.h"
#include "iree/hal/cuda/nop_executable_cache.h"
#include "iree/hal/cuda/descriptor_set_layout.h"

#include "iree/hal/cuda/event_semaphore.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/status_util.h"
#include "iree/hal/cuda/graph_command_buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda_device_t extensibility util
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_cuda_query_extensibility_set(
    iree_hal_cuda_features_t requested_features,
    iree_hal_cuda_extensibility_set_t set, iree_host_size_t string_capacity,
    const char** out_string_values, iree_host_size_t* out_string_count) {
  *out_string_count = 0;
  iree_status_t status = iree_ok_status();
  return status;
}

//===----------------------------------------------------------------------===//
// Queue selection
//===----------------------------------------------------------------------===//

#define IREE_HAL_CUDA_INVALID_QUEUE_FAMILY_INDEX (-1)

typedef struct {
  uint32_t dispatch_index;
  iree_host_size_t dispatch_queue_count;
  uint32_t transfer_index;
  iree_host_size_t transfer_queue_count;
} iree_hal_cuda_queue_family_info_t;

//===----------------------------------------------------------------------===//
// iree_hal_cuda_device_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Optional driver that owns the instance. We retain it for our lifetime to
  // ensure the instance remains valid.
  iree_hal_driver_t* driver;

  // Flags overriding default device behavior.
  iree_hal_cuda_device_flags_t flags;

  CUdevice device;

  // TODO: support multiple streams.
  CUstream stream;
  iree_hal_cuda_context_wrapper_t context_wrapper;
  iree_hal_allocator_t* device_allocator;

} iree_hal_cuda_device_t;

extern const iree_hal_device_vtable_t iree_hal_cuda_device_vtable;

static iree_hal_cuda_device_t* iree_hal_cuda_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_device_vtable);
  return (iree_hal_cuda_device_t*)base_value;
}

IREE_API_EXPORT void IREE_API_CALL iree_hal_cuda_device_options_initialize(
    iree_hal_cuda_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->flags = 0;
}

static void iree_hal_cuda_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);


  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);
  CUDA_CHECK_OK(device->context_wrapper.syms,
                cuStreamDestroy(device->stream));

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    CUdevice cu_device, CUstream stream,
    CUcontext context, iree::hal::cuda::DynamicSymbols* syms,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_cuda_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_cuda_device_vtable,
                               &device->resource);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  uint8_t* buffer_ptr = (uint8_t*)device + sizeof(*device);
  buffer_ptr += iree_string_view_append_to_buffer(
      identifier, &device->identifier, (char*)buffer_ptr);
  device->device = cu_device;
  device->stream = stream;
  device->context_wrapper.cu_context = context;
  device->context_wrapper.host_allocator = host_allocator;
  device->context_wrapper.syms = syms;
  *out_device = (iree_hal_device_t*)device;

  iree_status_t status = iree_hal_cuda_allocator_create(
      &device->context_wrapper,
      &device->device_allocator);

  return status;
}

iree_status_t iree_hal_cuda_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_cuda_features_t enabled_features,
    const iree_hal_cuda_device_options_t* options,
    iree::hal::cuda::DynamicSymbols* syms,
    CUdevice device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  CUcontext context;
  CUDA_RETURN_IF_ERROR(syms,
                       cuCtxCreate(&context, 0, device),
                       "cuCtxCreate");
  CUstream stream;
  CUDA_RETURN_IF_ERROR(syms, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                       "cuStreamCreate");

  return iree_hal_cuda_device_create_internal(driver, identifier, device,
                                              stream, context, syms,
                                              host_allocator, out_device);
}

static iree_string_view_t iree_hal_cuda_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_cuda_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->context_wrapper.host_allocator;
}

static iree_hal_allocator_t* iree_hal_cuda_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->device_allocator;
}

static iree_status_t iree_hal_cuda_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_graph_command_buffer_allocate(
      &device->context_wrapper, mode, command_categories,
      out_command_buffer);
}

static iree_status_t iree_hal_cuda_device_create_descriptor_set(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  // TODO(benvanik): rework the create fn to take the bindings.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "non-push descriptor sets still need work");
}

static iree_status_t iree_hal_cuda_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_descriptor_set_layout_create(
      &device->context_wrapper, usage_type, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_cuda_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_nop_executable_cache_create(
      &device->context_wrapper, identifier, out_executable_cache);
}

static iree_status_t iree_hal_cuda_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constants,
    iree_hal_executable_layout_t** out_executable_layout) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_executable_layout_create(
      &device->context_wrapper, set_layout_count, set_layouts, push_constants,
      out_executable_layout);
}

static iree_status_t iree_hal_cuda_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_semaphore_create(
      &device->context_wrapper, initial_value, out_semaphore);
}

static iree_status_t iree_hal_cuda_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories, uint64_t queue_affinity,
    iree_host_size_t batch_count, const iree_hal_submission_batch_t* batches) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  for (int i = 0; i < batch_count; i++) {
    for (int j = 0; j < batches[i].command_buffer_count; j++) {
      CUgraphExec exec = iree_hal_cuda_graph_command_buffer_exec(
          batches[i].command_buffers[j]);
      CUDA_RETURN_IF_ERROR(device->context_wrapper.syms,
                           cuGraphLaunch(exec, device->stream),
                           "cuGraphLaunch");
    }
  }
  CUDA_RETURN_IF_ERROR(device->context_wrapper.syms,
                       cuStreamSynchronize(device->stream),
                       "cuStreamSynchronize");
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_wait_semaphores_with_timeout(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list,
    iree_duration_t timeout_ns) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "semaphore not implemented");
}

static iree_status_t iree_hal_cuda_device_wait_semaphores_with_deadline(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "semaphore not implemented");
}

static iree_status_t iree_hal_cuda_device_wait_idle_with_deadline(
    iree_hal_device_t* base_device, iree_time_t deadline_ns) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
 /* if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Fast path for using vkDeviceWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    return CU_RESULT_TO_STATUS(device->logical_device->syms()->vkDeviceWaitIdle(
                                   *device->logical_device),
                               "vkDeviceWaitIdle");
  }
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    IREE_RETURN_IF_ERROR(device->queues[i]->WaitIdle(deadline_ns));
  }*/
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_wait_idle_with_timeout(
    iree_hal_device_t* base_device, iree_duration_t timeout_ns) {
  return iree_hal_cuda_device_wait_idle_with_deadline(
      base_device, iree_relative_timeout_to_deadline_ns(timeout_ns));
}

const iree_hal_device_vtable_t iree_hal_cuda_device_vtable = {
    /*.destroy=*/iree_hal_cuda_device_destroy,
    /*.id=*/iree_hal_cuda_device_id,
    /*.host_allocator=*/iree_hal_cuda_device_host_allocator,
    /*.device_allocator=*/iree_hal_cuda_device_allocator,
    /*.create_command_buffer=*/iree_hal_cuda_device_create_command_buffer,
    /*.create_descriptor_set=*/iree_hal_cuda_device_create_descriptor_set,
    /*.create_descriptor_set_layout=*/
    iree_hal_cuda_device_create_descriptor_set_layout,
    /*.create_event=*/iree_hal_cuda_device_create_event,
    /*.create_executable_cache=*/
    iree_hal_cuda_device_create_executable_cache,
    /*.create_executable_layout=*/
    iree_hal_cuda_device_create_executable_layout,
    /*.create_semaphore=*/iree_hal_cuda_device_create_semaphore,
    /*.queue_submit=*/iree_hal_cuda_device_queue_submit,
    /*.wait_semaphores_with_deadline=*/
    iree_hal_cuda_device_wait_semaphores_with_deadline,
    /*.wait_semaphores_with_timeout=*/
    iree_hal_cuda_device_wait_semaphores_with_timeout,
    /*.wait_idle_with_deadline=*/
    iree_hal_cuda_device_wait_idle_with_deadline,
    /*.wait_idle_with_timeout=*/
    iree_hal_cuda_device_wait_idle_with_timeout,
};
