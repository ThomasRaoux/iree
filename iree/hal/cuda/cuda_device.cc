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
#include "iree/hal/cuda/command_queue.h"
#include "iree/hal/cuda/direct_command_buffer.h"
#include "iree/hal/cuda/direct_command_queue.h"
//#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/handle_util.h"
#include "iree/hal/cuda/status_util.h"

using namespace iree::hal::cuda;

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
  // Which optional extensions are active and available on the device.
  iree_hal_cuda_device_extensions_t device_extensions;

  CUdevice physical_device;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // All queues available on the device; the device owns these.
  iree_host_size_t queue_count;
  CommandQueue** queues;
  // The subset of queues that support dispatch operations. May overlap with
  // transfer_queues.
  iree_host_size_t dispatch_queue_count;
  CommandQueue** dispatch_queues;
  // The subset of queues that support transfer operations. May overlap with
  // dispatch_queues.
  iree_host_size_t transfer_queue_count;
  CommandQueue** transfer_queues;

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

  // Drop all command queues. These may wait until idle in their destructor.
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    delete device->queues[i];
  }

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_device_query_extensibility_set(
    iree_hal_cuda_features_t requested_features,
    iree_hal_cuda_extensibility_set_t set, iree::Arena* arena,
    iree_hal_cuda_string_list_t* out_string_list) {
  IREE_RETURN_IF_ERROR(iree_hal_cuda_query_extensibility_set(
      requested_features, set, 0, NULL, &out_string_list->count));
  out_string_list->values = (const char**)arena->AllocateBytes(
      out_string_list->count * sizeof(out_string_list->values[0]));
  IREE_RETURN_IF_ERROR(iree_hal_cuda_query_extensibility_set(
      requested_features, set, out_string_list->count, out_string_list->values,
      &out_string_list->count));
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_cuda_features_t enabled_features,
    const iree_hal_cuda_device_options_t* options,
    iree_hal_cuda_syms_t* opaque_syms,
    CUdevice physical_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  DynamicSymbols* instance_syms = (DynamicSymbols*)opaque_syms;
  
  // TODO: code to create the CUDA device
  return iree_ok_status();
}

static iree_string_view_t iree_hal_cuda_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_cuda_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->host_allocator;
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

  // TODO: code to create the command buffer
  return iree_ok_status();
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
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
  /*return iree_hal_cuda_native_descriptor_set_layout_create(
      device->logical_device, usage_type, binding_count, bindings,
      out_descriptor_set_layout);*/
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
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constants,
    iree_hal_executable_layout_t** out_executable_layout) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
 return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "semaphore not implemented");      
}

// Returns the queue to submit work to based on the |queue_affinity|.
static CommandQueue* iree_hal_cuda_device_select_queue(
    iree_hal_cuda_device_t* device,
    iree_hal_command_category_t command_categories, uint64_t queue_affinity) {
  // TODO(benvanik): meaningful heuristics for affinity. We don't generate
  // anything from the compiler that uses multiple queues and until we do it's
  // best not to do anything too clever here.
  if (command_categories == IREE_HAL_COMMAND_CATEGORY_TRANSFER) {
    return device
        ->transfer_queues[queue_affinity % device->transfer_queue_count];
  }
  return device->dispatch_queues[queue_affinity % device->dispatch_queue_count];
}

static iree_status_t iree_hal_cuda_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories, uint64_t queue_affinity,
    iree_host_size_t batch_count, const iree_hal_submission_batch_t* batches) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  CommandQueue* queue = iree_hal_cuda_device_select_queue(
      device, command_categories, queue_affinity);
  return queue->Submit(batch_count, batches);
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
  }*/
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    IREE_RETURN_IF_ERROR(device->queues[i]->WaitIdle(deadline_ns));
  }
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
