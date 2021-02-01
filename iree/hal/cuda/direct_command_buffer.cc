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

#include "iree/hal/cuda/direct_command_buffer.h"

#include "absl/container/inlined_vector.h"
#include "iree/base/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/status_util.h"

using namespace iree::hal::cuda;

// Command buffer implementation that directly maps to VkCommandBuffer.
// This records the commands on the calling thread without additional threading
// indirection.
typedef struct {
  iree_hal_resource_t resource;
  CuDeviceHandle* logical_device;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t allowed_categories;

  CuCommandPoolHandle* command_pool;

  DynamicSymbols* syms;

  // The current descriptor set group in use by the command buffer, if any.
  // This must remain valid until all in-flight submissions of the command
  // buffer complete.
  //DescriptorSetGroup descriptor_set_group;
} iree_hal_cuda_direct_command_buffer_t;

extern const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_direct_command_buffer_vtable;

static iree_hal_cuda_direct_command_buffer_t*
iree_hal_cuda_direct_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_cuda_direct_command_buffer_vtable);
  return (iree_hal_cuda_direct_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_direct_command_buffer_allocate(
    iree::hal::cuda::CuDeviceHandle* logical_device,
    iree::hal::cuda::CuCommandPoolHandle* command_pool,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(command_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static void iree_hal_cuda_direct_command_buffer_reset(
    iree_hal_cuda_direct_command_buffer_t* command_buffer) {
  // NOTE: we require that command buffers not be recorded while they are
  // in-flight so this is safe.
  //IREE_IGNORE_ERROR(command_buffer->descriptor_set_group.Reset());
}

static void iree_hal_cuda_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_command_category_t
iree_hal_cuda_direct_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* base_command_buffer) {
  return ((const iree_hal_cuda_direct_command_buffer_t*)base_command_buffer)
      ->allowed_categories;
}

static iree_status_t iree_hal_cuda_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);

  iree_hal_cuda_direct_command_buffer_reset(command_buffer);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                 "not implemented yet for CUDA");
  ;
}

static iree_status_t iree_hal_cuda_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");;
}

static iree_status_t iree_hal_cuda_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // NOTE: we could use this to prevent queue family transitions.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

static iree_status_t iree_hal_cuda_direct_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  iree_hal_cuda_direct_command_buffer_t* command_buffer =
      iree_hal_cuda_direct_command_buffer_cast(base_command_buffer);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_direct_command_buffer_vtable = {
        /*.destroy=*/iree_hal_cuda_direct_command_buffer_destroy,
        /*.allowed_categories=*/
        iree_hal_cuda_direct_command_buffer_allowed_categories,
        /*.begin=*/iree_hal_cuda_direct_command_buffer_begin,
        /*.end=*/iree_hal_cuda_direct_command_buffer_end,
        /*.execution_barrier=*/
        iree_hal_cuda_direct_command_buffer_execution_barrier,
        /*.signal_event=*/
        iree_hal_cuda_direct_command_buffer_signal_event,
        /*.reset_event=*/iree_hal_cuda_direct_command_buffer_reset_event,
        /*.wait_events=*/iree_hal_cuda_direct_command_buffer_wait_events,
        /*.discard_buffer=*/
        iree_hal_cuda_direct_command_buffer_discard_buffer,
        /*.fill_buffer=*/iree_hal_cuda_direct_command_buffer_fill_buffer,
        /*.update_buffer=*/
        iree_hal_cuda_direct_command_buffer_update_buffer,
        /*.copy_buffer=*/iree_hal_cuda_direct_command_buffer_copy_buffer,
        /*.push_constants=*/
        iree_hal_cuda_direct_command_buffer_push_constants,
        /*.push_descriptor_set=*/
        iree_hal_cuda_direct_command_buffer_push_descriptor_set,
        /*.bind_descriptor_set=*/
        iree_hal_cuda_direct_command_buffer_bind_descriptor_set,
        /*.dispatch=*/iree_hal_cuda_direct_command_buffer_dispatch,
        /*.dispatch_indirect=*/
        iree_hal_cuda_direct_command_buffer_dispatch_indirect,
};
