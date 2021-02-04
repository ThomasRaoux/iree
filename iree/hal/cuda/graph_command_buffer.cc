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

#include "iree/hal/cuda/graph_command_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/cuda/status_util.h"
#include "iree/hal/cuda/cuda_buffer.h"

// Command buffer implementation that directly maps to cuda graph.
// This records the commands on the calling thread without additional threading
// indirection.
typedef struct {
  iree_hal_resource_t resource;
  CUcontext context;
  iree::hal::cuda::DynamicSymbols* syms;
  iree_allocator_t host_allocator;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t allowed_categories;
  CUgraph graph;
  CUgraphExec exec;
} iree_hal_cuda_graph_command_buffer_t;

extern const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_graph_command_buffer_vtable;

static iree_hal_cuda_graph_command_buffer_t*
iree_hal_cuda_graph_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_cuda_graph_command_buffer_vtable);
  return (iree_hal_cuda_graph_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_graph_command_buffer_allocate(
    CUcontext context,
    iree::hal::cuda::DynamicSymbols* syms,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUgraph graph;
  CUDA_RETURN_IF_ERROR(syms, cuGraphCreate(&graph, /*flags=*/0), "cuGraphCreate");

  iree_hal_cuda_graph_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*command_buffer), (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_graph_command_buffer_vtable,
                                 &command_buffer->resource);
    command_buffer->context = context;
    command_buffer->mode = mode;
    command_buffer->allowed_categories = command_categories;
    command_buffer->syms = syms;
    command_buffer->graph = graph;
    command_buffer->host_allocator = host_allocator;

    *out_command_buffer = (iree_hal_command_buffer_t*)command_buffer;
  } else {
    syms->cuGraphDestroy(graph);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_graph_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUDA_CHECK_OK(command_buffer->syms, cuGraphDestroy(command_buffer->graph));
  //CUDA_CHECK_OK(command_buffer->syms, cuGraphExecDestroy(command_buffer->exec));
  iree_allocator_free(command_buffer->host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

CUgraphExec iree_hal_cuda_graph_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  return command_buffer->exec;
}

static iree_hal_command_category_t
iree_hal_cuda_graph_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* base_command_buffer) {
  const iree_hal_cuda_graph_command_buffer_t* command_buffer =
      (const iree_hal_cuda_graph_command_buffer_t*)(base_command_buffer);
  return command_buffer->allowed_categories;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  // nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  size_t num_nodes;
  CUDA_RETURN_IF_ERROR(
      command_buffer->syms,
      cuGraphGetNodes(command_buffer->graph, nullptr, &num_nodes),
      "cuGraphGetNodes");
  IREE_DVLOG(2) << "Instantiating graph with " << num_nodes << " nodes";

  CUgraphNode error_node;
  char log[1024];
  auto result = command_buffer->syms->cuGraphInstantiate(
      &command_buffer->exec, command_buffer->graph, &error_node, log,
      sizeof(log));
  // CUDA_RETURN_IF_ERROR(cuGraphDestroy(command_buffer->graph));

 // if (result != CUDA_SUCCESS) {
//    return ::util::Annotate(CudaResultToStatus(result),
 //                           absl::string_view(buffer.data()));
  //}
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // nothing to do.
  return iree_ok_status();
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
static uint32_t splat_pattern(const void* pattern, size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *static_cast<const uint8_t*>(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *static_cast<const uint16_t*>(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *static_cast<const uint32_t*>(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_cuda_graph_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  void* target_device_buffer = iree_hal_cuda_buffer_base_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  uint32_t dword_pattern = splat_pattern(pattern, pattern_length);
  CUgraphNode node;
  CUDA_MEMSET_NODE_PARAMS params = {};
  params.dst = (CUdeviceptr)(target_device_buffer) + target_offset;
  params.elementSize = pattern_length;
  params.width = length;
  params.height = 1;
  params.value = dword_pattern;
  IREE_DVLOG(2) << "memset " << params.dst << " " << params.elementSize << " "
                << params.width << " " << params.value;
  CUDA_RETURN_IF_ERROR(
      command_buffer->syms,
      cuGraphAddMemsetNode(&node, command_buffer->graph, nullptr, 0, &params,
                           command_buffer->context),
      "cuGraphAddMemsetNode");
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

CUgraphExec iree_hal_cuda_graph_command_buffer_exec(
    const iree_hal_command_buffer_t* base_command_buffer) {
  const iree_hal_cuda_graph_command_buffer_t* command_buffer =
      (const iree_hal_cuda_graph_command_buffer_t*)(base_command_buffer);
  return command_buffer->exec;
}

const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_graph_command_buffer_vtable = {
        /*.destroy=*/iree_hal_cuda_graph_command_buffer_destroy,
        /*.allowed_categories=*/
        iree_hal_cuda_graph_command_buffer_allowed_categories,
        /*.begin=*/iree_hal_cuda_graph_command_buffer_begin,
        /*.end=*/iree_hal_cuda_graph_command_buffer_end,
        /*.execution_barrier=*/
        iree_hal_cuda_graph_command_buffer_execution_barrier,
        /*.signal_event=*/
        iree_hal_cuda_graph_command_buffer_signal_event,
        /*.reset_event=*/iree_hal_cuda_graph_command_buffer_reset_event,
        /*.wait_events=*/iree_hal_cuda_graph_command_buffer_wait_events,
        /*.discard_buffer=*/
        iree_hal_cuda_graph_command_buffer_discard_buffer,
        /*.fill_buffer=*/iree_hal_cuda_graph_command_buffer_fill_buffer,
        /*.update_buffer=*/
        iree_hal_cuda_graph_command_buffer_update_buffer,
        /*.copy_buffer=*/iree_hal_cuda_graph_command_buffer_copy_buffer,
        /*.push_constants=*/
        iree_hal_cuda_graph_command_buffer_push_constants,
        /*.push_descriptor_set=*/
        iree_hal_cuda_graph_command_buffer_push_descriptor_set,
        /*.bind_descriptor_set=*/
        iree_hal_cuda_graph_command_buffer_bind_descriptor_set,
        /*.dispatch=*/iree_hal_cuda_graph_command_buffer_dispatch,
        /*.dispatch_indirect=*/
        iree_hal_cuda_graph_command_buffer_dispatch_indirect,
};
