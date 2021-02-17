// Copyright 2021 Google LLC
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

#include "iree/hal/cuda/native_executable.h"

#include "iree/base/memory.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/status_util.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/cuda_executable_def_reader.h"
#include "iree/schemas/cuda_executable_def_verifier.h"


typedef struct {
  iree_hal_resource_t resource;
  iree_hal_cuda_context_wrapper_t* context;
  iree_host_size_t entry_count;
  CUmodule module;
  CUfunction entry_functions[];
} iree_hal_cuda_native_executable_t;

extern const iree_hal_executable_vtable_t
    iree_hal_cuda_native_executable_vtable;

static iree_hal_cuda_native_executable_t*
iree_hal_cuda_native_executable_cast(iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_native_executable_vtable);
  return (iree_hal_cuda_native_executable_t*)base_value;
}

iree_status_t iree_hal_cuda_native_executable_create(
    iree_hal_cuda_context_wrapper_t* context,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(executable_spec);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_native_executable_t* executable = NULL;
  
  // TODO: Verify the flat buffer.
  iree_CudaExecutableDef_table_t executable_def =
      iree_CudaExecutableDef_as_root(executable_spec->executable_data.data);

  // Create the kernel module.
  flatbuffers_uint8_vec_t kernel_code =
      iree_CudaExecutableDef_kernel_library_get(executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_CudaExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_count =
      flatbuffers_string_vec_len(entry_points_vec);
  iree_host_size_t total_size =
      sizeof(*executable) + entry_count * sizeof(CUfunction);
  iree_status_t status = iree_allocator_malloc(context->host_allocator,
                                               total_size, (void**)&executable);
  char log_buffer[1024 * 1024] = {};
  CUjit_option jit_options[] = {CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                CU_JIT_INFO_LOG_BUFFER};
  void* jit_values[] = {(void*)(sizeof(log_buffer)),
                        log_buffer};

  CUmodule module = nullptr;
  CUDA_RETURN_IF_ERROR(
      context->syms,
      cuModuleLoadDataEx(&module, kernel_code,
                         sizeof(jit_options) / sizeof(jit_options[0]),
                         jit_options, jit_values),
      "cuModuleLoadDataEx");
  if (std::strlen(log_buffer)) {
    IREE_DVLOG(2) << "Compilation log:\n" << log_buffer;
  }

  for(iree_host_size_t i = 0; i < entry_count; i++) {
    CUfunction function = nullptr;
    const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
    CUDA_RETURN_IF_ERROR(
        context->syms,
        cuModuleGetFunction(&function, module, entry_name),
        "cuModuleGetFunction");
    executable->entry_functions[i] = function;
  }

  iree_hal_resource_initialize(&iree_hal_cuda_native_executable_vtable,
                               &executable->resource);
  executable->module = module;
  executable->context = context;
  *out_executable = (iree_hal_executable_t*)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

CUfunction iree_hal_cuda_native_executable_for_entry_point(
    iree_hal_executable_t* base_executable, int32_t entry_point) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  return executable->entry_functions[entry_point];
}

static void iree_hal_cuda_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  iree_allocator_t host_allocator =
      executable->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_executable_vtable_t iree_hal_cuda_native_executable_vtable = {
    /*.destroy=*/iree_hal_cuda_native_executable_destroy,
};
