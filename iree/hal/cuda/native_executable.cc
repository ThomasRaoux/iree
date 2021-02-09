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
#include "iree/hal/cuda/handle_util.h"
#include "iree/hal/cuda/status_util.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
//#include "iree/schemas/spirv_executable_def_reader.h"
//#include "iree/schemas/spirv_executable_def_verifier.h"


typedef struct {
  iree_hal_resource_t resource;
  iree::hal::cuda::CuContextHandle* logical_device;
  iree_host_size_t pipeline_count;
  CUfunction function;
  CUmodule module;
} iree_hal_cuda_native_executable_t;

extern const iree_hal_executable_vtable_t
    iree_hal_cuda_native_executable_vtable;

static iree_hal_cuda_native_executable_t*
iree_hal_cuda_native_executable_cast(iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_native_executable_vtable);
  return (iree_hal_cuda_native_executable_t*)base_value;
}

iree_status_t iree_hal_cuda_native_executable_create(
    iree::hal::cuda::CuContextHandle* logical_device,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(executable_spec);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_native_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable);// + pipeline_count * sizeof(*executable->pipelines);
  iree_status_t status = iree_allocator_malloc(logical_device->host_allocator(),
                                               total_size, (void**)&executable);
  // Hardcoded add op.
 const std::string ptx =
      ".version 6.4\n"
      ".target sm_30\n"
      ".address_size 64\n"
      ".visible .entry Kernel(\n"
      "    .param .u64 Kernel_param_0,\n"
      "    .param .u64 Kernel_param_1,\n"
      "    .param .u64 Kernel_param_2\n"
      ") {\n"
      "    .reg .f32 %f<4>;\n"
      "    .reg .b32 %r<2>;\n"
      "    .reg .b64 %rd<11>;\n"
      "    ld.param.u64 %rd1, [Kernel_param_0];\n"
      "    ld.param.u64 %rd2, [Kernel_param_1];\n"
      "    ld.param.u64 %rd3, [Kernel_param_2];\n"
      "    cvta.to.global.u64 %rd4, %rd3;\n"
      "    cvta.to.global.u64 %rd5, %rd2;\n"
      "    cvta.to.global.u64 %rd6, %rd1;\n"
      "    mov.u32 %r1, %tid.x;\n"
      "    mul.wide.u32 %rd7, %r1, 4;\n"
      "    add.s64 %rd8, %rd6, %rd7;\n"
      "    ld.global.f32 %f1, [%rd8];\n"
      "    add.s64 %rd9, %rd5, %rd7;\n"
      "    ld.global.f32 %f2, [%rd9];\n"
      "    add.f32 %f3, %f1, %f2;\n"
      "    add.s64 %rd10, %rd4, %rd7;\n"
      "    st.global.f32   [%rd10], %f3;\n"
      "    ret;\n"
      "}\n";
  const std::string entry_point = "Kernel";

  char log_buffer[1024 * 1024] = {};
  CUjit_option jit_options[] = {CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                CU_JIT_INFO_LOG_BUFFER};
  void* jit_values[] = {(void*)(sizeof(log_buffer)),
                        log_buffer};

  CUmodule module = nullptr;
  CUDA_RETURN_IF_ERROR(
      logical_device->syms().get(),
      cuModuleLoadDataEx(&module, ptx.c_str(),
                         sizeof(jit_options) / sizeof(jit_options[0]),
                         jit_options, jit_values),
      "cuModuleLoadDataEx");
  if (std::strlen(log_buffer)) {
    IREE_DVLOG(2) << "Compilation log:\n" << log_buffer;
  }

  CUfunction function = nullptr;
  CUDA_RETURN_IF_ERROR(
      logical_device->syms().get(),
      cuModuleGetFunction(&function, module, entry_point.c_str()),
      "cuModuleGetFunction");
  iree_hal_resource_initialize(&iree_hal_cuda_native_executable_vtable,
                               &executable->resource);      
  executable->function = function;
  executable->module = module;
  executable->logical_device   = logical_device;
  *out_executable = (iree_hal_executable_t*)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

CUfunction iree_hal_cuda_native_executable_for_entry_point(
    iree_hal_executable_t* base_executable, int32_t entry_point) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  return executable->function;
}

static void iree_hal_cuda_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  iree_allocator_t host_allocator =
      executable->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_executable_vtable_t iree_hal_cuda_native_executable_vtable = {
    /*.destroy=*/iree_hal_cuda_native_executable_destroy,
};
