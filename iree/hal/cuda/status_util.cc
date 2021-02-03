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

#include "iree/hal/cuda/status_util.h"
#include "iree/hal/cuda/dynamic_symbols.h"

iree_status_t iree_hal_cuda_result_to_status(iree::hal::cuda::DynamicSymbols* syms, 
                                              CUresult result,
                                               const char* file,
                                               uint32_t line) {
  if (result == CUDA_SUCCESS) {
    return iree_ok_status();
  }

  const char* error_name;
  if (syms->cuGetErrorName(result, &error_name) != CUDA_SUCCESS) {
    error_name = "UNKNOWN";
  }

  const char* error_string;
  if (syms->cuGetErrorString(result, &error_string) != CUDA_SUCCESS) {
    error_string = "Unknown error.";
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "CUDA driver error '%s' (%d): %s", error_name, result,
                          error_string);
  // TODO(thomasraoux): print better error for out of memory.
  /*if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    size_t free, total;
    if (cuMemGetInfo(&free, &total) == CUDA_SUCCESS) {
      error_message +=
          absl::StrFormat("\n%zu bytes free of %zu bytes total.", free, total);
    }
  }*/
  
}
