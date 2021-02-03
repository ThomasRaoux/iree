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

#ifndef IREE_HAL_CUDA_ALLOCATOR_H_
#define IREE_HAL_CUDA_ALLOCATOR_H_

#include "iree/hal/api.h"
#include "iree/hal/cuda/handle_util.h"
#include "iree/hal/cuda/status_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a cuda allocator.
iree_status_t iree_hal_cuda_allocator_create(
    CUcontext context,
    iree::hal::cuda::DynamicSymbols* syms,
    iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Free an allocation represent by the given |pointer|.
void iree_hal_cuda_allocator_free(iree_hal_allocator_t* allocator,
                                  void* pointer,
                                  iree_hal_memory_type_t memory_type);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_ALLOCATOR_H_
