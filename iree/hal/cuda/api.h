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

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_CUDA_API_H_
#define IREE_HAL_CUDA_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
//===----------------------------------------------------------------------===//
// iree_hal_cuda_driver_t
//===----------------------------------------------------------------------===//

// Cuda driver creation options.
typedef struct {
  // Cuda version that will be requested, e.g. `VK_API_VERSION_1_0`.
  // Driver creation will fail if the required version is not available.
  uint32_t api_version;

  // Index of the default Cuda device to use within the list of available
  // devices.
  int default_device_index;
} iree_hal_cuda_driver_options_t;

IREE_API_EXPORT void IREE_API_CALL iree_hal_cuda_driver_options_initialize(
    iree_hal_cuda_driver_options_t* out_options);

// Creates a Cuda HAL driver that manages its own VkInstance.
//
// |out_driver| must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_cuda_driver_create(
    iree_string_view_t identifier,
    const iree_hal_cuda_driver_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);


#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_API_H_
