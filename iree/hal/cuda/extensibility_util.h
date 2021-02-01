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

#ifndef IREE_HAL_VULKAN_EXTENSIBILITY_UTIL_H_
#define IREE_HAL_VULKAN_EXTENSIBILITY_UTIL_H_

#include "iree/base/arena.h"
#include "iree/hal/cuda/api.h"
#include "iree/hal/cuda/dynamic_symbols.h"

typedef struct {
  iree_host_size_t count;
  const char** values;
} iree_hal_cuda_string_list_t;

// Bits for enabled device extensions.
// We must use this to query support instead of just detecting symbol names as
// ICDs will resolve the functions sometimes even if they don't support the
// extension (or we didn't ask for it to be enabled).
typedef struct {

} iree_hal_cuda_device_extensions_t;

#endif  // IREE_HAL_VULKAN_EXTENSIBILITY_UTIL_H_
