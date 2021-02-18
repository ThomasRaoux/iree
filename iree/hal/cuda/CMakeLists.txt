# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT ${IREE_HAL_DRIVER_CUDA})
  return()
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    cuda
  HDRS
    "api.h"
  SRCS
    "api.h"
    "context_wrapper.h"
    "cuda_allocator.cc"
    "cuda_allocator.h"
    "cuda_buffer.cc"
    "cuda_buffer.h"
    "cuda_device.cc"
    "cuda_device.h"
    "cuda_driver.cc"
    "status_util.cc"
    "status_util.h"
  DEPS
    ::dynamic_symbols
    iree::base::api
    iree::base::core_headers
    iree::base::flatcc
    iree::base::internal
    iree::base::logging
    iree::base::status
    iree::base::synchronization
    iree::base::tracing
    iree::hal::api
    iree::schemas::cuda_executable_def_c_fbs
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
  SRCS
    "cuda_headers.h"
    "dynamic_symbols.cc"
    "dynamic_symbols_tables.h"
  DEPS
    absl::span
    cuda_headers
    iree::base::core_headers
    iree::base::dynamic_library
    iree::base::status
    iree::base::tracing
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_symbols_test
  SRCS
    "dynamic_symbols_test.cc"
  DEPS
    ::dynamic_symbols
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=cuda"
)
