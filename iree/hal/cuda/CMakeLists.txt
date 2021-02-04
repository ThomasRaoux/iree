# Copyright 2019 Google LLC
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
    "cuda_device.h"
    "cuda_driver.h"
  SRCS
    "api.cc"
    "cuda_allocator.cc"
    "cuda_buffer.cc"
    "cuda_device.cc"
    "cuda_driver.cc"
    "cuda_headers.h"
    "graph_command_buffer.cc"
    "handle_util.h"
    "status_util.cc"
    "status_util.h"
  DEPS
    ::dynamic_symbols
    absl::core_headers
    absl::flat_hash_map
    absl::inlined_vector
    absl::memory
    absl::span
    absl::strings
    absl::synchronization
    iree::base::api
    iree::base::arena
    iree::base::core_headers
    iree::base::flatcc
    iree::base::intrusive_list
    iree::base::logging
    iree::base::ref_ptr
    iree::base::status
    iree::base::synchronization
    iree::base::tracing
    iree::hal::api
  PUBLIC
)


iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    #"dynamic_symbol_tables.h"
    "dynamic_symbols.h"
  SRCS
    "dynamic_symbols.cc"
    "cuda_headers.h"
  DEPS
    absl::core_headers
    absl::memory
    absl::span
    absl::strings
    iree::base::core_headers
    iree::base::dynamic_library
    iree::base::ref_ptr
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
