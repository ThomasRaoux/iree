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
  SRCS
    "context_wrapper.h"
    "cuda_allocator.cc"
    "cuda_buffer.cc"
    "cuda_device.cc"
    "cuda_driver.cc"
    "cuda_headers.h"
    "cuda_event.cc"
    "descriptor_set_layout.cc"
    "event_semaphore.cc"
    "executable_layout.cc"
    "graph_command_buffer.cc"
    "native_executable.cc"
    "nop_executable_cache.cc"
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
    cuda_headers
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
    iree::schemas::cuda_executable_def_c_fbs
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
    cuda_headers
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
