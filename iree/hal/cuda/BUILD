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

# HAL implementation using Cuda and (likely) SPIR-V executables.

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_CUDA})
  return()
endif()
""",
)

cc_library(
    name = "cuda",
    srcs = [
        "api.cc",
        "command_queue.h",
        "cuda_device.cc",
        "cuda_driver.cc",
        "cuda_headers.h",
        "debug_reporter.cc",
        "debug_reporter.h",
        "descriptor_pool_cache.cc",
        "descriptor_pool_cache.h",
        "descriptor_set_arena.cc",
        "descriptor_set_arena.h",
        "direct_command_buffer.cc",
        "direct_command_buffer.h",
        "direct_command_queue.cc",
        "direct_command_queue.h",
        "emulated_semaphore.cc",
        "emulated_semaphore.h",
        "handle_util.h",
        "status_util.c",
        "status_util.h",
        "timepoint_util.cc",
        "timepoint_util.h",
    ],
    hdrs = [
        # TODO(benvanik): hide all but api.h.
        "api.h",
        "cuda_device.h",
        "cuda_driver.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dynamic_symbols",
        "//iree/base:api",
        "//iree/base:arena",
        "//iree/base:core_headers",
        "//iree/base:flatcc",
        "//iree/base:intrusive_list",
        "//iree/base:logging",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:synchronization",
        "//iree/base:tracing",
        "//iree/hal:api",
        "//iree/schemas:spirv_executable_def_c_fbs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dynamic_symbols",
    srcs = [
        "dynamic_symbols.cc",
        "cuda_headers.h",
    ],
    hdrs = [
        "dynamic_symbol_tables.h",
        "dynamic_symbols.h",
    ],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:dynamic_library",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@iree_cuda_headers//:cuda_headers",
    ],
)

cc_test(
    name = "dynamic_symbols_test",
    srcs = ["dynamic_symbols_test.cc"],
    tags = ["driver=cuda"],
    deps = [
        ":dynamic_symbols",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
