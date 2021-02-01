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

#include "iree/hal/cuda/debug_reporter.h"

#include "iree/base/tracing.h"
#include "iree/hal/cuda/status_util.h"

struct iree_hal_cuda_debug_reporter_s {
  iree_allocator_t host_allocator;
  iree::hal::cuda::DynamicSymbols* syms;
};


iree_status_t iree_hal_cuda_debug_reporter_allocate(
    iree::hal::cuda::DynamicSymbols* syms,
    iree_allocator_t host_allocator,
    iree_hal_cuda_debug_reporter_t** out_reporter) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(out_reporter);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate our struct first as we need to pass the pointer to the userdata
  // of the messager instance when we create it.
  iree_hal_cuda_debug_reporter_t* reporter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*reporter),
                                (void**)&reporter));
  reporter->host_allocator = host_allocator;
  reporter->syms = syms;

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "non-push descriptor sets still need work");
}

void iree_hal_cuda_debug_reporter_free(
    iree_hal_cuda_debug_reporter_t* reporter) {
  if (!reporter) return;
  iree_allocator_t host_allocator = reporter->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, reporter);

  IREE_TRACE_ZONE_END(z0);
}
