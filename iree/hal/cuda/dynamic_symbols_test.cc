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

#include "iree/hal/cuda/dynamic_symbols.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cuda {
namespace {

#define CUDE_CHECK_ERRORS(expr)     \
  {                                 \
    CUresult status = expr;         \
    ASSERT_EQ(CUDA_SUCCESS, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  auto status_or_syms = DynamicSymbols::CreateFromSystemLoader();
  IREE_ASSERT_OK(status_or_syms);
  ref_ptr<DynamicSymbols> syms = std::move(status_or_syms.value());
  
  int device_count = 0;
  CUDE_CHECK_ERRORS(syms->cuInit(0));
  CUDE_CHECK_ERRORS(syms->cuDeviceGetCount(&device_count));
  CUdevice device;
  CUDE_CHECK_ERRORS(syms->cuDeviceGet(&device, /*ordinal=*/0));
}

}  // namespace
}  // namespace cuda
}  // namespace hal
}  // namespace iree
