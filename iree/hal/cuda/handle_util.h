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

// Helpers for wrapping Cuda handles that don't require us to wrap every type.
// This keeps our compilation time reasonable (as the cudacpp library is
// insane) while giving us nice safety around cleanup and ensuring we use
// dynamic symbols and consistent allocators.
//
// Do not add functionality beyond handle management to these types. Keep our
// Cuda usage mostly functional and C-like to ensure minimal code size and
// readability.

#ifndef IREE_HAL_CUDA_HANDLE_UTIL_H_
#define IREE_HAL_CUDA_HANDLE_UTIL_H_

// clang-format off: Must be included before all other headers:
#include "iree/hal/cuda/cuda_headers.h"
// clang-format on

#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/base/synchronization.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/extensibility_util.h"
#include "iree/hal/cuda/status_util.h"

namespace iree {
namespace hal {
namespace cuda {

class CuContextHandle : public RefObject<CuContextHandle> {
 public:
  CuContextHandle(DynamicSymbols* syms,
                 bool owns_device, iree_allocator_t host_allocator)
      : syms_(add_ref(syms)),
        owns_device_(owns_device),
        host_allocator_(host_allocator) {}
  ~CuContextHandle() { reset(); }

  CuContextHandle(const CuContextHandle&) = delete;
  CuContextHandle& operator=(const CuContextHandle&) = delete;
  CuContextHandle(CuContextHandle&& other) noexcept
      : value_(iree::exchange(other.value_,
                              static_cast<CUcontext>(NULL))),
        syms_(std::move(other.syms_)),
        host_allocator_(other.host_allocator_) {}

  void reset() {
    if(value_ == NULL) return;
    if(owns_device_) {
      syms_->cuCtxDestroy(value_);
    }
    value_ = NULL;
  }

  CUcontext value() const noexcept { return value_; }
  CUcontext* mutable_value() noexcept { return &value_; }
  operator CUcontext() const noexcept { return value_; }

  const ref_ptr<DynamicSymbols>& syms() const noexcept { return syms_; }
  iree_allocator_t host_allocator() const noexcept { return host_allocator_; }

 private:
  CUcontext value_ = NULL;
  ref_ptr<DynamicSymbols> syms_;
  bool owns_device_;
  iree_allocator_t host_allocator_;
};

}  // namespace cuda
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CUDA_HANDLE_UTIL_H_
