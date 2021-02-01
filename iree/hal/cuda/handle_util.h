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

class CuDeviceHandle : public RefObject<CuDeviceHandle> {
 public:
  CuDeviceHandle(DynamicSymbols* syms,
                 iree_hal_cuda_device_extensions_t enabled_extensions,
                 bool owns_device, iree_allocator_t host_allocator)
      : syms_(add_ref(syms)),
        enabled_extensions_(enabled_extensions),
        owns_device_(owns_device),
        host_allocator_(host_allocator) {}
  ~CuDeviceHandle() { reset(); }

  CuDeviceHandle(const CuDeviceHandle&) = delete;
  CuDeviceHandle& operator=(const CuDeviceHandle&) = delete;
  CuDeviceHandle(CuDeviceHandle&& other) noexcept
      : value_(iree::exchange(other.value_,
                              static_cast<CUdevice>(NULL))),
        syms_(std::move(other.syms_)),
        enabled_extensions_(other.enabled_extensions_),
        host_allocator_(other.host_allocator_) {}

  void reset() {
    if (value_ == NULL) return;
    value_ = NULL;
  }

  CUdevice value() const noexcept { return value_; }
  CUdevice* mutable_value() noexcept { return &value_; }
  operator CUdevice() const noexcept { return value_; }

  const ref_ptr<DynamicSymbols>& syms() const noexcept { return syms_; }
  iree_allocator_t host_allocator() const noexcept { return host_allocator_; }

  const iree_hal_cuda_device_extensions_t& enabled_extensions() const {
    return enabled_extensions_;
  }

 private:
  CUdevice value_ = NULL;
  ref_ptr<DynamicSymbols> syms_;
  iree_hal_cuda_device_extensions_t enabled_extensions_;
  bool owns_device_;
  iree_allocator_t host_allocator_;
};

class CuCommandPoolHandle {
 public:
  explicit CuCommandPoolHandle(CuDeviceHandle* logical_device)
      : logical_device_(logical_device) {
    iree_slim_mutex_initialize(&mutex_);
  }
  ~CuCommandPoolHandle() {
    reset();
    iree_slim_mutex_deinitialize(&mutex_);
  }

  CuCommandPoolHandle(const CuCommandPoolHandle&) = delete;
  CuCommandPoolHandle& operator=(const CuCommandPoolHandle&) = delete;
  CuCommandPoolHandle(CuCommandPoolHandle&& other) noexcept
      : logical_device_(std::move(other.logical_device_)) {}
  CuCommandPoolHandle& operator=(CuCommandPoolHandle&& other) {
    std::swap(logical_device_, other.logical_device_);
    return *this;
  }

  void reset() {
  }

  const CuDeviceHandle* logical_device() const noexcept {
    return logical_device_;
  }
  const ref_ptr<DynamicSymbols>& syms() const noexcept {
    return logical_device_->syms();
  }

 private:
  CuDeviceHandle* logical_device_;

  // Cuda command pools are not thread safe and require external
  // synchronization. Since we allow arbitrary threads to allocate and
  // deallocate the HAL command buffers we need to externally synchronize.
  iree_slim_mutex_t mutex_;
};

}  // namespace cuda
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CUDA_HANDLE_UTIL_H_
