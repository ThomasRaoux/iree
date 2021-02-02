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

#ifndef IREE_HAL_CUDA_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_CUDA_DYNAMIC_SYMBOLS_H_

// clang-format off: Must be included before all other headers:
#include "iree/hal/cuda/cuda_headers.h"
// clang-format on

#include <cstdint>
#include <functional>
#include <memory>

#include "iree/base/dynamic_library.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
//#include "iree/hal/cuda/dynamic_symbol_tables.h"

namespace iree {
namespace hal {
namespace cuda {

struct FunctionPtrInfo;

// Dynamic Cuda function loader for use with cuda.hpp.
// This loader is a subset of the DispatchLoaderDynamic implementation that only
// loads functions we are interested in (a compute-specific subset) and avoids
// extensions we will never use.
//
// This exposes all Cuda methods as function pointer members. Optional
// methods will be nullptr if not present. Excluded methods will be omitted.
//
// DynamicSymbols instances are designed to be passed to cuda.hpp methods as
// the last argument, though they may also be called directly.
// **Always make sure to pass the loader to cuda.hpp methods!**
//
// Loading is performed by walking a table of required and optional functions
// (defined in dynamic_symbol_tables.h) and populating the member function
// pointers exposed on this struct when available. For example, if the
// vkSomeFunction method is marked in the table as OPTIONAL the loader will
// attempt to lookup the function and if successful set the
// DynamicSymbols::vkSomeFunction pointer to the resolved address. If the
// function is not found then it will be set to nullptr so users can check for
// function availability.
//
// Documentation:
// https://github.com/KhronosGroup/Cuda-Hpp#extensions--per-device-function-pointers
//
// Usage:
//  IREE_ASSIGN_OR_RETURN(auto syms, DynamicSymbols::CreateFromSystemLoader());
//  VkInstance instance = VK_NULL_HANDLE;
//  syms->vkCreateInstance(..., &instance);
//  IREE_RETURN_IF_ERROR(syms->LoadFromInstance(instance));
struct DynamicSymbols : public RefObject<DynamicSymbols> {
  DynamicSymbols();
  ~DynamicSymbols();

  // Loads all required and optional Cuda functions from the Cuda loader.
  // This will look for a Cuda loader on the system (like libcuda.so) and
  // dlsym the functions from that.
  //
  // The loaded function pointers will point to thunks in the ICD. This may
  // enable additional debug checking and more readable stack traces (as
  // errors come from within the ICD, where we have symbols).
  static StatusOr<ref_ptr<DynamicSymbols>> CreateFromSystemLoader();

  // Loads all required and optional Cuda functions from the given device,
  // falling back to the instance when required.
  //
  // This attempts to directly query the methods from the device, bypassing any
  // ICD or shim layers. These methods will generally have less overhead at
  // runtime as they need not jump through the various trampolines.
  Status LoadFromDevice(CUdevice device);

#define CU_PFN_DECL(cudaSymbolName) \
  std::add_pointer<decltype(::cudaSymbolName)>::type cudaSymbolName;

#include "dynamic_symbol_tables.def"
#undef CU_PFN_DECL

 private:
  // Optional Cuda Loader dynamic library.
  std::unique_ptr<DynamicLibrary> loader_library_;
};

}  // namespace cuda
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CUDA_DYNAMIC_SYMBOLS_H_
