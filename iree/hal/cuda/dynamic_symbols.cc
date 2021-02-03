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

#include <cstddef>

#include "absl/base/attributes.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace cuda {

// Read-only table of function pointer information designed to be in .rdata.
// To reduce binary size this structure is packed (knowing that we won't have
// gigabytes of function pointers :).
struct FunctionPtrInfo {
  // Name of the function (like 'vkSomeFunction').
  const char* function_name;
};

namespace {

static const char* kCudaLoaderSearchNames[] = {
    "libcuda.so",
};

}  // namespace

// static
StatusOr<ref_ptr<DynamicSymbols>> DynamicSymbols::CreateFromSystemLoader() {
  IREE_TRACE_SCOPE0("DynamicSymbols::CreateFromSystemLoader");

  IREE_ASSIGN_OR_RETURN(
      auto loader_library,
      DynamicLibrary::Load(absl::MakeSpan(kCudaLoaderSearchNames)));
  auto syms = make_ref<DynamicSymbols>();
  syms->loader_library_ = std::move(loader_library);

  auto* loader_library_ptr = syms->loader_library_.get();

#define CU_PFN_DECL(cudaSymbolName)                                           \
  {                                                                           \
    using FuncPtrT = std::add_pointer<decltype(::cudaSymbolName)>::type;      \
    static const char* kName = #cudaSymbolName;                               \
    syms->cudaSymbolName = syms->loader_library_->GetSymbol<FuncPtrT>(kName); \
    if (!syms->cudaSymbolName) {                                              \
      return UnavailableErrorBuilder(IREE_LOC)                                \
             << "Required method " << kName << " not found in cuda library";  \
    }                                                                         \
  }

#include "dynamic_symbol_tables.def"
#undef CU_PFN_DECL

  return syms;
}

Status DynamicSymbols::LoadFromDevice(CUdevice device) {
  IREE_TRACE_SCOPE0("DynamicSymbols::LoadFromDevice");
  return OkStatus();
}

DynamicSymbols::DynamicSymbols() = default;

DynamicSymbols::~DynamicSymbols() = default;

}  // namespace cuda
}  // namespace hal
}  // namespace iree
