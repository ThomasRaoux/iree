# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of various special logic throughout.


include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  message(STATUS "add cuda")
  enable_language(CUDA)
else()
  message(SEND_ERROR
    "Building cuda backend requires a working CUDA install")
endif()


iree_add_all_subdirs()
include(FindCUDAToolkit)
SET(LIBDEVICEPATH "${CUDAToolkit_LIBRARY_ROOT}/nvvm/libdevice/libdevice.10.bc")

# TODO(#5893): iree_c_embed_data requires a relative path and we need it to be
# here to keep the path consistent with bazel build file. Fix it by making cmake
# target more flexible.
iree_c_embed_data_rel(
  NAME
    cuda_libdevice
  SRCS
  "${LIBDEVICEPATH}"
  C_FILE_OUTPUT
    "libdevice.c"
  H_FILE_OUTPUT
    "libdevice.h"
  FLATTEN
  PUBLIC
)

iree_cc_library(
  NAME
    CUDA
  HDRS
    "CUDATarget.h"
    "LLVMPasses.h"
  SRCS
    "CUDATarget.cpp"
    "NoLoopUnrollPass.cpp"
  DEPS
    LLVMAnalysis
    LLVMBitReader
    LLVMCore
    LLVMipo
    LLVMLinker
    LLVMNVPTXCodeGen
    LLVMSupport
    LLVMTarget
    MLIRGPUOps
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMIR
    MLIRNVVMToLLVMIRTranslation
    MLIRPass
    MLIRSupport
    MLIRTargetLLVMIRExport
    iree::base::internal::flatcc::building
    iree::compiler::Codegen::LLVMGPU
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Utils
    iree::schemas::cuda_executable_def_c_fbs
  PUBLIC
)
