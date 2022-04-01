// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"

namespace mlir {
namespace iree_compiler {

constexpr unsigned kWorkgroupTileLevel = 0;
/// Pick the biggest case to not be limited by the verifier. If running on a
/// smaller machine the model may still report an error.
/// Currently A100 allows address up tp 163K per threadblock:
/// https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#sm-occupancy
constexpr int kSharedMemSizeBytes = 163 * 1024;

LogicalResult verifyGPUMatmulSimtPassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  auto pipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt;
  StringRef pipelineName = stringifyEnum(pipeline);
  if (workgroupSize.empty()) {
    return op->emitOpError("expected workgroup size for GPU pipelines");
  }

  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();  // Only verify batched and unbatched matmul.
  }

  Type inputType = op->getOperand(0).getType();
  SmallVector<int64_t> firstLevelTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);

  if (linalg::BatchMatmulOp batchMatmulOp =
          dyn_cast<linalg::BatchMatmulOp>(op)) {
    // Inspect first tile dimensions separately for batched. It should be 1 for
    // parallelizable loops and 0 for non-parallelizable. Continue with other
    // dimensions for remaining comparisons.
    if (cast<IREE::Flow::PartitionableLoopsInterface>(op).getPartitionableLoops(
            kNumMaxParallelDims)[0] == 0) {  // The first dimension is
      if (firstLevelTileSizes[0] > 1) {
        return op->emitError("Received first tile dimension of ")
               << firstLevelTileSizes[0] << " instead of 1 or 0 for "
               << pipelineName;
      }
    } else {
      if (firstLevelTileSizes[0] != 0) {
        return op->emitError("Received first tile dimension of ")
               << firstLevelTileSizes[0] << " instead of 0 for "
               << pipelineName;
      }
    }
    firstLevelTileSizes = {firstLevelTileSizes[1], firstLevelTileSizes[2],
                           firstLevelTileSizes[3]};
  }


  // Verify the total workgroup size is <= 1024
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > 1024) {
    return op->emitOpError("expected workgroup size to be <=1024 for ")
           << pipelineName << ", got " << totalWorkgroupSize;
  }

  // Verify the workgroup.z component should always be 1
  if (workgroupSize[2] != 1) {
    return op->emitOpError("expected workgroup z component to be 1 for ")
           << pipelineName << ", got " << workgroupSize[2];
  }

  // Verify shared memory usage of operands after tiling requires <= 64Kb
  // combined space.
  unsigned bytesSize =
      inputType.cast<ShapedType>().getElementType().getIntOrFloatBitWidth() / 8;

  // Input shape sizes: A [ M x K],  B [ K x N]
  unsigned totalSharedMemSizeBytes =
      (firstLevelTileSizes[0] * firstLevelTileSizes[2] +
       firstLevelTileSizes[1] * firstLevelTileSizes[2]) *
      bytesSize;

  if (totalSharedMemSizeBytes > kSharedMemSizeBytes) {
    return op->emitOpError("expected shared memory usage <= 64Kb for ")
           << pipelineName << ", got " << totalSharedMemSizeBytes;
  }

  // This pipeline doesn't use multi-buffering so we can pipeline at max with a
  // depth of 1.
  unsigned pipelineDepth = translationInfo.getSoftwarePipelineDepth();
  if (pipelineDepth > 1)
    return op->emitError("expected pipeline depth maximum of 1");

  return success();
}

LogicalResult verifyGPUMatmulTensorCorePipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  auto pipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulTensorCore;
  StringRef pipelineName = stringifyEnum(pipeline);
  if (workgroupSize.empty()) {
    return op->emitOpError("expected workgroup size for GPU pipelines");
  }

  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();  // Only verify batched and unbatched matmul.
  }

  Type inputType = op->getOperand(0).getType();
  ArrayRef<int64_t> lhsShape =
      op->getOperand(0).getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape =
      op->getOperand(1).getType().cast<ShapedType>().getShape();
  SmallVector<int64_t> firstLevelTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);

  if (linalg::BatchMatmulOp batchMatmulOp =
          dyn_cast<linalg::BatchMatmulOp>(op)) {
    // First dimension is the batch dimension. We don't check the shape batch.
    lhsShape = lhsShape.drop_front(1);
    rhsShape = rhsShape.drop_front(1);

    // Inspect first tile dimensions separately for batched. It should be 1 for
    // parallelizable loops and 0 for non-parallelizable. Continue with other
    // dimensions for remaining comparisons.
    if (cast<IREE::Flow::PartitionableLoopsInterface>(op).getPartitionableLoops(
            kNumMaxParallelDims)[0] == 0) {  // The first dimension is
      if (firstLevelTileSizes[0] > 1) {
        return op->emitError("Received first tile dimension of ")
               << firstLevelTileSizes[0] << " instead of 1 or 0 for "
               << pipelineName;
      }
    } else {
      if (firstLevelTileSizes[0] != 0) {
        return op->emitError("Received first tile dimension of ")
               << firstLevelTileSizes[0] << " instead of 0 for "
               << pipelineName;
      }
    }
    firstLevelTileSizes = {firstLevelTileSizes[1], firstLevelTileSizes[2],
                           firstLevelTileSizes[3]};
  }

  // Verify the total workgroup size is <= 1024
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > 1024) {
    return op->emitOpError("expected workgroup size to be <=1024 for ")
           << pipelineName << ", got " << totalWorkgroupSize;
  }

  // Verify that the workgroup X dimension is 32 aligned
  if (workgroupSize[0] % 32 != 0) {
    return op->emitOpError("workgroup size is not 32 aligned for ")
           << pipelineName << ", got " << workgroupSize[0];
  }

  // Verify the workgroup.z component should always be 1
  if (workgroupSize[2] != 1) {
    return op->emitOpError("expected workgroup z component to be 1 for ")
           << pipelineName << ", got " << workgroupSize[2];
  }

  // The second level of tiling = [M / numWarp.y, N / numWarp.x, K].
  SmallVector<int64_t, 3> secondLevelTileSizes = {
      firstLevelTileSizes[0] / workgroupSize[1],
      firstLevelTileSizes[1] / (workgroupSize[0] / kWarpSize),
      firstLevelTileSizes[2]};

  // Verify the TensorCore size divides the second level tile size
  SmallVector<int64_t, 3> tensorCoreSize({16, 16, 8});
  if (secondLevelTileSizes[0] % tensorCoreSize[0] != 0 ||
      secondLevelTileSizes[1] % tensorCoreSize[1] != 0 ||
      secondLevelTileSizes[2] % tensorCoreSize[2] != 0) {
    return op->emitOpError(
               "tensorcore size doesn't factor into second level tile size "
               "for ")
           << pipelineName;
  }

  // Verify the first level tile size divides the matmul
  // inputs A [M x K] & B [K x N]
  if (lhsShape[0] % firstLevelTileSizes[0] != 0 ||
      lhsShape[1] % firstLevelTileSizes[2] != 0) {
    return op->emitOpError(
               "lhsShape doesn't factor into first level tile size for ")
           << pipelineName << " [ " << lhsShape[0] << ", " << lhsShape[1]
           << "]";
  }
  if (rhsShape[0] % firstLevelTileSizes[2] != 0 ||
      rhsShape[1] % firstLevelTileSizes[1] != 0) {
    return op->emitOpError(
               "rhsShape doesn't factor into first level tile size for ")
           << pipelineName << " [ " << rhsShape[0] << ", " << rhsShape[1]
           << "]";
  }

  // This pipeline doesn't use multi-buffering so we can pipeline at max with a
  // depth of 1.
  unsigned pipelineDepth = translationInfo.getSoftwarePipelineDepth();
  if (pipelineDepth >= lhsShape[1] / firstLevelTileSizes[2])
    return op->emitError(
        "expected pipeline to be smaller than K dimensions divided by tileK");

  // Verify shared memory usage of operands after tiling requires <= 64Kb
  // combined space.
  unsigned bytesSize =
      inputType.cast<ShapedType>().getElementType().getIntOrFloatBitWidth() / 8;

  // Input shape sizes: A [ M x K],  B [ K x N]
  unsigned totalSharedMemSizeBytes =
      (firstLevelTileSizes[0] * firstLevelTileSizes[2] +
       firstLevelTileSizes[1] * firstLevelTileSizes[2]) *
      bytesSize;
  // When pipelining with a depth greater than 1 we need to multibuffer the
  // shared memory. Therefore we need to consider it when calculating the amount
  // of shared memory used.
  unsigned numberOfBuffers = std::max<unsigned>(1, pipelineDepth);
  if (totalSharedMemSizeBytes * numberOfBuffers > kSharedMemSizeBytes) {
    return op->emitOpError("expected shared memory usage <= 64Kb for ")
           << pipelineName << ", got " << totalSharedMemSizeBytes;
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
