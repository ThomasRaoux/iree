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

#include "iree/hal/cuda/direct_command_queue.h"

#include <cstdint>

#include "iree/base/tracing.h"
#include "iree/hal/cuda/direct_command_buffer.h"
#include "iree/hal/cuda/status_util.h"

namespace iree {
namespace hal {
namespace cuda {

DirectCommandQueue::DirectCommandQueue(
    CuDeviceHandle* logical_device, std::string name,
    iree_hal_command_category_t supported_categories, CUstream queue)
    : CommandQueue(logical_device, std::move(name), supported_categories,
                   queue) {}

DirectCommandQueue::~DirectCommandQueue() = default;

iree_status_t DirectCommandQueue::Submit(
    iree_host_size_t batch_count, const iree_hal_submission_batch_t* batches) {
  IREE_TRACE_SCOPE0("DirectCommandQueue::Submit");

  // Map the submission batches to VkSubmitInfos.
  // Note that we must keep all arrays referenced alive until submission
  // completes and since there are a bunch of them we use an arena.
  /*Arena arena(4 * 1024);
  auto submit_infos = arena.AllocateSpan<VkSubmitInfo>(batch_count);
  auto timeline_submit_infos =
      arena.AllocateSpan<VkTimelineSemaphoreSubmitInfo>(batch_count);
  for (int i = 0; i < batch_count; ++i) {
    IREE_RETURN_IF_ERROR(TranslateBatchInfo(&batches[i], &submit_infos[i],
                                            &timeline_submit_infos[i], &arena));
  }*/

  iree_slim_mutex_lock(&queue_mutex_);
  //iree_status_t status = CU_RESULT_TO_STATUS(
  //    syms()->vkQueueSubmit(queue_, static_cast<uint32_t>(submit_infos.size()),
  //                          submit_infos.data(), VK_NULL_HANDLE),
  //    "vkQueueSubmit");
  iree_slim_mutex_unlock(&queue_mutex_);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

iree_status_t DirectCommandQueue::WaitIdle(iree_time_t deadline_ns) {
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Fast path for using vkQueueWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#vkQueueWaitIdle");
    iree_slim_mutex_lock(&queue_mutex_);
 //   iree_status_t status =
 //       CU_RESULT_TO_STATUS(syms()->vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    iree_slim_mutex_unlock(&queue_mutex_);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

  IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#Fence");

  // Create a new fence just for this wait. This keeps us thread-safe as the
  // behavior of wait+reset is racey.
  /*VkFenceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  VkFence fence = VK_NULL_HANDLE;
  //CUDA_RETURN_IF_ERROR(
  //    syms()->vkCreateFence(*logical_device_, &create_info,
   //                         logical_device_->allocator(), &fence),
   //   "vkCreateFence");
*/
  uint64_t timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    // Do not wait.
    timeout_ns = 0;
  } else if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Wait forever.
    timeout_ns = UINT64_MAX;
  } else {
    // Convert to relative time in nanoseconds.
    // The implementation may not wait with this granularity (like by 10000x).
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns < now_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    timeout_ns = (uint64_t)(deadline_ns - now_ns);
  }

  iree_slim_mutex_lock(&queue_mutex_);
 // iree_status_t status = CU_RESULT_TO_STATUS(
 //     syms()->vkQueueSubmit(queue_, 0, nullptr, fence), "vkQueueSubmit");
  iree_slim_mutex_unlock(&queue_mutex_);

  /* if (iree_status_is_ok(status)) {
     CUresult result = syms()->vkWaitForFences(*logical_device_, 1, &fence,
                                               VK_TRUE, timeout_ns);
     switch (result) {
       case VK_SUCCESS:
         status = iree_ok_status();
         break;
       case VK_TIMEOUT:
         status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
         break;
       default:
         status = CU_RESULT_TO_STATUS(result, "vkWaitForFences");
         break;
     }*/
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

}  // namespace cuda
}  // namespace hal
}  // namespace iree
