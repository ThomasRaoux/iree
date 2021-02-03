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

#include "iree/hal/cuda/cuda_driver.h"

#include <memory>

#include "iree/base/memory.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/api.h"
#include "iree/hal/cuda/debug_reporter.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/extensibility_util.h"
#include "iree/hal/cuda/status_util.h"
#include "iree/hal/cuda/cuda_device.h"

using namespace iree::hal::cuda;

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple Cuda versions can be exposed in the
  // same process.
  iree_string_view_t identifier;

  iree_hal_cuda_device_options_t device_options;
  int default_device_index;

  iree_hal_cuda_features_t enabled_features;

  // (Partial) loaded Cuda symbols. Devices created within the driver may have
  // different function pointers for device-specific functions that change
  // behavior with enabled layers/extensions.
  iree::ref_ptr<DynamicSymbols> syms;

  // Optional debug reporter: may be disabled or unavailable (no debug layers).
  iree_hal_cuda_debug_reporter_t* debug_reporter;
} iree_hal_cuda_driver_t;

// Pick a fixed lenght size for device names.
typedef char cude_device_name_t[100];

extern const iree_hal_driver_vtable_t iree_hal_cuda_driver_vtable;

static iree_hal_cuda_driver_t* iree_hal_cuda_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_driver_vtable);
  return (iree_hal_cuda_driver_t*)base_value;
}

IREE_API_EXPORT void IREE_API_CALL iree_hal_cuda_driver_options_initialize(
    iree_hal_cuda_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->api_version = CUDA_VERSION;
  out_options->requested_features = 0;
  iree_hal_cuda_device_options_initialize(&out_options->device_options);
  out_options->default_device_index = 0;
}

static iree_status_t iree_hal_cuda_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_cuda_driver_options_t* options,
    const iree_hal_cuda_string_list_t* enabled_extensions,
    iree_hal_cuda_syms_t* opaque_syms,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  auto* syms = (DynamicSymbols*)opaque_syms;

  iree_hal_cuda_debug_reporter_t* debug_reporter = NULL;

  iree_hal_cuda_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver);
  if (!iree_status_is_ok(status)) {
    // Need to clean up if we fail (as we own these).
    iree_hal_cuda_debug_reporter_free(debug_reporter);
    return status;
  }
  iree_hal_resource_initialize(&iree_hal_cuda_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  memcpy(&driver->device_options, &options->device_options,
         sizeof(driver->device_options));
  driver->default_device_index = options->default_device_index;
  driver->enabled_features = options->requested_features;
  driver->syms = iree::add_ref(syms);
  driver->debug_reporter = debug_reporter;
  *out_driver = (iree_hal_driver_t*)driver;
  return status;
}

static void iree_hal_cuda_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_debug_reporter_free(driver->debug_reporter);

  driver->syms.reset();
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_driver_query_extensibility_set(
    iree_hal_cuda_features_t requested_features,
    iree_hal_cuda_extensibility_set_t set, iree::Arena* arena,
    iree_hal_cuda_string_list_t* out_string_list) {
  IREE_RETURN_IF_ERROR(iree_hal_cuda_query_extensibility_set(
      requested_features, set, 0, NULL, &out_string_list->count));
  out_string_list->values = (const char**)arena->AllocateBytes(
      out_string_list->count * sizeof(out_string_list->values[0]));
  IREE_RETURN_IF_ERROR(iree_hal_cuda_query_extensibility_set(
      requested_features, set, out_string_list->count, out_string_list->values,
      &out_string_list->count));
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_driver_compute_enabled_extensibility_sets(
    iree::hal::cuda::DynamicSymbols* syms,
    iree_hal_cuda_features_t requested_features, iree::Arena* arena,
    iree_hal_cuda_string_list_t* out_enabled_layers,
    iree_hal_cuda_string_list_t* out_enabled_extensions) {

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "not implemented yet for CUDA");
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_cuda_driver_create(
    iree_string_view_t identifier,
    const iree_hal_cuda_driver_options_t* options,
    iree_hal_cuda_syms_t* opaque_syms, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(opaque_syms);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_SCOPE();

  auto* syms = (DynamicSymbols*)opaque_syms;

  // Query required and optional instance layers/extensions for the requested
  // features.
  iree::Arena arena;
  iree_hal_cuda_string_list_t enabled_layers;
  iree_hal_cuda_string_list_t enabled_extensions;
//  IREE_RETURN_IF_ERROR(
//      iree_hal_cuda_driver_compute_enabled_extensibility_sets(
//          syms, options->requested_features, &arena, &enabled_layers,
//          &enabled_extensions));

  iree_status_t status = iree_hal_cuda_driver_create_internal(
        identifier, options, &enabled_extensions, opaque_syms, host_allocator,
        out_driver);

  return status;
}

// Populates device information from the given Cuda physical device handle.
// |out_device_info| must point to valid memory and additional data will be
// appended to |buffer_ptr| and the new pointer is returned.
static uint8_t* iree_hal_cuda_populate_device_info(
    CUdevice physical_device, DynamicSymbols* syms, uint8_t* buffer_ptr,
    iree_hal_device_info_t* out_device_info) {
  cude_device_name_t device_name;
  CUDA_CHECK_OK(
      syms, cuDeviceGetName(device_name, sizeof(device_name), physical_device));
  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = (iree_hal_device_id_t)physical_device;

  iree_string_view_t device_name_string =
      iree_make_string_view(device_name, strlen(device_name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_string, &out_device_info->name, (char*)buffer_ptr);
  return buffer_ptr;
}

static iree_status_t iree_hal_cuda_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);
  DynamicSymbols* syms = driver->syms.get();
  // Query the number of available CUDA devices.
  int physical_device_count = 0;
  CUDA_RETURN_IF_ERROR(syms, cuDeviceGetCount(&physical_device_count),
                       "cuDeviceGetCount");

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size =
      physical_device_count * sizeof(iree_hal_device_info_t);
  for (int i = 0; i < physical_device_count; ++i) {
    total_size += sizeof(cude_device_name_t);
  }
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos +
        physical_device_count * sizeof(iree_hal_device_info_t);
    for (int i = 0; i < physical_device_count; ++i) {
      CUdevice device;
      CUDA_RETURN_IF_ERROR(syms, cuDeviceGet(&device, i), "cuDeviceGet");
      buffer_ptr = iree_hal_cuda_populate_device_info(
          device, driver->syms.get(), buffer_ptr, &device_infos[i]);
    }
    *out_device_info_count = physical_device_count;
    *out_device_infos = device_infos;
  }
  return status;
}

static iree_status_t iree_hal_cuda_driver_select_default_device(
    iree::hal::cuda::DynamicSymbols* syms, int default_device_index,
    iree_allocator_t host_allocator, CUdevice* out_device) {
  int device_count = 0;
  CUDA_RETURN_IF_ERROR(syms, cuDeviceGetCount(&device_count),
                       "cuDeviceGetCount");
  iree_status_t status = iree_ok_status();
  if (device_count == 0 || default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %d enumerated)",
                              default_device_index, device_count);
  } else {
    CUdevice device;
    CUDA_RETURN_IF_ERROR(syms, cuDeviceGet(&device, default_device_index),
                         "cuDeviceGet");
    *out_device = device;
  }
  return status;
}

static iree_status_t iree_hal_cuda_driver_create_device(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);
  
  CUDA_RETURN_IF_ERROR(driver->syms.get(), cuInit(0), "cuInit");
  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  CUdevice physical_device = (CUdevice)device_id;
  if (physical_device == 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_cuda_driver_select_default_device(
            driver->syms.get(), driver->default_device_index,
            host_allocator, &physical_device));
  }

  iree_string_view_t device_name = iree_make_cstring_view("cuda");

  // Attempt to create the device.
  // This may fail if the device was enumerated but is in exclusive use,
  // disabled by the system, or permission is denied.
  iree_status_t status = iree_hal_cuda_device_create(
      base_driver, device_name, driver->enabled_features,
      &driver->device_options, (iree_hal_cuda_syms_t*)driver->syms.get(),
      physical_device, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_driver_vtable_t iree_hal_cuda_driver_vtable = {
    /*.destroy=*/iree_hal_cuda_driver_destroy,
    /*.query_available_devices=*/
    iree_hal_cuda_driver_query_available_devices,
    /*.create_device=*/iree_hal_cuda_driver_create_device,
};
