/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp.

*******************************************************************************/

/**
  @file aspect.h

  @brief This file contains the aspect enum class.
*/
#ifndef RUNTIME_INCLUDE_SYCL_ASPECT_H_
#define RUNTIME_INCLUDE_SYCL_ASPECT_H_

namespace cl {
namespace sycl {

/** @brief Enumerates the aspects which can be queried on a @ref platform or
 * @ref device.
 */
enum class aspect_impl {
  host,
  cpu,
  gpu,
  accelerator,
  custom,
  fp16,
  fp64,
  int64_base_atomics,
  int64_extended_atomics,
  image,
  online_compiler,
  online_linker,
  queue_profiling,
  usm_device_allocations,
  usm_host_allocations,
  usm_shared_allocations,
  usm_restricted_shared_allocations,
  usm_system_allocator,
};

#if SYCL_LANGUAGE_VERSION >= 2020

/** @brief Enumerates the aspects which can be queried on a @ref platform or
 * @ref device.
 */
using aspect = aspect_impl;

#endif  // SYCL_LANGUAGE_VERSION >= 2020

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ASPECT_H_
