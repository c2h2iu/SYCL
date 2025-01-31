/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file group_functions.h
 *
 * @brief This file implements the group functions.
 */

#ifndef RUNTIME_INCLUDE_SYCL_GROUP_FUNCTIONS_H_
#define RUNTIME_INCLUDE_SYCL_GROUP_FUNCTIONS_H_

#include "SYCL/experimental/sub_group.h"
#include "SYCL/group.h"
#include "SYCL/memory_scope.h"
#include "SYCL/sycl_device_builtins.h"

#include "computecpp_export.h"

namespace cl {
namespace sycl {

#if SYCL_LANGUAGE_VERSION >= 2020

template <typename Group>
void group_barrier(Group grp, memory_scope fenceScope = Group::fence_scope);

template <>
void group_barrier<group<1>>(group<1> grp, memory_scope fenceScope) {
#ifdef __SYCL_DEVICE_ONLY__
  detail::barrier(
      detail::get_cl_mem_fence_flag(access::fence_space::global_and_local));
#else
  detail::host_barrier(grp);
#endif  // __SYCL_DEVICE_ONLY__
}

template <>
void group_barrier<group<2>>(group<2> grp, memory_scope fenceScope) {
#ifdef __SYCL_DEVICE_ONLY__
  detail::barrier(
      detail::get_cl_mem_fence_flag(access::fence_space::global_and_local));
#else
  detail::host_barrier(grp);
#endif  // __SYCL_DEVICE_ONLY__
}

template <>
void group_barrier<group<3>>(group<3> grp, memory_scope fenceScope) {
#ifdef __SYCL_DEVICE_ONLY__
  detail::barrier(
      detail::get_cl_mem_fence_flag(access::fence_space::global_and_local));
#else
  detail::host_barrier(grp);
#endif  // __SYCL_DEVICE_ONLY__
}

template <>
void group_barrier<experimental::sub_group>(experimental::sub_group subGrp,
                                            memory_scope fenceScope) {
#ifdef __SYCL_DEVICE_ONLY__
  detail::sub_group_barrier();
#endif  // __SYCL_DEVICE_ONLY__
}

#endif  // SYCL_LANGUAGE_VERSION >= 2020

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_GROUP_FUNCTIONS_H_

////////////////////////////////////////////////////////////////////////////////
