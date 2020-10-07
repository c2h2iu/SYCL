/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file group.h

  @brief This file contains the API for @ref cl::sycl::group
*/
#ifndef RUNTIME_INCLUDE_SYCL_GROUP_H_
#define RUNTIME_INCLUDE_SYCL_GROUP_H_

#include "SYCL/common.h"
#include "SYCL/device_event.h"
#include "SYCL/group_base.h"
#include "SYCL/item_base.h"
#include "SYCL/memory_scope.h"
#include "SYCL/multi_pointer.h"
#include "computecpp/gsl/gsl"

#ifdef __SYCL_DEVICE_ONLY__
#include "SYCL/compiler_hooks.h"  // for kernelgen_* functions
#endif

namespace cl {
namespace sycl {

/** @brief The cl::sycl::group object is a container for all information about a
 * work
 * group. The cl::sycl::group object is used within the parallel_for_work_group
 * function. The cl::sycl::group object can return information about the local
 * and global sizes of an enqueued nd-range as well as the number of groups and
 * the current group id.
 */
template <int dimensions = 1>
class group : public detail::group_base {
 protected:
  /** @cond COMPUTECPP_DEV */

  using base_t = detail::group_base;

  /** @brief This constructor should not be called.
   */
  group() = delete;

  /** @brief Constructor of the group class.
   * This constructor should only be called from the runtime.
   */
  group(id<dimensions> groupID, id<dimensions> groupRange,
        range<dimensions> globalRange)
      : detail::group_base(
            groupID, groupRange, globalRange,
            detail::index_array((globalRange[0] / groupRange[0]),
                                (globalRange[1] / groupRange[1]),
                                (globalRange[2] / groupRange[2]))) {}

  /** COMPUTECPP_DEV @endcond */

  /** @brief Checks if the ID of this group is all zeros
   * @return True if current ID is (0, 0, 0)
   */
  bool is_zero_id() const {
    const auto id = this->get_id();
    return ((id[0] == 0) && (id[1] == 0) && (id[2] == 0));
  }

 public:
#if SYCL_LANGUAGE_VERSION >= 2020

  static constexpr memory_scope fence_scope = memory_scope::work_group;

#endif  // SYCL_LANGUAGE_VERSION >= 2020

  /** @brief Conversion constructor from group<dimensions> to group_base
   * This constructor is used in all of the kernel invocation APIs that
   * use the group class.
   */
  group(const detail::group_base& gb)
      : detail::group_base(gb) {}  // NOLINT + false

  /** @brief Equality operator
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  friend inline bool operator==(const group& lhs, const group& rhs) {
    return lhs.is_equal<dimensions>(rhs);
  }

  /** @brief Non-equality operator
   * @param rhs Object to compare to
   * @return True if at least one member variable not the same
   *         as the corresponding rhs member variable
   */
  friend inline bool operator!=(const group& lhs, const group& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Get Group ID
   * @return the group id for all dimensions of the nd_range
   * @deprecated Use get_id instead
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(2017, "Use group::get_id() instead.")
  id<dimensions> get() const { return get_id(); }

  /** @brief Get Group ID
   * @param the dimension of the nd_range we should return the group ID for
   * @return the group id for that dimension
   * @deprecated Use get_id instead
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(2017,
                                    "Use group::get_id(unsigned) instead.")
  size_t get(unsigned int dimension) const { return get_id(dimension); }

  /** @brief Get Group ID
   * @return the group id for all dimensions of the nd_range
   */
  id<dimensions> get_id() const { return m_groupID; }

  /** @brief Get Group ID
   * @param the dimension of the nd_range we should return the group ID for
   * @return the group id for that dimension
   */
  size_t get_id(unsigned int dimension) const { return get_id()[dimension]; }

  /** @brief Get the global range for all dimensions
   * @return the value of the global range for each dimension of the nd_range.
   */
  range<dimensions> get_global_range() const {
    return range<dimensions>(this->m_globalRange);
  }

  /** @brief Returns the global range in a specified dimension.
   * @param dimension the dimension of the global range to be returned.
   * @return the value of the global range in the specified dimension.
   */
  size_t get_global_range(unsigned int dimension) const {
    return get_global_range()[dimension];
  }

  /** @brief Get the local range for all dimensions
   * @return the value of the local range for each dimension of the nd_range.
   */
  range<dimensions> get_local_range() const {
    return range<dimensions>(this->m_localRange);
  }

  /** @brief Returns the local range in a specified dimension.
   * @param dimension the dimension of the local range to be returned.
   * @return the value of the local range in the specified dimension.
   */
  size_t get_local_range(unsigned int dimension) const {
    return get_local_range()[dimension];
  }

  /** @briefGet the group range for all dimensions
   * @return the value of the group range for each dimension of the group.
   */
  range<dimensions> get_group_range() const {
    return range<dimensions>(this->m_groupRange);
  }

  /** @brief Returns the group range in a specified dimension.
   * @param dimension the dimension of the group range to be returned.
   * @return the value of the group range in the specified dimension.
   */
  size_t get_group_range(unsigned int dimension) const {
    return get_group_range()[dimension];
  }

  /** @brief Get the value for the given dimension
   * @return size_t with the value for the given dimension
   */
  size_t operator[](size_t dim) const {
    return this->get_id(static_cast<unsigned int>(dim));
  }

  /** @brief Waits on each given device_event
   * @tparam eventTN Pack of device_event types
   * @param events Pack of device_events
   */
  template <typename... eventTN>
  void wait_for(eventTN... events) const {
    static_assert(computecpp::gsl::conjunction<
                      std::is_same<cl::sycl::device_event, eventTN>...>::value,
                  "All events must be of type device_event");
    auto eventList = {events...};
    for (auto& event : eventList) {
      event.wait();
    }
  }

#ifndef __SYCL_DEVICE_ONLY__

  /** @brief Inner loop of parallel_for_work_group
   * @tparam workItemFunctionT Type of the functor to execute hierarchically
   * @param func The functor to execute
   */
  template <typename workItemFunctionT>
  void parallel_for_work_item(const workItemFunctionT& func) const {
    this->parallel_for_work_item(this->get_local_range(), func);
  }

  /** @brief Inner loop of parallel_for_work_group
   * @tparam workItemFunctionT Type of the functor to execute hierarchically
   * @param flexibleRange The logical local range
   * @param func The functor to execute
   */
  template <typename workItemFunctionT>
  void parallel_for_work_item(range<dimensions> flexibleRange,
                              const workItemFunctionT& func) const {
    const auto globalRange = this->get_global_range();
    const auto physicalLocalRange = this->get_local_range();
    const auto groupID = this->get_id();
    const auto globalIdBase = physicalLocalRange * groupID;

    id<dimensions> physicalLocalID;

    for (size_t itemZ = 0; itemZ < flexibleRange[2]; itemZ++) {
      physicalLocalID[2] = itemZ % physicalLocalRange[2];
      for (size_t itemY = 0; itemY < flexibleRange[1]; itemY++) {
        physicalLocalID[1] = itemY % physicalLocalRange[1];
        for (size_t itemX = 0; itemX < flexibleRange[0]; itemX++) {
          physicalLocalID[0] = itemX % physicalLocalRange[0];

          id<dimensions> localID(detail::index_array(itemX, itemY, itemZ));
          id<dimensions> globalID = globalIdBase + physicalLocalID;

          h_item<dimensions> currentItem(detail::h_item_base(
              detail::item_base(localID, flexibleRange),
              detail::item_base(physicalLocalID, physicalLocalRange),
              detail::item_base(globalID, globalRange)));

          func(currentItem);
        }
      }
    }
  }
#else  // __SYCL_DEVICE_ONLY__

  template <typename workItemFunctionT>
  void parallel_for_work_item(const workItemFunctionT& func) const {
    // NOT ACTUALLY CALLED ONLY HERE TO CALL THE KERNEL GEN FUNCTION
    detail::kernelgen_parallel_for_work_item<dimensions, workItemFunctionT>(
        *this, func);
  }

  template <typename workItemFunctionT>
  void parallel_for_work_item(range<dimensions> flexibleRange,
                              const workItemFunctionT& func) const {
    // NOT ACTUALLY CALLED ONLY HERE TO CALL THE KERNEL GEN FUNCTION
    detail::kernelgen_parallel_for_work_item<dimensions, workItemFunctionT>(
        *this, flexibleRange, func);
  }

#endif  // __SYCL_DEVICE_ONLY__

  /** @brief Asynchronous work group copy from a global pointer to local.
   * @tparam dataT Data type of the pointer
   * @param dest Pointer to the destination in local memory
   * @param src Pointer to the source in global memory
   * @param numElements Number of elements to copy
   */
  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src,
                                     size_t numElements) const {
#ifdef __SYCL_DEVICE_ONLY__
    __sycl_event_t previousEvent = 0;
    return device_event(
        detail::async_work_group_copy(dest, src, numElements, previousEvent));
#else   // __SYCL_DEVICE_ONLY__
    static constexpr size_t stride = 1;
    return this->async_work_group_copy(dest, src, numElements, stride);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Asynchronous work group copy from a local pointer to global.
   * @tparam dataT Data type of the pointer
   * @param dest Pointer to the destination in global memory
   * @param src Pointer to the source in local memory
   * @param numElements Number of elements to copy
   */
  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src,
                                     size_t numElements) const {
#ifdef __SYCL_DEVICE_ONLY__
    __sycl_event_t previousEvent = 0;
    return device_event(
        detail::async_work_group_copy(dest, src, numElements, previousEvent));
#else   // __SYCL_DEVICE_ONLY__
    static constexpr size_t stride = 1;
    return this->async_work_group_copy(dest, src, numElements, stride);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Asynchronous work group copy from a global pointer to local.
   * @tparam dataT Data type of the pointer
   * @param dest Pointer to the destination in local memory
   * @param src Pointer to the source in global memory
   * @param numElements Number of elements to copy
   * @param destStride Stride in the origin
   */
  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements,
                                     size_t srcStride) const {
#ifdef __SYCL_DEVICE_ONLY__
    __sycl_event_t previousEvent = 0;
    return device_event(detail::async_work_group_strided_copy(
        dest, src, numElements, srcStride, previousEvent));
#else   // __SYCL_DEVICE_ONLY__
    if (this->is_zero_id()) {
      size_t srcIter = 0;
      for (size_t i = 0; i < numElements; i++) {
        *(dest + i) = *(src + srcIter);
        srcIter += srcStride;
      }
    }
    return device_event();
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Asynchronous work group copy from a local pointer to global.
   * @tparam dataT Data type of the pointer
   * @param dest Pointer to the destination in global memory
   * @param src Pointer to the source in local memory
   * @param numElements Number of elements to copy
   * @param destStride Stride in the destination
   */
  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements,
                                     size_t destStride) const {
#ifdef __SYCL_DEVICE_ONLY__
    __sycl_event_t previousEvent = 0;
    return device_event(detail::async_work_group_strided_copy(
        dest, src, numElements, destStride, previousEvent));
#else   // __SYCL_DEVICE_ONLY__
    if (this->is_zero_id()) {
      size_t dstIter = 0;
      for (size_t i = 0; i < numElements; i++) {
        *(dest + dstIter) = *(src + i);
        dstIter += destStride;
      }
    }
    return device_event();
#endif  // __SYCL_DEVICE_ONLY__
  }
};

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_GROUP_H_
