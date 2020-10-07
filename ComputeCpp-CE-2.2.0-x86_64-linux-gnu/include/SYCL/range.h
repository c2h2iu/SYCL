/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

******************************************************************************/

/** @file range.h
 *
 * @brief This file implements the \ref cl::sycl::range class as defined by the
 * SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_RANGE_H_
#define RUNTIME_INCLUDE_SYCL_RANGE_H_

#include "SYCL/common.h"
#include "SYCL/index_array.h"
#include "SYCL/index_array_operators.h"
#include "SYCL/info.h"
#include "SYCL/nd_range_base.h"

namespace cl {
namespace sycl {

/** @brief range representation of a 1, 2 or 3 dimensional range.
 * A range represents the size of each dimension of the index space.
 * @tparam dimensions The range dimension, dimensions must be 1, 2 or 3
 */
template <int dimensions = 1>
class range
    : public detail::index_array_operators<dimensions, cl::sycl::range> {
  template <int nd>
  friend class nd_range;

 public:
  static_assert(
      (dimensions > 0 && dimensions < 4),
      "The allowed dimensionality is within the input range of [1,3].");
};

#if SYCL_LANGUAGE_VERSION >= 2020

/** Deduction guide for range class template.
 */

range(size_t)->range<1>;

range(size_t, size_t)->range<2>;

range(size_t, size_t, size_t)->range<3>;

#endif  // SYCL_LANGUAGE_VERSION >= 2020

/** @brief 1 dimension range definition.
 */
template <>
class range<1> : public detail::index_array_operators<1, cl::sycl::range> {
 public:
  /** @brief Default constructor. Initialize the range to 1.
   * Equivalent to range<1>(1)
   */
  range() : detail::index_array_operators<1, cl::sycl::range>(1, 1, 1) {}
  /** @brief Create a copy of a range.
   * @param rhs The range to copy
   */
  range(const range& rhs) = default;
  /** @brief Create a 1 dimension range initialized to dim1.
   * @param dim1 The size of the range.
   */
  range(size_t dim1)
      : detail::index_array_operators<1, cl::sycl::range>(dim1, 1, 1) {}
  /// @cond COMPUTECPP_DEV
  explicit range(
      const detail::index_array_operators<1, cl::sycl::range>& indexArray)
      : detail::index_array_operators<1, cl::sycl::range>(indexArray) {}
  range(const detail::index_array& indexArray) : range<1>(indexArray[0]) {}
  /// COMPUTECPP_DEV @endcond

  /** @brief Return the size of the range
   * @return the range size.
   */
  size_t size() const { return m_idx[0]; }
};

/** @brief 2 dimension range definition.
 */
template <>
class range<2> : public detail::index_array_operators<2, cl::sycl::range> {
 public:
  /** @brief Default constructor. Initialize the range to 1.
   * Equivalent to range<2>(1, 1)
   */
  range() : detail::index_array_operators<2, cl::sycl::range>(1, 1, 1) {}
  /** @brief Create a copy of a range.
   * @param rhs The range to copy
   */
  range(const range& rhs) = default;
  /** @brief Create a 2 dimension range initialized to dim1 for the first
   * dimension and dim2 for the second.
   * @param dim1 The size of the first dimension.
   * @param dim2 The size of the second dimension.
   */
  range(size_t dim1, size_t dim2)
      : detail::index_array_operators<2, cl::sycl::range>(dim1, dim2, 1) {}
  /// @cond COMPUTECPP_DEV
  explicit range(
      const detail::index_array_operators<2, cl::sycl::range>& indexArray)
      : detail::index_array_operators<2, cl::sycl::range>(indexArray) {}
  range(const detail::index_array& indexArray)
      : range<2>(indexArray[0], indexArray[1]) {}
  /// COMPUTECPP_DEV @endcond
  /** @brief Return the size of the range
   * @return the range size.
   */
  size_t size() const { return m_idx[0] * m_idx[1]; }
};

/** @brief 3 dimension range definition.
 */
template <>
class range<3> : public detail::index_array_operators<3, cl::sycl::range> {
 public:
  /** @brief Default constructor. Initialize the range to 1.
   * Equivalent to range<3>(1, 1, 1)
   */
  range() : detail::index_array_operators<3, cl::sycl::range>(1, 1, 1) {}
  /** @brief Create a copy of a range.
   * @param rhs The range to copy
   */
  range(const range& rhs) = default;
  /** @brief Create a 3 dimension range initialized to dim1 for the first
   * dimension, dim2 for the second and dim3 for the third.
   * @param dim1 The size of the first dimension.
   * @param dim2 The size of the second dimension.
   * @param dim3 The size of the third dimension.
   */
  range(size_t dim1, size_t dim2, size_t dim3)
      : detail::index_array_operators<3, cl::sycl::range>(dim1, dim2, dim3) {}
  /// @cond COMPUTECPP_DEV
  explicit range(
      const detail::index_array_operators<3, cl::sycl::range>& indexArray)
      : detail::index_array_operators<3, cl::sycl::range>(indexArray) {}
  range(const detail::index_array& indexArray)
      : range<3>(indexArray[0], indexArray[1], indexArray[2]) {}
  /// COMPUTECPP_DEV @endcond

  /** @brief Return the size of the range
   * @return the range size.
   */
  size_t size() const { return m_idx[0] * m_idx[1] * m_idx[2]; }
};

/**
@brief Specialization of info_convert for converting a pointer size_t type to a
range<3> type.
@ref cl::sycl::info_convert
*/
template <>
struct info_convert<size_t*, range<3>> {
  static range<3> cl_to_sycl(size_t* clPtr, size_t numElems, cl_uint clParam) {
    if (numElems != 3) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          CL_SUCCESS, detail::cpp_error_code::TARGET_FORMAT_ERROR, nullptr,
          "Unable to convert size_t[X] to range<3> because X != 3")
    }
    range<3> syclResult(clPtr[0], clPtr[1], clPtr[2]);
    return syclResult;
  }
};

/** @brief Implements the nd_range class of the SYCL specification.
 * An nd_range contains a global and a local range and an offset.
 */
template <int dimensions = 1>
class nd_range : public detail::nd_range_base {
  using base_t = detail::nd_range_base;

 public:
  static_assert(
      (dimensions > 0 && dimensions < 4),
      "The allowed dimensionality is within the input range of [1,3].");
  /** @brief Construct a nd_range object specifying the global and local range
   * and an optional offset. Note that the global range must divisible by the
   * local range in order to be usable by a \ref handler::parallel_for.
   * @param globalRange The global \ref range
   * @param localRange The local \ref range
   * @param globalOffset The global offset (optional, default to 0)
   */
  nd_range(const range<dimensions> globalRange,
           const range<dimensions> localRange,
           const id<dimensions> globalOffset = id<dimensions>())
      : detail::nd_range_base(globalRange, localRange, globalOffset) {}
  /** @brief Copy constructor. Create a copy of another nd_range.
   * @param ndRangeBase The nd_range to copy
   */
  nd_range(
      const detail::nd_range_base& ndRangeBase)  // NOLINT false +, conversion
      : detail::nd_range_base(ndRangeBase) {}

  /** \brief Return the global \ref range
   * @return The global range
   */
  COMPUTECPP_DEPRECATED_API(
      "SYCL 1.2.1 revision 3 replaces nd_range::get_global with "
      "nd_range::get_global_range.")
  range<dimensions> get_global() const { return this->get_global_range(); }

  /** \brief Return the global \ref range
   * @return The global range
   */
  range<dimensions> get_global_range() const {
    return base_t::get_global_range();
  }
  /** \brief Return the local \ref range
   * @return The local range
   */
  COMPUTECPP_DEPRECATED_API(
      "SYCL 1.2.1 revision 3 replaces nd_range::get_local with "
      "nd_range::get_local_range.")
  range<dimensions> get_local() const { return this->get_local_range(); }

  /** \brief Return the local \ref range
   * @return The local range
   */
  range<dimensions> get_local_range() const {
    return base_t::get_local_range();
  }

  /** \brief Compute the group \ref range
   * @return The group range
   */
  COMPUTECPP_DEPRECATED_API(
      "SYCL 1.2.1 revision 3 replaces nd_range::get_group with "
      "nd_range::get_group_range.")
  range<dimensions> get_group() const { return base_t::get_group_range(); }

  /** \brief Compute the group \ref range
   * @return The group range
   */
  range<dimensions> get_group_range() const {
    return base_t::get_group_range();
  }

  /** \brief Return the queue offset
   * @return The offset
   */
  id<dimensions> get_offset() const { return base_t::get_offset(); }

  /** @brief Equality operator
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  friend inline bool operator==(const nd_range& lhs, const nd_range& rhs) {
    return lhs.is_equal<dimensions>(rhs);
  }

  /** @brief Non-equality operator
   * @param rhs Object to compare to
   * @return True if at least one member variable not the same
   *         as the corresponding rhs member variable
   */
  friend inline bool operator!=(const nd_range& lhs, const nd_range& rhs) {
    return !(lhs == rhs);
  }
};

}  // namespace sycl
}  // namespace cl
#endif  // RUNTIME_INCLUDE_SYCL_RANGE_H_
