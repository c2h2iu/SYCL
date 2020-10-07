/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file id.h
 *
 * @brief This file implement the @ref cl::sycl::id class as defined by the SYCL
 * 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ID_H_
#define RUNTIME_INCLUDE_SYCL_ID_H_

#include "SYCL/common.h"
#include "SYCL/index_array_operators.h"
#include "SYCL/item_base.h"
#include "SYCL/range.h"

namespace cl {
namespace sycl {

/** @brief base class which the other versions implements specialisations of
 */
template <int dimensions = 1>
class id : public detail::index_array_operators<dimensions, cl::sycl::id> {
 public:
  static_assert(
      (dimensions > 0 && dimensions < 4),
      "The allowed dimensionality is within the input range of [1,3].");
};

#if SYCL_LANGUAGE_VERSION >= 2020

/** Deduction guide for id class template.
 */

id(size_t)->id<1>;

id(size_t, size_t)->id<2>;

id(size_t, size_t, size_t)->id<3>;

#endif  // SYCL_LANGUAGE_VERSION >= 2020

/** @brief Implementation of id for 1 dimension
 */
template <>
class id<1> : public detail::index_array_operators<1, cl::sycl::id> {
 public:
  /** @cond COMPUTECPP_DEV */

  /** @brief Constructor taking in a item
   */
  id(const detail::item_base& itemBase)
      : detail::index_array_operators<1, cl::sycl::id>(
            itemBase.get_id(0), itemBase.get_id(1), itemBase.get_id(2)) {}
  /** COMPUTECPP_DEV @endcond */

  /** @brief Default constructor for id, initialized to { 0, 0, 0 }
   */
  id() : detail::index_array_operators<1, cl::sycl::id>(0, 0, 0) {}

  /** @brief Default copy constructor for id.
   */
  id(const id& rhs) = default;

  /** @brief Create a one dimensional id from a single size_t parameter.
   */
  id(size_t x) : detail::index_array_operators<1, cl::sycl::id>(x, 0, 0) {}

  /** @cond COMPUTECPP_DEV */

  /** @brief copy constructor from the templated operator interface class used
   * internally to the public templated class.
   */
  explicit id(const detail::index_array_operators<1, cl::sycl::id>&
                  indexArray)  // NOLINT false +,  conversion
      : detail::index_array_operators<1, cl::sycl::id>(indexArray) {}

  /** @brief copy constructor from the non templated base class used internally
   * to the public templated class.
   */
  id(const detail::index_array& indexArray) : id<1>(indexArray[0]) {}

  /** @brief copy constructor from the non templated base class used internally
   * to the public templated class.
   */
  explicit id(const item<1, true>& it)  // NOLINT false +,  conversion
      : id(*reinterpret_cast<const detail::item_base*>(&it)) {}

  /** COMPUTECPP_DEV @endcond */

  /** @brief conversion from range to id constructor.
   */
  id(const range<1>& r)  // NOLINT false +,  conversion
      : id(r[0]) {}
};

/** @brief Implementation of id for 2 dimensions
 */
template <>
class id<2> : public detail::index_array_operators<2, cl::sycl::id> {
 public:
  /** @cond COMPUTECPP_DEV */

  /** @brief Constructor taking in a item
   */
  id(const detail::item_base& itemBase)
      : detail::index_array_operators<2, cl::sycl::id>(
            itemBase.get_id(0), itemBase.get_id(1), itemBase.get_id(2)) {}
  /** COMPUTECPP_DEV @endcond */

  /** @brief Default constructor for id, initialized to { 0, 0, 0 }
   */
  id() : detail::index_array_operators<2, cl::sycl::id>(0, 0, 0) {}

  /** @brief Default copy constructor for id.
   */
  id(const id& rhs) = default;

  /** @brief Create a two dimensional id from two size_t parameters.
   */
  id(size_t x, size_t y)
      : detail::index_array_operators<2, cl::sycl::id>(x, y, 0) {}
  /** @cond COMPUTECPP_DEV */

  /** @brief  copy constructor from the templated operator interface class used
   * internally to the public templated class.
   */
  explicit id(const detail::index_array_operators<2, cl::sycl::id>&
                  indexArray)  // NOLINT false +,  conversion
      : detail::index_array_operators<2, cl::sycl::id>(indexArray) {}
  /** @brief copy constructor from the non templated base class used internally
   * to the public templated class.
   */
  id(const detail::index_array& indexArray)
      : id<2>(indexArray[0], indexArray[1]) {}
  /** @brief meant for internal use (conversion for operators).
   */
  id(const detail::index_array_operators<2, range>&
         r)  // NOLINT false +,  conversion
      : id(range<2>(r)) {}

  /** COMPUTECPP_DEV @endcond */

  /** @brief conversion from range to id constructor.
   */
  id(const range<2>& r)  // NOLINT false +,  conversion
      : id(r[0], r[1]) {}

  /** @brief copy constructor from the non templated base class used internally
   * to the public templated class.
   */
  explicit id(const item<2, true>& itemBase)  // NOLINT false +,  conversion
      : id(*reinterpret_cast<const detail::item_base*>(&itemBase)) {}

  /** @brief conversion to int2 vector as a useful conversion for accessing
   * image coordinates.
   */
  operator int2() const {
    return int2(static_cast<int>(m_idx[0]), static_cast<int>(m_idx[1]));
  }
};

/** @brief Implementation of id for 3 dimensions
 */
template <>
class id<3> : public detail::index_array_operators<3, cl::sycl::id> {
 public:
  /** @cond COMPUTECPP_DEV */

  /** @brief Constructor taking in a item
   */
  id(const detail::item_base& itemBase)
      : detail::index_array_operators<3, cl::sycl::id>(
            itemBase.get_id(0), itemBase.get_id(1), itemBase.get_id(2)) {}
  /** COMPUTECPP_DEV @endcond */

  /** @brief Default constructor for id, initialized to { 1, 1, 1 }
   */
  id() : detail::index_array_operators<3, cl::sycl::id>(0, 0, 0) {}

  /** @brief Default copy constructor for id.
   */
  id(const id& rhs) = default;

  /** Create a three dimensional id from three size_t parameters.
   */
  id(size_t x, size_t y, size_t z)
      : detail::index_array_operators<3, cl::sycl::id>(x, y, z) {}

  /** @cond COMPUTECPP_DEV */
  /** @brief copy constructor from the templated operator interface class used
   * internally to the public templated class.
   */
  explicit id(const detail::index_array_operators<3, cl::sycl::id>&
                  indexArray)  // NOLINT false +,  conversion
      : detail::index_array_operators<3, cl::sycl::id>(indexArray) {}

  /** @brief copy constructor from the non templated base class used internally
   * to the public templated class.
   */
  id(const detail::index_array& indexArray)
      : id<3>(indexArray[0], indexArray[1], indexArray[2]) {}

  /** @brief copy constructor from the non templated base class used internally
   * to the public templated class.
   */
  explicit id(const item<3, true>& itemBase)  // NOLINT false +,  conversion
      : id(*reinterpret_cast<const detail::item_base*>(&itemBase)) {}

  /** COMPUTECPP_DEV @endcond */

  /** @brief conversion from range to id constructor.
   */
  id(const range<3>& r)  // NOLINT false +,  conversion
      : id(r[0], r[1], r[2]) {}

  /** @brief conversion to int3 vector as a useful conversion for accessing
   * image coordinates.
   */
  operator int3() const {
    return int3(static_cast<int>(m_idx[0]), static_cast<int>(m_idx[1]),
                static_cast<int>(m_idx[2]));
  }
};

/* @cond COMPUTECPP_DEV */
// specify the id/range op rules
namespace detail {
/** Type trait to define the return type of index_array_operators operations.
 * An operation between an id and a range yields an id of the same dimension.
 */
template <int dim>
struct index_array_ops_results<dim, cl::sycl::id, cl::sycl::range> {
  using ResTy = cl::sycl::id<dim>;
};
/** Type trait to define the return type of index_array_operators operations.
 * An operation between a range and an id yields an id of the same dimension.
 */
template <int dim>
struct index_array_ops_results<dim, cl::sycl::range, cl::sycl::id> {
  using ResTy = cl::sycl::id<dim>;
};
}  // namespace detail

/* COMPUTECPP_DEV @endcond */

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ID_H_
