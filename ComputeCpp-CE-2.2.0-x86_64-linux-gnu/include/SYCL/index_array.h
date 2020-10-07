/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file index_array.h
 *
 * @brief This file implement the base class for @ref cl::sycl::id and @ref
 * cl::sycl::range classes.
 */

#ifndef RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_H_
#define RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_H_

#include "SYCL/assert.h"
#include "SYCL/common.h"
#include "SYCL/error.h"

/** @cond COMPUTECPP_DEV */

namespace cl {
namespace sycl {
namespace detail {

class index_array {
 public:
  using array_ref_t = size_t (&)[3];
  using const_array_ref_t = const size_t (&)[3];

  index_array() = default;
  /** Copy constructor
   */
  index_array(const index_array& rhs) = default;

  /** 3D constructor
   * It initializes only the first index
   */
  index_array(size_t x, size_t y, size_t z);

  /** Subscript operator
   */
  size_t& operator[](unsigned int dimension);

  /** Subscript operator (const)
   */
  size_t operator[](unsigned int dimension) const;

  /** Assignment Operator.
   * Assigns each element of the integer
   * array with the corresponding values
   * in
   * rhs.
   * @param the rhs cl::sycl::index_array
   * object.
   */
  index_array& operator=(const index_array& rhs) = default;

  /** Multiply Operator.
   * element wise multiplication
   * @param the rhs cl::sycl::index_array
   * object.
   */
  index_array operator*(const index_array& rhs) const;

  /** \brief get value for a specific dimension
   * @param dimension
   * @return index in dimension
   */
  size_t get(unsigned int dimension) const { return m_idx[dimension]; }

  /** @brief Get the id array as a size_t array. Can be used for OpenCL interop.
   * @return Pointer to id values for all dimensions
   */
  array_ref_t get() { return m_idx; }

  /** @brief Get the id array as a size_t array. Can be used for OpenCL interop.
   * @return Pointer to id values for all dimensions
   */
  const_array_ref_t get() const { return m_idx; }

  /// @cond COMPUTECPP_DEV

  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  inline bool is_equal(const index_array& rhs) const {
    bool isEqual = (m_idx[0] == rhs.m_idx[0]);
    if (dimensions > 1) {
      isEqual = isEqual && (m_idx[1] == rhs.m_idx[1]);
      if (dimensions > 2) {
        isEqual = isEqual && (m_idx[2] == rhs.m_idx[2]);
      }
    }
    return isEqual;
  }

  /** @brief Calculates the number of elements covered by this index array.
   *        Only valid if this object represents a range.
   * @return Number of elements across three dimensions
   */
  constexpr size_t get_count_impl() const noexcept {
    return m_idx[0] * m_idx[1] * m_idx[2];
  }

  /// COMPUTECPP_DEV @endcond

 protected:
  size_t m_idx[3];
};

inline index_array::index_array(size_t x, size_t y, size_t z) {
  this->m_idx[0] = x;
  this->m_idx[1] = y;
  this->m_idx[2] = z;
}

inline size_t& index_array::operator[](unsigned int dimension) {
  COMPUTECPP_ASSERT(dimension < 3, "Incorrect number of dimensions");
  return m_idx[dimension];
}

inline size_t index_array::operator[](unsigned int dimension) const {
  COMPUTECPP_ASSERT(dimension < 3, "Incorrect number of dimensions");
  return m_idx[dimension];
}

inline index_array index_array::operator*(const index_array& rhs) const {
  return index_array(m_idx[0] * rhs.m_idx[0], m_idx[1] * rhs.m_idx[1],
                     m_idx[2] * rhs.m_idx[2]);
}

/* Hold the index_array object and basic access to it */
class index_array_base {
 public:
  index_array_base() = default;
  /** Copy constructor
   */
  index_array_base(const index_array_base& rhs) = default;

  /** Ctor from an index_array
   */
  index_array_base(const index_array& rhs) : m_idx(rhs) {}

  /** 3D constructor
   * It initializes only the first index
   */
  index_array_base(size_t x, size_t y, size_t z);

  /** Subscript operator
   */
  size_t& operator[](unsigned int dimension);

  /** Subscript operator (const)
   */
  size_t operator[](unsigned int dimension) const;

  /** Assignment Operator.
   * Assigns each element of the integer
   * array with the corresponding values in rhs.
   * @param the rhs cl::sycl::index_array_base object.
   */
  index_array_base& operator=(const index_array_base& rhs) = default;

  /** \brief get value for a specific dimension
   * @param dimension
   * @return index in dimension
   */
  size_t get(unsigned int dimension) const { return m_idx[dimension]; }

  /** \brief conversion operator from a dimension aware index class to the
   * dimension agnostic class.
   * @return The dimension agnostic object representation of this index class.
   */
  operator index_array() const { return m_idx; }

  /// @cond COMPUTECPP_DEV

  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  inline bool is_equal(const index_array_base& rhs) const {
    return m_idx.is_equal<dimensions>(rhs.m_idx);
  }

  /// COMPUTECPP_DEV @endcond

 protected:
  index_array m_idx;
};

inline index_array_base::index_array_base(size_t x, size_t y, size_t z) {
  this->m_idx[0] = x;
  this->m_idx[1] = y;
  this->m_idx[2] = z;
}

inline size_t& index_array_base::operator[](unsigned int dimension) {
  COMPUTECPP_ASSERT(dimension < 3, "Incorrect number of dimensions");
  return m_idx[dimension];
}

inline size_t index_array_base::operator[](unsigned int dimension) const {
  COMPUTECPP_ASSERT(dimension < 3, "Incorrect number of dimensions");
  return m_idx[dimension];
}

/** @brief Calculates a row-major linearized index from an offset and a range
 * @param offset The offset from the beginning
 * @param range The original range
 * @return The linearized index
 */
inline size_t construct_linear_row_major_index(
    const detail::index_array& offset, const detail::index_array& range) {
  return construct_linear_row_major_index(offset[0], offset[1], offset[2],
                                          range[0], range[1], range[2]);
}

}  // namespace detail

}  // namespace sycl
}  // namespace cl

/** COMPUTECPP_DEV @endcond */
#endif  // RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_H_
