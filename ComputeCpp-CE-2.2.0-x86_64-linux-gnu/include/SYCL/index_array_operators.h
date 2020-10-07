/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file index_array_operators.h

  @brief This file implement an internal base class for @ref cl::sycl::range and
  @ref cl::sycl::id classes. This base class provides basic operator overloads
  required by the SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_OPERATORS_H_
#define RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_OPERATORS_H_

#include "SYCL/common.h"
#include "SYCL/index_array.h"

namespace cl {
namespace sycl {
namespace detail {
/** @cond COMPUTECPP_DEV */
/** Type trait to define the return type of index_array_operators operations.
 */
template <int dim, template <int> class C1, template <int> class C2>
struct index_array_ops_results {
  // typedef C1 ResTy;
};
/** Type trait to define the return type of index_array_operators operations.
 * An operation involving the same class yields an object of the same class in
 * the same dimension.
 */
template <int dim, template <int> class Class>
struct index_array_ops_results<dim, Class, Class> {
  using ResTy = Class<dim>;
};
}  // namespace detail
}  // namespace sycl
}  // namespace cl

/** Macro that defines operators for the index_array_operators class.
 * This macro implements an operator where the left hand side is a const
 * index_array_operators object( or any derived class object) and a size_t
 * scalar value. e.g. ans = a / 2;
 */
#define COMPUTECPP_IDX_SCALAR_OPERATOR(op, a, b)                        \
  {                                                                     \
    typename index_array_operators<dimensions, Child>::Concrete result; \
    result[0] = (a)[0] op(b);                                           \
    if (dimensions > 1) result[1] = (a)[1] op(b);                       \
    if (dimensions > 2) result[2] = (a)[2] op(b);                       \
    return result;                                                      \
  }

/** Macro that defines operators for the index_array_operators class.
 * This macro implements an operator where the right hand side is a const
 * index_array_operators object( or any derived class object) and a size_t
 * scalar value. e.g. ans = 2 / a;
 */
#define COMPUTECPP_SCALAR_IDX_OPERATOR(op, a, b)                        \
  {                                                                     \
    typename index_array_operators<dimensions, Child>::Concrete result; \
    result[0] = a op b[0];                                              \
    if (dimensions > 1) result[1] = a op b[1];                          \
    if (dimensions > 2) result[2] = a op b[2];                          \
    return result;                                                      \
  }

/** Macro that defines operators for the index_array_operators class.
 * This macro implements an operator where the inputs are const
 * index_array_operators objects( or any derived class objects)
 * value. e.g. ans = a / b;
 */
#define COMPUTECPP_IDX_IDX_OPERATOR(op, a, b)                           \
  {                                                                     \
    typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy \
        result;                                                         \
    result[0] = (a)[0] op(b)[0];                                        \
    if (dimensions > 1) result[1] = (a)[1] op(b)[1];                    \
    if (dimensions > 2) result[2] = (a)[2] op(b)[2];                    \
    return result;                                                      \
  }

/** Macro that defines comparison operators for the index_array_operators class.
 * This macro implements an operator where the inputs are const
 * index_array_operators objects( or any derived class objects) and the output
 * is the result as a boolean.
 * value. e.g. ans = a / b;
 */
#define COMPUTECPP_IDX_COMPARISON_OPERATOR(op, a, b)                    \
  {                                                                     \
    typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy \
        result;                                                         \
    result[0] = (a)[0] op(b)[0];                                        \
    if (dimensions > 1) result[1] = ((a)[1] op(b)[1]);                  \
    if (dimensions > 2) result[2] = ((a)[2] op(b)[2]);                  \
    return result;                                                      \
  }

/** Macro that defines comparison operators for the index_array_operators class.
 * This macro implements an operator where the inputs are const
 * index_array_operators objects( or any derived class objects) and the output
 * is the result as a boolean.
 * value. e.g. ans = a / b;
 */
#define COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(op)                               \
  {                                                                          \
    this->m_idx[0] op rhs.m_idx[0];                                          \
    if (dimensions > 1) this->m_idx[1] op rhs.m_idx[1];                      \
    if (dimensions > 2) this->m_idx[2] op rhs.m_idx[2];                      \
    return *reinterpret_cast<                                                \
        typename index_array_operators<dimensions, Child>::Concrete*>(this); \
  }

/** Macro that defines comparison operators between index_array_operator class
 * and scalar.
 * This macro implements an operator where the inputs is an index_array_operator
 * object( or any derived class objects) and a scalar and the output
 * is the result as a index_array_operator object.
 * value. e.g. ans = a / b;
 */
#define COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(op)                        \
  {                                                                          \
    this->m_idx[0] = this->m_idx[0] op rhs;                                  \
    if (dimensions > 1) this->m_idx[1] = this->m_idx[1] op rhs;              \
    if (dimensions > 2) this->m_idx[2] = this->m_idx[2] op rhs;              \
    return *reinterpret_cast<                                                \
        typename index_array_operators<dimensions, Child>::Concrete*>(this); \
  }

namespace cl {
namespace sycl {
namespace detail {

template <int dims, template <int> class Child>
class index_array_operators : public index_array_base {
 public:
  using Concrete = Child<dims>;
  static_assert(
      (dims > 0 && dims < 4),
      "The allowed dimensionality of is within the input range of [1,3].");

 protected:
  // index_array_operators() = default;
  index_array_operators(size_t a, size_t b, size_t c)
      : index_array_base(a, b, c) {}

 public:
  explicit index_array_operators(const index_array& rhs)
      : index_array_base(rhs) {}

  /** @brief Equality operator
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  friend inline bool operator==(const index_array_operators& lhs,
                                const index_array_operators& rhs) {
    return lhs.is_equal<dims>(rhs);
  }

  /** @brief Non-equality operator
   * @param rhs Object to compare to
   * @return True if at least one member variable not the same
   *         as the corresponding rhs member variable
   */
  friend inline bool operator!=(const index_array_operators& lhs,
                                const index_array_operators& rhs) {
    return !(lhs == rhs);
  }

  /** assignment operator
   * It assigns all values of rhs to the current index_array_operators
   * object.
   * \param[in] rhs input index_array_operators object of the same
   *                      dimensionality
   * \returns reference to current index_array_operators object
   */
  // NOLINTNEXTLINE(misc-unconventional-assign-operator)
  Concrete& operator=(const index_array_operators<dims, Child>& rhs);

  /** Subtraction assignment operator, which subtracts the input
   * index_array_operators from the current index_array_operators object
   * \param[in] rhs input index_array_operators object of the same
   *                      dimensionality
   * \returns reference to current index_array_operators object
   */
  Concrete& operator-=(const index_array_operators<dims, Child>& rhs);

  /** Division assignment operator, which divides the current
   * index_array_operators object by the input index_array_operators
   * \param[in] rhs input index_array_operators object of the same
   *                       dimensionality
   * \returns reference to current index_array_operators object
   */
  Concrete& operator/=(const index_array_operators<dims, Child>& rhs);

  /** \brief Multiplication assignment operator, which multiplies the current
   * index_array_operators object with the input index_array_operators
   * \param[in] rhs input index_array_operators object of the same
   *                       dimensionality
   * \returns reference to current index_array_operators object
   */
  Concrete& operator*=(const index_array_operators<dims, Child>& rhs);

  /** Remainder assignment operator, which divides the current
   * index_array_operators object by the input index_array_operators and assigns
   * back the remainder of the division
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator%=(const index_array_operators<dims, Child>& rhs);

  /** Shift right assignment operator, which shifts the current
   * index_array_operators object by the input index_array_operators
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator>>=(const index_array_operators<dims, Child>& rhs);

  /** Shift left assignment operator, which shifts left the current
   * index_array_operators object by the input index_array_operators and assigns
   * back the remainder of the division
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator<<=(const index_array_operators<dims, Child>& rhs);

  /** XOR assignment operator, which computes the logical XOR of the current
   * index_array_operators and the input index_array_operators objects
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator^=(const index_array_operators<dims, Child>& rhs);

  /** \brief AND assignment operator, which computes the logical AND of the
   * current
   * index_array_operators and the input index_array_operators objects
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator&=(const index_array_operators<dims, Child>& rhs);

  /** OR assignment operator, which applies a logical OR to the current
   * object with the input index_array_operators object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator|=(const index_array_operators<dims, Child>& rhs);

  /** addition operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator+=(const index_array_operators<dims, Child>& rhs);

  /** addition operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator+=(const size_t& rhs);

  /** subtraction operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator-=(const size_t& rhs);

  /** multiplication operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator*=(const size_t& rhs);

  /** division operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator/=(const size_t& rhs);

  /** mod operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator%=(const size_t& rhs);

  /** right shift operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator>>=(const size_t& rhs);

  /** left shift operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator<<=(const size_t& rhs);

  /** bitwise and operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator&=(const size_t& rhs);

  /** bitwise or operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator|=(const size_t& rhs);

  /** bitwise xor operator, which adds the inputs of the input object to
   * the current object.
   * \param[in] rhs input index_array_operators object of the same
   * dimensionality \returns reference to current index_array_operators object
   */
  Concrete& operator^=(const size_t& rhs);
};

/** implementation of the assignment operator */
// NOLINTNEXTLINE(misc-unconventional-assign-operator)
template <int dimensions, template <int> class Child>
// NOLINTNEXTLINE(misc-unconventional-assign-operator)
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(=)
}

/** implementation of the assignment shift right operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator>>=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(>>=)
}

/** implementation of the assignment minus operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator-=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(-=)
}

/** implementation of the division assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator/=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(/=)
}

/** implementation of the division assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator*=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(*=)
}

/** implementation of the modulo assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator%=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(%=)
}

/** implementation of the assignment shift right operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator<<=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(<<=)
}

/** implementation of the assignment logical XOR operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator^=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(^=)
}

/** implementation of the assignment logical AND operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator&=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(&=)
}

/** implementation of the assignment bitwise OR operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator|=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(|=)
}

/** implementation of the addition assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator+=(
    const index_array_operators<dimensions, Child>& rhs) {
  COMPUTECPP_IDX_ASSIGNMENT_OPERATOR(+=)
}

/** implementation of the addition assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator+=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(+=)
}

/** implementation of the subtraction assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator-=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(-=)
}

/** implementation of the multiplication assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator*=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(*=)
}

/** implementation of the division assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator/=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(/=)
}

/** implementation of the mod assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator%=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(%=)
}

/** implementation of the right shift assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator>>=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(>>=)
}

/** implementation of the left shift assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator<<=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(<<=)
}

/** implementation of the logical and assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator&=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(&=)
}

/** implementation of the logical or assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator|=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(|=)
}

/** implementation of the logical xor assignment operator */
template <int dimensions, template <int> class Child>
inline typename index_array_operators<dimensions, Child>::Concrete&
index_array_operators<dimensions, Child>::operator^=(const size_t& rhs) {
  COMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATOR(^=)
}

/** non-member less comparison operator of two index_array_operators of the
 * same dimensionality.
 * \param[in] a input typename index_array_operators<dimensions,
 * Child>::Concrete object \param[in] b input typename
 * index_array_operators<dimensions, Child>::Concrete object \returns a typename
 * index_array_operators<dimensions, Child>::Concrete of the same dimensionality
 * as a and b, which is the result of the comparison of a > b.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
inline typename index_array_operators<dimensions, C1>::Concrete operator>(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_COMPARISON_OPERATOR(>, a, b)
}

/** non-member comparison greater or equal operator of two
 * index_array_operators of the same dimensionality.
 * \param[in] a input typename index_array_operators<dimensions,
 * Child>::Concrete object \param[in] b input typename
 * index_array_operators<dimensions, Child>::Concrete object \returns a typename
 * index_array_operators<dimensions, Child>::Concrete of the same dimensionality
 * as a and b, which is the result of the comparison of a >= b.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
inline typename index_array_operators<dimensions, C1>::Concrete operator>=(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_COMPARISON_OPERATOR(>=, a, b)
}

/** non-member comparison less operator of two index_array_operators of the
 * same dimensionality.
 * \param[in] a input typename index_array_operators<dimensions,
 * Child>::Concrete object \param[in] b input typename
 * index_array_operators<dimensions, Child>::Concrete object \returns a typename
 * index_array_operators<dimensions, Child>::Concrete of the same dimensionality
 * as a and b, which is the result of the comparison of a < b.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
inline typename index_array_operators<dimensions, C1>::Concrete operator<(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_COMPARISON_OPERATOR(<, a, b)
}

/** non-member comparison less or equal operator of two index_array_operators
 * of the same dimensionality.
 * \param[in] a input typename index_array_operators<dimensions,
 * Child>::Concrete object \param[in] b input typename
 * index_array_operators<dimensions, Child>::Concrete object \returns a typename
 * index_array_operators<dimensions, Child>::Concrete of the same dimensionality
 * as a and b, which is the result of the comparison of a <= b.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
inline typename index_array_operators<dimensions, C1>::Concrete operator<=(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_COMPARISON_OPERATOR(<=, a, b)
}

/** non-member addition operator of two ranges of the same dimensionality.
 * \param[in] a input range object
 * \param[in] b input range object
 * \returns a range object of the same dimensionality as a and b, where the
 * value of each dimension is the result of the addition of the corresponding
 * values for a and b.
 */

template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator+(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(+, a, b)
}

/** non-member division operator of two ranges of the same dimensionality.
 * \param[in] a input range object
 * \param[in] b input range object
 * \returns a range of the same dimensionality as a and b, where the value of
 * each dimension is the result of the division of a by b.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator/(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(/, a, b)
}

/** non-member subtraction operator of two ranges of the same dimensionality.
 * \param[in] a input range object
 * \param[in] b input range object
 * \returns a range of the same dimensionality as a and b, which  is the result
 * of the subtraction of a and b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator-(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(-, a, b)
}

/** non-member remainder operator of two ranges of the same dimensionality.
 * \param[in] a input range object
 * \param[in] b input range object
 * \returns a range of the same dimensionality as a and b, which is the  result
 * of the remainder of the division of a by b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator%(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(%, a, b)
}

/** non-member multiplication operator of two ranges of the same dimensionality.
 * \param[in] a input range object
 * \param[in] b input range object
 * \returns a range of the same dimensionality as a and b, which is the
 * result of the remainder of the multiplication of a by b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator*(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(*, a, b)
}

/** non-member shift left operator of two ranges of the same
 * dimensionality.
 * \param[in] a input range object
 * \param[in] b input range object
 * \returns a range object of the same dimensionality as a and b, which
 * is the result of shifting left a by b  for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator<<(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(<<, a, b)
}

/** non-member shift left operator of two ranges of the same dimensionality.
 * \param[in] a input range object
 * \param[in] b input range object
 * \returns a range of the same dimensionality as a and b, which
 * is the result of shifting left a by b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator>>(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(>>, a, b)
}

/** non-member bitwise AND operator for ranges of the same dimensionality
 *  \param[in] a input range
 *  \param[in] b input range
 *  \returns a range which is the result of a&b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator&(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(&, a, b)
}

/** non-member logical AND operator for ranges of the same dimensionality
 * \param[in] a input range
 * \param[in] b input range
 * \returns a range which is the result of a&&b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator&&(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(&&, a, b)
}

/** non-member bitwise OR operator on ranges of the same dimensionality
 * \param[in] a input range
 * \param[in] b input range
 * \returns a range of the same dimensionality as a and b, which is the result
 * of a|b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator|(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(|, a, b)
}

/** non-member logical OR operator of ranges with the same dimensionality
 * \param[in] a input range
 * \param[in] b input range
 * \returns a range of the same dimensionality as a and b,which is the
 * outcome of a||b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator||(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(||, a, b)
}

/** non-member logical XOR operator of ranges with the same dimensionality.
 * \param[in] a input range
 * \param[in] b input range
 * \returns a range where the value of each dimension is the outcome of
 * a XOR b for each dimension.
 */
template <int dimensions, template <int> class C1, template <int> class C2>
typename detail::index_array_ops_results<dimensions, C1, C2>::ResTy operator^(
    const index_array_operators<dimensions, C1>& a,
    const index_array_operators<dimensions, C2>& b) {
  COMPUTECPP_IDX_IDX_OPERATOR(^, a, b)
}

/** non-member multiplication operator of a range with a scalar.
 * \param[in] a range input
 * \param[in] b scalar input
 * \returns a range of the same dimensionality as a, where the value of every
 * dimension is the result of a*b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator*(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(*, a, b)
}

/** non-member multiplication operator of a scalar with a range
 * \param[in] a scalar input
 * \param[in] b range input
 * \returns a range of the same dimensionality as b, where the value of every
 * dimension is the result of a*b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator*(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(*, a, b)
}

/** non-member division operator of scalar with a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as a and b, where the value of
 * each dimension is the result of a/b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator/(
    size_t a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(/, a, b)
}

/** non-member division operator of a typename index_array_operators<dimensions,
 * Child>::Concrete with a scalar
 * \param[in] a input index_array_operators
 * \param[in] b input scalar
 * \returns a typename index_array_operators<dimensions, Child>::Concrete where
 * the value of each dimension is the
 * result of
 * a/b
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator/(
    const index_array_operators<dimensions, Child>& a, const size_t b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(/, a, b)
}

/** non-member addition operator of a scalar and a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a+b
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator+(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(+, a, b)
}

/** non-member addition operator of a range and a scalar
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as a, where the value of each
 * dimension is the result of a+b
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator+(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(+, a, b)
}

/** non-member subtraction operator of a range from a scalar
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of  a-b
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator-(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(-, a, b)
}

/** non-member subtraction operator of a scalar from a range
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range where the value of each dimension is the  result of a-b
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator-(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(-, a, b)
}

/** non-member modulo operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 *dimension is the result of a-b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator%(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(%, a, b)
}

/** non-member modulo operator of a range by a scalar
 * \param[in] a input index_array_operators
 * \param[in] b input scalar
 * \returns a range where the value of each dimension is the result of a-b
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator%(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(%, a, b)
}

/** non-member shift left operator of a scalar by a range
 * \param[in] a scalar input
 * \param[in] b range input
 * \returns a range of the same dimensionality as b, where the value of each
 * dimensions is the result of a<<b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator<<(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(<<, a, b)
}

/* non-member shift left operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a, where the value of each
 * dimension is the result of a<<b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator<<(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(<<, a, b)
}

/** non-member shift right operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a>>b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator>>(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(>>, a, b)
}

/** non-member shift right operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a>>b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator>>(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(>>, a, b)
}

/** non-member bitwise or operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a|b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator|(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(|, a, b)
}

/** non-member bitwise or operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a|b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator|(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(|, a, b)
}

/** non-member bitwise xor operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a^b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator^(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(^, a, b)
}

/** non-member bitwise xor operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a^b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator^(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(^, a, b)
}

/** non-member logical and operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b,
 * whCOMPUTECPP_IDX_SCALAR_ASSIGNMENT_OPERATORere the value of each
 * dimension is the result of a&&b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator&&(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(&&, a, b)
}

/** non-member logical and operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a&&b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator&&(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(&&, a, b)
}

/** non-member logical or operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a||b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator||(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(||, a, b)
}

/** non-member logical or operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a||b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator||(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(||, a, b)
}

/** non-member greater than operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a>b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator>(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(>, a, b)
}

/** non-member greater than operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a>b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator>(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(>, a, b)
}

/** non-member less than operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a<b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator<(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(<, a, b)
}

/** non-member less than operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a<b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator<(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(<, a, b)
}

/** non-member greater than or equal operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a>=b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator>=(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(>=, a, b)
}

/** non-member greater than or equal operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a>=b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator>=(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(>=, a, b)
}

/** non-member less than or equal operator of a scalar by a range
 * \param[in] a input scalar
 * \param[in] b input range
 * \returns a range of the same dimensionality as b, where the value of each
 * dimension is the result of a<=b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator<=(
    const size_t& a, const index_array_operators<dimensions, Child>& b) {
  COMPUTECPP_SCALAR_IDX_OPERATOR(<=, a, b)
}

/** non-member less than or equal operator of a range by a scalar
 * \param[in] a input range
 * \param[in] b input scalar
 * \returns a range of the same dimensionality as a a, where the value of each
 * dimension is the result of a<=b.
 */
template <int dimensions, template <int> class Child>
typename index_array_operators<dimensions, Child>::Concrete operator<=(
    const index_array_operators<dimensions, Child>& a, const size_t& b) {
  COMPUTECPP_IDX_SCALAR_OPERATOR(<=, a, b)
}

}  // namespace detail
}  // namespace sycl
}  // namespace cl
#undef COMPUTECPP_IDX_SCALAR_OPERATOR
#undef COMPUTECPP_SCALAR_IDX_OPERATOR
#undef COMPUTECPP_IDX_IDX_OPERATOR
#undef COMPUTECPP_IDX_COMPARISON_OPERATOR
#undef COMPUTECPP_IDX_ASSIGNMENT_OPERATOR

/* COMPUTECPP_DEV @endcond */
#endif
