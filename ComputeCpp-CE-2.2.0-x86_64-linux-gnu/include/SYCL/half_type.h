/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_HALF_TYPE_H_
#define RUNTIME_INCLUDE_SYCL_HALF_TYPE_H_

#include "computecpp_export.h"

namespace cl {
namespace sycl {

#ifndef __SYCL_DEVICE_ONLY__
/** @brief Definition of half type
 *
 * This class is used to represent a 16 bit floating point number. Code
 * compiled for the device this class will use the compiler builtin __fp16
 * and compile directly to native half instructions. For the host it only
 * operates as a storage type: floating point numbers will be transformed
 * into 16 bits and back but the actual computation will be performed by
 * casting back to a 32 bit float.
 */
class COMPUTECPP_EXPORT half {
 public:
  /** @brief default constructor, inits as zero */
  half() : m_bitpattern(0) {}

  /** @brief Takes in a 32 bit float and converts it to 16 bits
   * @param f the float to be converted to 16 bits
   */
  half(const float& f);

  /** @brief Implicit cast to float
   *  Converts the 16 bit half back to a 32 bit float for perfoming the actual
   *  computation
   */
  operator float() const;

  /** @brief Applies == to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the == operation.
   */
  bool operator==(const half& rhs) const {
    return static_cast<float>(*this) == static_cast<float>(rhs);
  }

  /** @brief Applies != to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the != operation.
   */
  bool operator!=(const half& rhs) const {
    return static_cast<float>(*this) != static_cast<float>(rhs);
  }

  /** @brief Applies < to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the < operation.
   */
  bool operator<(const half& rhs) const {
    return static_cast<float>(*this) < static_cast<float>(rhs);
  }

  /** @brief Applies > to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the > operation.
   */
  bool operator>(const half& rhs) const {
    return static_cast<float>(*this) > static_cast<float>(rhs);
  }

  /** @brief Applies <= to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the <= operation.
   */
  bool operator<=(const half& rhs) const {
    return static_cast<float>(*this) <= static_cast<float>(rhs);
  }

  /** @brief Applies >= to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the >= operation.
   */
  bool operator>=(const half& rhs) const {
    return static_cast<float>(*this) >= static_cast<float>(rhs);
  }

  /** @brief Applies += to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the += operation.
   */
  half& operator+=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp += static_cast<float>(rhs);
    *this = half(temp);
    return *this;
  }

  /** @brief Applies -= to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the -= operation.
   */
  half& operator-=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp -= static_cast<float>(rhs);
    *this = half(temp);
    return *this;
  }

  /** @brief Applies *= to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the *= operation.
   */
  half& operator*=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp *= static_cast<float>(rhs);
    *this = half(temp);
    return *this;
  }

  /** @brief Applies /= to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the /= operation.
   */
  half& operator/=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp /= static_cast<float>(rhs);
    *this = half(temp);
    return *this;
  }

  /** @brief Applies && to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the && operation.
   */
  bool operator&&(const half& rhs) const {
    return (static_cast<float>(*this) != 0.f) &&
           (static_cast<float>(rhs) != 0.f);
  }

  /** @brief Applies || to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the || operation.
   */
  bool operator||(const half& rhs) const {
    return (static_cast<float>(*this) != 0.f) ||
           (static_cast<float>(rhs) != 0.f);
  }

  /** @brief Applies ++ to this half.
   * @return This half after the application of the ++ operation.
   */
  half& operator++() {
    (*this) += 1;
    return *this;
  }

  /** @brief Applies ++ to this half.
   * @return A copy of this half before the ++ operation.
   */
  half operator++(int) {
    half save = *this;
    (*this) += 1;
    return save;
  }

  /** @brief Applies -- to this half.
   * @return This half after the application of the -- operation.
   */
  half& operator--() {
    (*this) -= 1;
    return *this;
  }

  /** @brief Applies -- to this half.
   * @return A copy of this half before the -- operation.
   */
  half operator--(int) {
    half save = *this;
    (*this) -= 1;
    return save;
  }

 private:
  // Contains the floating point number a unsigned short following the
  // 1 bit sign, 5 bit exponent, 10 bit mantissa as set out in the IEEE 754
  // standard
  unsigned short m_bitpattern;
};
#else
typedef __fp16 half;
#endif  // __SYCL_DEVICE_ONLY__
}  // namespace sycl
}  // namespace cl
#endif  // RUNTIME_INCLUDE_SYCL_HALF_TYPE_H_
