/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

    * atomic.h *

*******************************************************************************/

/** @file atomic.h
 * @brief This file contains an implementation of the atomic class as described
 * in the SYCL specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ATOMIC_H_
#define RUNTIME_INCLUDE_SYCL_ATOMIC_H_

#include <algorithm>
#include <atomic>

namespace cl {
namespace sycl {

namespace detail {

// Forward declaration
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class accessor_buffer_interface;

}  // namespace detail

/** @brief Only relaxed memory order is supported in SYCL 1.2.1
 */
enum class memory_order : int {
  relaxed = static_cast<int>(std::memory_order_relaxed),
};

/** @brief Atomic class template
 *
 * This template class specifies the interface and internal data of atomics
 * as specified by SYCL. It offers several different atomic operations,
 * including min/max which are not otherwise available in C++ 11 code.
 * Most of the file is visible to the device compiler only;
 * this is so that the runtime can call the appropriate atomic function based
 * on the type of the elements. A portion is visible to both (class declaration
 * and global functions) with a small section for the host-only implementation.
 * The device compiler section has separate specialisations for each pair of
 * template parameters. They are organised primarily by type (cl_int, cl_uint
 * etc.) and secondarily by address space (global then local). It is done like
 * this because the SPIR function to be called is different based on the type
 * and address space of the atomic.
 */
template <typename T, access::address_space addressSpace =
                          access::address_space::global_space>
struct atomic;

/** @brief atomic int object with default global address space
 */
using atomic_int = atomic<cl_int>;
/** @brief atomic unsigned int object with default global address space
 */
using atomic_uint = atomic<cl_uint>;
/** @brief atomic float object with default global address space
 */
using atomic_float = atomic<cl_float>;

template <typename elemT, access::address_space addressSpace>
struct device_type {
  /** @brief Underlying type of the device pointer
   */
  using underlying_t = elemT;

  /** @brief Pointer type used on device
   */
  using ptr_t = multi_ptr<underlying_t, addressSpace>;
};

/** @brief Implementation of the SYCL atomic class according to 1.2 spec.
 * (section 3.8). On host, calls C++ atomic functions on an
 * std::atomic; on device uses SPIR-mangled OpenCL 1.2 functions
 * to achieve same result.
 */
template <typename T, access::address_space addressSpace>
struct atomic {
 private:
  /** @brief Pointer type used on the device
   */
  using device_ptr_t = typename device_type<T, addressSpace>::ptr_t;

  /* @cond COMPUTECPP_DEV */
  /* Host has a single C++ 11 atomic, device simply has a pointer in the
   * global address space */

#ifndef __SYCL_DEVICE_ONLY__
  /** @brief Pointer to std::atomic<T>, host only.
   */
  std::atomic<T>* m_data;
#else
  /** @brief pointer decorated with address space. Device only.
   */
  device_ptr_t m_data;
#endif

  /** @brief Factory function only visible to accessors. Stores the address
   * provided internally and operates on that location atomically.
   * @param datum The address to be operated on atomically, obtained
   * from an atomic accessor.
   */
  static atomic make_from_device_ptr(device_ptr_t datum) {
    atomic result;
#ifndef __SYCL_DEVICE_ONLY__
    memcpy(&result.m_data, &datum, sizeof(m_data));
#else
    result.m_data = datum;
#endif
    return result;
  }

  /** @brief Private default constructor that is meant to be used in
   * make_from_device_ptr only.
   */
  atomic() : m_data{nullptr} {}

  /* @endcond  */

 public:
  /* @brief The accessors are friends so that it can access the constructor but
   * user code can't. */
  template <typename elemT, int kDims, access::mode kMode,
            access::target kTarget, access::placeholder isPlaceholder>
  friend class detail::accessor_buffer_interface;

  /** @brief Constructs an instance of SYCL atomic which is associated with the
   *        pointer ptr, converted to a pointer of data type T.
   *
   *        Permitted data types for pointerT is any valid scalar data type
   *        which is the same size in bytes as T.
   *
   * @tparam pointerT Underlying type of the pointer ptr
   * @param ptr Pointer to be used in an atomic manner
   */
  template <typename pointerT>
  atomic(multi_ptr<pointerT, addressSpace> ptr)
      : atomic(make_from_device_ptr(ptr.get())) {}

  /* @brief Functions as mandated by specification. Global functions simply
   * forward on to these function calls. */

  /** @brief Atomically store operand in m_data. Calls C++11 equivalent on host,
   * on device it calls exchange, discarding the result.
   * @param operand the value to store in m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   */
  void store(T operand, memory_order mem_order = memory_order::relaxed);
  /** @brief Atomically load from m_data. Calls C++11 equivalent on host,
   * on device it either calls atomic_add with operand = 0, discarding
   * the result.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return The value loaded from m_data.
   */
  T load(memory_order mem_order = memory_order::relaxed) const;
  /** @brief Atomically exchange operand with *m_data.
   * @param operand the value to store in *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return The old value of *m_data.
   */
  T exchange(T operand, memory_order mem_order = memory_order::relaxed);
  /** @brief Atomically compare and optionally exchange expected with *m_data.
   * Calls C++11 equivalent on host, has to be implemented "by hand" on device
   * because OpenCL 1.2 and C++ 11 have different semantics for compare and
   * exchange.
   * If *m_data == expected, performs *m_data = desired and returns true.
   * Otherwise, performs expected = *m_data and returns false.
   * @param expected The value to compare against *m_data.
   * @param desired The value to store in *m_data on success.
   * @param success the ordering to use when comparison succeeds. Can only
   * be memory_order_relaxed.
   * @param fail the ordering to use when comparison fails. Can only
   * be memory_order_relaxed.
   * @return True if comparison succeeds, false if it fails.
   */
  cl_bool compare_exchange_strong(T& expected, T desired,
                                  memory_order success = memory_order::relaxed,
                                  memory_order fail = memory_order::relaxed);
  /** @brief Atomically add operand to *m_data.
   * param operand the value to add to *m_data.
   * param mem_order the ordering to use. Can only be memory_order_relaxed.
   * return the old value of *m_data.
   */
  T fetch_add(T operand, memory_order mem_order = memory_order::relaxed);
  /** @brief Atomically subtract operand from *m_data.
   * @param operand the value to subtract from *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of m_data.
   */
  T fetch_sub(T operand, memory_order mem_order = memory_order::relaxed);
  /** @brief Atomically bitwise-and operand with *m_data.
   * @param operand the value to and with *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_and(T operand, memory_order mem_order = memory_order::relaxed);
  /** @brief Atomically bitwise-or operand with *m_data.
   * @param operand the value to or with *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_or(T operand, memory_order = memory_order::relaxed);
  /** @brief Atomically bitwise-xor operand with *m_data.
   * @param operand the value to xor with *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_xor(T operand, memory_order mem_order = memory_order::relaxed);
  /** @brief Atomically compare operand to *m_data, storing the smaller of the
   * two
   * in *m_data.
   * @param operand the value to compare to *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_min(T operand, memory_order mem_order = memory_order::relaxed);
  /** @brief Atomically compare operand to *m_data, storing the larger of the
   * two in *m_data.
   * @param operand the value to compare to *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_max(T operand, memory_order mem_order = memory_order::relaxed);
};

#ifndef __SYCL_DEVICE_ONLY__
/** @cond COMPUTECPP_DEV */

template <typename T, access::address_space addressSpace>
inline void atomic<T, addressSpace>::store(T operand, memory_order mem_order) {
  m_data->store(operand, static_cast<std::memory_order>(mem_order));
  return;
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::load(memory_order mem_order) const {
  return m_data->load(static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::exchange(T operand, memory_order mem_order) {
  return m_data->exchange(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline cl_bool atomic<T, addressSpace>::compare_exchange_strong(
    T& expected, T desired, memory_order success, memory_order fail) {
  return m_data->compare_exchange_strong(
      expected, desired, static_cast<std::memory_order>(success),
      static_cast<std::memory_order>(fail));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_add(T operand, memory_order mem_order) {
  return m_data->fetch_add(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_sub(T operand, memory_order mem_order) {
  return m_data->fetch_sub(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_and(T operand, memory_order mem_order) {
  return m_data->fetch_and(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_or(T operand, memory_order mem_order) {
  return m_data->fetch_or(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_xor(T operand, memory_order mem_order) {
  return m_data->fetch_xor(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_min(const T operand,
                                            const memory_order mem_order) {
  /* Standard C++ 11 defines no "min" operation, so this function emulates the
   * behaviour by executing a loop. First, the value is loaded, then compared
   * against the operand. After this a compare_exchange is used to check that
   * the value hasn't been updated sneakily (by another thread say); if it has
   * (i.e. m_data doesn't equal its old value) store the new value of the atomic
   * and try again. */
  T old = m_data->load(std::memory_order_relaxed);
  do {
    if (old < operand) {
      break;
    }
  } while (!m_data->compare_exchange_weak(
      old, operand, static_cast<std::memory_order>(mem_order)));
  return old;
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_max(const T operand,
                                            const memory_order mem_order) {
  /* Standard C++ 11 defines no "max" operation, so this function emulates the
   * behaviour by executing a loop. First, the value is loaded, then compared
   * against the operand. After this a compare_exchange is used to check that
   * the value hasn't been updated sneakily (by another thread say); if it has
   * (i.e. m_data doesn't equal its old value) store the new value of the atomic
   * and try again. */
  T old = m_data->load(std::memory_order_relaxed);
  do {
    if (operand < old) {
      break;
    }
  } while (!m_data->compare_exchange_weak(
      old, operand, static_cast<std::memory_order>(mem_order)));
  return old;
}

/** COMPUTECPP_DEV @endcond  */

#endif  // __SYCL_DEVICE_ONLY__

/* global function definitions
 * --------------------------------------------------------------------------*/

/* For each of these global functions f(atomic * a, operands...), the code is
 * simply:
 * a->f(operands) */

/** @function Global function atomic_load. Calls load on SYCL atomic object.
 * @param object The atomic object to load from
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_load(atomic<T, addressSpace> object,
                     memory_order mem_order = memory_order::relaxed) {
  return object.load(mem_order);
}

/** @brief Global function atomic_store. Calls store on SYCL atomic object.
 * @param object The atomic object to store to
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 */
template <typename T, access::address_space addressSpace>
inline void atomic_store(atomic<T, addressSpace> object, T operand,
                         memory_order mem_order = memory_order::relaxed) {
  return object.store(operand, mem_order);
}

/** @brief Global function atomic_exchange. Calls exchange on SYCL atomic
 * object.
 * @param object The atomic object to exchange with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return the old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_exchange(atomic<T, addressSpace> object, T operand,
                         memory_order mem_order = memory_order::relaxed) {
  return object.exchange(operand, mem_order);
}

/** @brief Global function atomic_compare_exchange. Calls compare_exchange on
 * SYCL
 * atomic object.
 * @param object The atomic object to compare_exchange with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return Whether comparison succeeds or fails
 */
template <typename T, access::address_space addressSpace>
inline cl_bool atomic_compare_exchange_strong(
    atomic<T, addressSpace> object, T& expected, T desired,
    memory_order success = memory_order::relaxed,
    memory_order fail = memory_order::relaxed) {
  return object.compare_exchange_strong(expected, desired, success, fail);
}

/** @brief Global function atomic_add. Calls add on SYCL atomic object.
 * @param object The atomic object to add to
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_add(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_add(operand, mem_order);
}

/** @brief Global function atomic_sub. Calls sub on SYCL atomic object.
 * @param object The atomic object to subtract from
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_sub(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_sub(operand, mem_order);
}

/** @brief Global function atomic_and. Calls and on SYCL atomic object.
 * @param object The atomic object to and with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_and(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_and(operand, mem_order);
}

/** @brief Global function atomic_or. Calls or on SYCL atomic object.
 * @param object The atomic object to or with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_or(atomic<T, addressSpace> object, T operand,
                         memory_order mem_order = memory_order::relaxed) {
  return object.fetch_or(operand, mem_order);
}

/** @brief Global function atomic_xor. Calls xor on SYCL atomic object.
 * @param object The atomic object to xor with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_xor(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_xor(operand, mem_order);
}

/** @brief Global function atomic_min. Calculates min(object, operand), storing
 * the
 * result in object.
 * @param object The atomic object to perform min on
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_min(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_min(operand, mem_order);
}

/** @brief Global function atomic_max. Calculates max(object, operand), storing
 * the
 * result in object.
 * @param object The atomic object to perform max on
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_max(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_max(operand, mem_order);
}

namespace detail {

/** @brief Retrieves the address space, suitable for use in an atomic,
 *        from the access target.
 *
 *        The general case is to use global_space - this value needs to be
 *        available even for cases that the atomic class doesn't support.
 *
 * @tparam accessTarget Access target to retrieve the address space for
 */
template <access::target accessTarget>
struct get_atomic_address_space {
  /** @brief Most targets will correspond to the global address space,
   *        even though it's only valid for access::target::global_buffer
   */
  static constexpr auto value = access::address_space::global_space;
};

/** @brief Retrieves the address space, suitable for use in an atomic,
 *        from the access::target::local access target
 */
template <>
struct get_atomic_address_space<access::target::local> {
  /** @brief The local target corresponds to the local address space
   */
  static constexpr auto value = access::address_space::local_space;
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ATOMIC_H_
