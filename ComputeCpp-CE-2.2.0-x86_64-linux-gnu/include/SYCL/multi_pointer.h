/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file multi_pointer.h
 *
 * @brief This file contains the definition of the SYCL pointer classes as
 * defined by the SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_MULTI_POINTER_H_
#define RUNTIME_INCLUDE_SYCL_MULTI_POINTER_H_

#include "SYCL/abacus_types.h"
#include "SYCL/common.h"
#include "SYCL/vec.h"

#include <cstddef>

namespace cl {
namespace sycl {

// We need the accessor forward declaration to create pointer class constructor
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class accessor;

namespace detail {

/** @brief Type traits to convert cl::sycl::access::address_space into a
 * qualified
 * type with an address space.
 * @tparam dataType Type to which we append the address space.
 * @tparam asp The address space in which the type lives.
 */
template <typename dataType, cl::sycl::access::address_space asp>
struct address_space_trait {
  /** Template struct to delay the evaluation of the static_assert.
   * This allow the static_assert expression to only be evaluated if this struct
   * gets instantiated.
   *
   * If this address_space_trait struct get instantiated, the static_assert will
   * evaluate to false. This should only happen if one of the
   * address_space_trait partial specialization is missing.
   */
  template <cl::sycl::access::address_space delay>
  struct delayed_assert {
    enum { value = false };
  };
  static_assert(delayed_assert<asp>::value, "Unhandled address space.");
};

/** @internal
 * @brief @ref address_space_trait specialization for private address space
 * @tparam dataType Type to which we append the address space.
 */
template <typename dataType>
struct address_space_trait<dataType,
                           cl::sycl::access::address_space::private_space> {
  using original_type = dataType;
  using address_space_type = COMPUTECPP_CL_ASP_PRIVATE dataType;
  /** @brief There is no access target that would correspond to private_space,
   *        but we still need a valid target when doing SFINAE.
   */
  static constexpr bool hasTarget = false;
  static constexpr auto target = access::target::global_buffer;
};

/** @internal
 * @brief @ref address_space_trait specialization for global address space
 * @tparam dataType Type to which we append the address space.
 */
template <typename dataType>
struct address_space_trait<dataType,
                           cl::sycl::access::address_space::global_space> {
  using original_type = dataType;
  using address_space_type = COMPUTECPP_CL_ASP_GLOBAL dataType;
  static constexpr bool hasTarget = true;
  static constexpr auto target = access::target::global_buffer;
};

/** @internal
 * @brief @ref address_space_trait specialization for constant address space
 * @tparam dataType Type to which we append the address space.
 */
template <typename dataType>
struct address_space_trait<dataType,
                           cl::sycl::access::address_space::constant_space> {
  using original_type = dataType;
  using address_space_type = COMPUTECPP_CL_ASP_CONSTANT dataType;
  static constexpr bool hasTarget = true;
  static constexpr auto target = access::target::constant_buffer;
};

/** @internal
 * @brief @ref address_space_trait specialization for local address space
 * @tparam dataType Type to which we append the address space.
 */
template <typename dataType>
struct address_space_trait<dataType,
                           cl::sycl::access::address_space::local_space> {
  using original_type = dataType;
  using address_space_type = COMPUTECPP_CL_ASP_LOCAL dataType;
  static constexpr bool hasTarget = true;
  static constexpr auto target = access::target::local;
};

/** @internal
 * @brief @ref address_space_trait specialization for subgroup_local address
 * space
 * @tparam dataType Type to which we append the address space.
 */
template <typename dataType>
struct address_space_trait<dataType,
                           access::address_space::subgroup_local_space> {
  using original_type = dataType;
  using address_space_type = COMPUTECPP_CL_ASP_SUBGROUP_LOCAL dataType;
  static constexpr bool hasTarget = true;
  static constexpr auto target = access::target::subgroup_local;
};

/** @internal
 * @brief CRTP helper class to define host vector conversion operators without
 * having to specialize concrete classes. Vector class have extra conversion
 * that we need to handle.
 * The default instance (non-sycl-vector types) do nothing extra.
 * @tparam dataType Data type that the class manipulates.
 * @tparam DerivedT CRTP type, which is an instance of \ref ptr_class_base.
 */
template <typename dataType, cl::sycl::access::address_space asp,
          typename DerivedT>
class vec_ptr_class_base {};

/** @internal
 * @brief CRTP helper class to define host vector conversion operators without
 * having to specialize concrete classes. This is the only specialization we
 * have to do for vectors.
 * @tparam dataType Data type hold by the vector class that the class
 * manipulates.
 * @tparam width Vector width.
 * @tparam DerivedT CRTP type, which is an instance of \ref ptr_class_base.
 */
template <typename dataType, int width, cl::sycl::access::address_space asp,
          typename DerivedT>
class vec_ptr_class_base<vec<dataType, width>, asp, DerivedT> {
  DerivedT& derived() { return static_cast<DerivedT&>(*this); }

 public:
#ifdef __SYCL_DEVICE_ONLY__
  using vector_asp_type = typename detail::address_space_trait<
      detail::__sycl_vector<dataType, width>, asp>::address_space_type;

  operator vector_asp_type*() {
    return reinterpret_cast<vector_asp_type*>(derived().m_elem);
  }
#else
  /**
      Conversion operator between explicit pointer type that contains a
      cl::sycl::vector and the corresponding pointer of and abacus_vector
      type.
      This is needed for all of the math built-ins that take a pointer to
      a vector type.
  */
  using abacus_type =
      typename abacus::convert_abacus_sycl<vec<dataType, width>>::abacus_type;

  operator abacus_type*() {
    abacus_type* abacusVec = reinterpret_cast<abacus_type*>(derived().m_elem);
    return abacusVec;
  }
#endif
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename dataType, cl::sycl::access::address_space asp>
typename detail::address_space_trait<dataType, asp>::original_type*
get_pointer_internal_type(
    typename detail::address_space_trait<dataType, asp>::address_space_type*
        ptr);

template <typename dataType, cl::sycl::access::address_space asp>
typename detail::address_space_trait<
    dataType, asp>::address_space_type* __attribute__((__offload__))
get_pointer_internal_type(
    typename detail::address_space_trait<dataType, asp>::address_space_type*
        ptr) {
  return ptr;
}

template <typename dataType, cl::sycl::access::address_space asp>
typename detail::address_space_trait<
    dataType, asp>::address_space_type* __attribute__((__offload__(2)))
get_pointer_internal_type(
    typename detail::address_space_trait<dataType, asp>::address_space_type*
        ptr) {
  return ptr;
}
#else   //  __SYCL_DEVICE_ONLY__
/** Internal hooks for address space deduction.
 */
template <typename dataType, cl::sycl::access::address_space asp>
typename detail::address_space_trait<dataType, asp>::address_space_type*
get_pointer_internal_type(
    typename detail::address_space_trait<dataType, asp>::address_space_type*
        ptr) {
  return ptr;
}
#endif  // __SYCL_DEVICE_ONLY__

////////////////////////////////////////////////////////////////////////////////
// multi_ptr_base

/** @brief Base class for all instances of multi_ptr.
 *
 *        Implements most of the functionality of multi_ptr, but leaves out
 *        reference types because they're not supported by the void
 *        specialization.
 * @tparam dataType Data type the object manipulates.
 * @tparam asp The address space the pointer class points to.
 */
template <typename dataType, cl::sycl::access::address_space asp>
class multi_ptr_base
    : public detail::vec_ptr_class_base<dataType, asp,
                                        multi_ptr_base<dataType, asp>> {
 protected:
  using asp_type =
      typename detail::address_space_trait<dataType, asp>::address_space_type;
  using original_type_ptr =
      typename detail::address_space_trait<dataType, asp>::original_type*;

  /** @brief Underlying type of the contained pointer
   */
  using element_type = dataType;

  /** @brief Type for offsetting pointers
   */
  using difference_type = std::ptrdiff_t;

  /** @brief Raw pointer definition.
   */
  using pointer_t = asp_type*;

  /** @brief Raw pointer-to-const definition.
   */
  using const_pointer_t = const asp_type*;

  using value_type = dataType;
  using pointer = asp_type*;
  using iterator_category = std::random_access_iterator_tag;

 public:
  /** The address space that this multi_ptr class handles
   */
  static constexpr cl::sycl::access::address_space address_space = asp;

  //////////////////////////////////////////////////////////////////////////////
  // Constructors

  /** @brief Default constructor
   */
  multi_ptr_base() : m_elem(nullptr) {}

  /** @brief Initialize the object using the given pointer.
   * @param ptr Pointer that the class should manipulate.
   */
  multi_ptr_base(pointer_t ptr) : m_elem(ptr) {}

 protected:
  friend class detail::vec_ptr_class_base<dataType, asp,
                                          multi_ptr_base<dataType, asp>>;

  original_type_ptr get_pointer_internal() const {
    return detail::get_pointer_internal_type<dataType, asp>(m_elem);
  }

  original_type_ptr get_pointer_internal() const volatile {
    return detail::get_pointer_internal_type<dataType, asp>(m_elem);
  }

  /** @brief Move the underlying pointer by r elements
   * @param r How many elements to move the pointer by
   */
  inline void increment_pointer(difference_type r) { m_elem += r; }

 public:
  //////////////////////////////////////////////////////////////////////////////
  // Pointer retrieval

  /** @return Returns the underlying OpenCL C pointer
   */
  inline pointer_t get() const { return m_elem; }

  /** @return Pointer to the data the object points to.
   */
  original_type_ptr operator->() const { return get_pointer_internal(); }

  //////////////////////////////////////////////////////////////////////////////
  // Conversion operators

  /** Cast operator to the internal pointer representation
   */
  operator original_type_ptr() const { return get_pointer_internal(); }

  /** Cast operator to the internal pointer representation
   */
  operator original_type_ptr() const volatile { return get_pointer_internal(); }

#if defined(__SYCL_DEVICE_ONLY__)
  /** Cast operator to the internal pointer representation
   */
  template <typename U = pointer_t>
  operator typename std::enable_if<!std::is_same<original_type_ptr, U>::value,
                                   U>::type() {
    return m_elem;
  }
  /** Cast operator to the internal pointer representation
   */
  template <typename U = pointer_t>
  operator typename std::enable_if<!std::is_same<original_type_ptr, U>::value,
                                   U>::type() const {
    return m_elem;
  }
  /** Cast operator to the internal pointer representation
   */
  template <typename U = pointer_t>
  operator typename std::enable_if<!std::is_same<original_type_ptr, U>::value,
                                   U>::type() volatile {
    return m_elem;
  }
  /** Cast operator to the internal pointer representation
   */
  template <typename U = pointer_t>
  operator typename std::enable_if<!std::is_same<original_type_ptr, U>::value,
                                   U>::type() const volatile {
    return m_elem;
  }
#endif  // defined(__SYCL_DEVICE_ONLY__)

  //////////////////////////////////////////////////////////////////////////////
  // prefetch

  /** @brief Prefetches a number of elements specified by numElements
   *        into the global memory cache.
   *
   *        This operation is an implementation defined optimization.
   * @tparam COMPUTECPP_ENABLE_IF Only available for the global address space
   * @param numElements Number of elements to prefetch
   */
  template <COMPUTECPP_ENABLE_IF(dataType,
                                 (asp == access::address_space::global_space))>
  void prefetch(size_t /*numElements*/) const {
    // Note: not implemented
  }

 private:
  /** @brief The pointer that this object handles
   */
  pointer_t m_elem;
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// multi_ptr

/** @brief multi_ptr, generic pointer class. This class have the same interface
 * as the explicit pointer classes (global_ptr, private_ptr, local_ptr and
 * constant_ptr). The address space where the data point to is defined by the
 * template parameter Space. A cast operator allow the conversion from a
 * multi_ptr object to its equivalent explicit one.
 * @tparam dataType Data type the object manipulates.
 * @tparam asp The address space the pointer class points to.
 */
template <typename dataType, cl::sycl::access::address_space asp>
class multi_ptr : public detail::multi_ptr_base<dataType, asp> {
 protected:
  using multi_ptr_base = detail::multi_ptr_base<dataType, asp>;

  using original_type_ref =
      typename detail::address_space_trait<dataType, asp>::original_type&;
  using original_type_cref =
      const typename detail::address_space_trait<dataType, asp>::original_type&;

  /** @brief Alias to the non-const qualified data type
   */
  using non_const_data_t = typename std::remove_const<dataType>::type;

 public:
  /// @cond COMPUTECPP_DEV

  /** Pointer type independent of the SYCL version
   * @note Internal use only
   */
  using ptr_t = typename multi_ptr_base::pointer;

  /// COMPUTECPP_DEV @endcond

  /** @brief Raw reference definition.
   */
  using reference_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(2020, "Use reference") =
      typename multi_ptr_base::asp_type&;

  /** @brief Raw reference-to-const definition.
   */
  using const_reference_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      2020,
      "Use add_const_t<reference>") = const typename multi_ptr_base::asp_type&;

  using element_type COMPUTECPP_DEPRECATED_BY_SYCL_VER(2020, "Use value_type") =
      typename multi_ptr_base::element_type;
  using difference_type = typename multi_ptr_base::difference_type;
  using pointer_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(2020,
                                                    "Use pointer") = ptr_t;
  using const_pointer_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      2020, "Use std::add_const_t<pointer>") =
      typename multi_ptr_base::const_pointer_t;

#if SYCL_LANGUAGE_VERSION >= 2020
  /// Underlying data type
  using value_type = typename multi_ptr_base::value_type;
  /// Underlying raw pointer type
  using pointer = typename multi_ptr_base::pointer;
  /// Reference to underlying data type
  using reference = typename multi_ptr_base::asp_type&;
  /// multi_ptr can be used as a random access iterator
  using iterator_category = typename multi_ptr_base::iterator_category;
#endif  // SYCL_LANGUAGE_VERSION

  //////////////////////////////////////////////////////////////////////////////
  // Constructors

  /** @brief Default constructor
   */
  multi_ptr() = default;

  /** @brief Initialize the object using the given pointer.
   * @param ptr Pointer that the class should manipulate.
   */
  multi_ptr(ptr_t ptr) : multi_ptr_base(ptr) {}

  /** @brief Copy constructor from a non-const multi_ptr
   * @tparam COMPUTECPP_ENABLE_IF Only available when dataType is const
   * @param rhs Non-const multi_ptr
   */
  template <COMPUTECPP_ENABLE_IF(dataType, (std::is_const<dataType>::value))>
  multi_ptr(const multi_ptr<non_const_data_t, asp>& rhs)
      : multi_ptr_base(rhs.get()) {}

  /** @brief Copy constructor from a void multi_ptr
   * @param rhs void multi_ptr
   */
  multi_ptr(const multi_ptr<void, asp>& rhs)
      : multi_ptr_base(static_cast<dataType*>(rhs.get())) {}

  /** @brief Copy constructor from a const void multi_ptr
   * @tparam COMPUTECPP_ENABLE_IF Only available when dataType is const
   * @param rhs const void multi_ptr
   */
  template <COMPUTECPP_ENABLE_IF(dataType, (std::is_const<dataType>::value))>
  multi_ptr(const multi_ptr<const void, asp>& rhs)
      : multi_ptr_base(static_cast<const dataType*>(rhs.get())) {}

  /** @brief Initialize the object using an accessor to non-const data
   * @tparam dimensions Accessor dimensions
   * @tparam Mode Accessor mode
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @tparam COMPUTECPP_ENABLE_IF This constructor is only available for
   *         access::address_space::global_space
   *         access::address_space::constant_space
   *         access::address_space::local_space.
   *         It is available regardless of dataType being const or not.
   * @param acc Accessor to retrieve the pointer from
   */
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder = access::placeholder::false_t,
            COMPUTECPP_ENABLE_IF(dataType, (detail::address_space_trait<
                                               dataType, asp>::hasTarget))>
  multi_ptr(cl::sycl::accessor<
            non_const_data_t, dimensions, Mode,
            detail::address_space_trait<non_const_data_t, asp>::target,
            isPlaceholder>
                acc)
      : multi_ptr(acc.get_pointer()) {}

  /** @brief Initialize the object using an accessor to const data
   * @tparam dimensions Accessor dimensions
   * @tparam Mode Accessor mode
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @tparam COMPUTECPP_ENABLE_IF This constructor is only available for
   *         access::address_space::global_space
   *         access::address_space::constant_space
   *         access::address_space::local_space.
   *         Available only if dataType is const.
   * @param acc Accessor to retrieve the pointer from
   */
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder = access::placeholder::false_t,
            COMPUTECPP_ENABLE_IF(dataType, (detail::address_space_trait<
                                                dataType, asp>::hasTarget &&
                                            std::is_const<dataType>::value))>
  multi_ptr(cl::sycl::accessor<
            const non_const_data_t, dimensions, Mode,
            detail::address_space_trait<const non_const_data_t, asp>::target,
            isPlaceholder>
                acc)
      : multi_ptr(acc.get_pointer()) {}

  /** @brief Initialize the object using the given
   *        non address space qualified pointer.
   *
   *        This conversion is defined by the device compiler.
   *
   * @tparam COMPUTECPP_ENABLE_IF Only available if the provided pointer
   *         is not address space qualified.
   * @param ptr Pointer that is not address space qualified
   *        that the class should manipulate
   * @note This constructor has to be declared in order for the device compiler
   *       to deduce address spaces, but it should not be defined because it
   *       should never actually be used.
   */
  template <COMPUTECPP_ENABLE_IF(dataType,
                                 (!std::is_same<ptr_t, dataType*>::value))>
  multi_ptr(dataType* ptr);

  //////////////////////////////////////////////////////////////////////////////
  // Reference retrieval

  /** @return Reference to the data the object points to.
   */
  original_type_ref operator*() { return *this->get_pointer_internal(); }

  /** @return Const reference to the data the object points to.
   */
  original_type_cref operator*() const { return *this->get_pointer_internal(); }

  /** @param i Index.
   * @return Reference to the i-th element the object points to.
   * @deprecated Use operator* or operator-> instead
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      2017, "multi_ptr::operator[] is no longer available.")
  original_type_ref operator[](size_t i) {
    return this->get_pointer_internal()[i];
  }

  /** @param i Index.
   * @return Const reference to the i-th element the object points to.
   * @deprecated Use operator* or operator-> instead
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      2017, "multi_ptr::operator[] is no longer available.")
  original_type_cref operator[](size_t i) const {
    return this->get_pointer_internal()[i];
  }

  //////////////////////////////////////////////////////////////////////////////
  // Arithmetic operators

  /** @brief Increments the underlying pointer by 1
   * @return This object
   */
  inline multi_ptr& operator++() {
    this->increment_pointer(1);
    return *this;
  }

  /** @brief Increments the underlying pointer by 1 and returns a new multi_ptr
   *        with the value of the previous pointer
   * @return New multi_ptr object with the old pointer value
   */
  inline multi_ptr operator++(int) {
    auto copy = multi_ptr(*this);
    this->operator++();
    return copy;
  }

  /** @brief Decrements the underlying pointer by 1
   * @return This object
   */
  inline multi_ptr& operator--() {
    this->increment_pointer(-1);
    return *this;
  }

  /** @brief Decrements the underlying pointer by 1 and returns a new multi_ptr
   *        with the value of the previous pointer
   * @return New multi_ptr object with the old pointer value
   */
  inline multi_ptr operator--(int) {
    auto copy = multi_ptr(*this);
    this->operator--();
    return copy;
  }

  /** @brief Increments the underlying pointer by r
   * @param r Number of elements to increment the underlying pointer by
   * @return This object
   */
  inline multi_ptr& operator+=(difference_type r) {
    this->increment_pointer(r);
    return *this;
  }

  /** @brief Creates a new multi_ptr that points r forward compared to *this
   * @param r Number of elements to increment the underlying pointer by
   * @return New multi_ptr object with the pointer advanced by r
   */
  inline multi_ptr operator+(difference_type r) const {
    return multi_ptr(this->get_pointer_internal() + r);
  }

  /** @brief Decrements the underlying pointer by r
   * @param r Number of elements to decrement the underlying pointer by
   * @return This object
   */
  inline multi_ptr& operator-=(difference_type r) {
    this->increment_pointer(-r);
    return *this;
  }

  /** @brief Creates a new multi_ptr that points r backward compared to *this
   * @param r Number of elements to decrement the underlying pointer by
   * @return New multi_ptr object with the pointer moved back by r
   */
  inline multi_ptr operator-(difference_type r) const {
    return multi_ptr(this->get_pointer_internal() - r);
  }
};

/** @brief Generic pointer class, specialization for void.
 * @tparam asp The address space the pointer class points to.
 */
template <cl::sycl::access::address_space asp>
class multi_ptr<void, asp> : public detail::multi_ptr_base<void, asp> {
 protected:
  using multi_ptr_base = detail::multi_ptr_base<void, asp>;

 public:
  /// @cond COMPUTECPP_DEV

  /** Pointer type independent of the SYCL version
   * @note Internal use only
   */
  using ptr_t = typename multi_ptr_base::pointer;

  /// COMPUTECPP_DEV @endcond

  using element_type COMPUTECPP_DEPRECATED_BY_SYCL_VER(2020, "Use value_type") =
      typename multi_ptr_base::element_type;
  using difference_type = typename multi_ptr_base::difference_type;
  using pointer_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(2020, "Use pointer") =
      typename multi_ptr_base::pointer_t;
  using const_pointer_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      2020, "Use std::add_const_t<pointer>") =
      typename multi_ptr_base::const_pointer_t;

#if SYCL_LANGUAGE_VERSION >= 2020
  /// Underlying data type
  using value_type = typename multi_ptr_base::value_type;
  /// Underlying raw pointer type
  using pointer = typename multi_ptr_base::pointer;
  /// multi_ptr can be used as a random access iterator
  using iterator_category = typename multi_ptr_base::iterator_category;
#endif  // SYCL_LANGUAGE_VERSION

  //////////////////////////////////////////////////////////////////////////////
  // Constructors

  /** @brief Default constructor
   */
  multi_ptr() = default;

  /** @brief Initialize the object using the given pointer.
   * @param ptr Pointer that the class should manipulate.
   */
  multi_ptr(ptr_t ptr) : multi_ptr_base(ptr) {}

  /** @brief Initialize the object using an accessor
   * @tparam dimensions Accessor dimensions
   * @tparam Mode Accessor mode
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @tparam COMPUTECPP_ENABLE_IF This constructor is only available for
   *         access::address_space::global_space
   *         access::address_space::constant_space
   *         access::address_space::local_space
   * @param acc Accessor to retrieve the pointer from
   */
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder = access::placeholder::false_t,
            COMPUTECPP_ENABLE_IF(
                void, (detail::address_space_trait<void, asp>::hasTarget))>
  multi_ptr(cl::sycl::accessor<void, dimensions, Mode,
                               detail::address_space_trait<void, asp>::target,
                               isPlaceholder>
                acc)
      : multi_ptr(acc.get_pointer()) {}

  /** @brief Initialize the object using the given
   *        non address space qualified pointer.
   *
   *        This conversion is defined by the device compiler.
   *
   * @tparam COMPUTECPP_ENABLE_IF Only available if the provided pointer
   *         is not address space qualified.
   * @param ptr Pointer that is not address space qualified
   *        that the class should manipulate
   * @note This constructor has to be declared in order for the device compiler
   *       to deduce address spaces, but it should not be defined because it
   *       should never actually be used.
   */
  template <COMPUTECPP_ENABLE_IF(void, (!std::is_same<ptr_t, void*>::value))>
  multi_ptr(void* ptr);

  /** @brief Explicit conversion from a multi_ptr<ElementType>
   * @tparam ElementType Underlying type of the pointer to convert
   * @tparam COMPUTECPP_ENABLE_IF Only enabled if ElementType is not void
   * @param ptr Pointer to convert to multi_ptr<void>
   */
  template <typename ElementType,
            COMPUTECPP_ENABLE_IF(ElementType,
                                 (!std::is_same<ElementType, void>::value))>
  explicit multi_ptr(const multi_ptr<ElementType, asp>& ptr)
      : multi_ptr_base(static_cast<ptr_t>(ptr.get())) {}

  /** @brief Initialize the object using an accessor
   * @tparam ElementType Underlying type of the accessor data
   * @tparam dimensions Accessor dimensions
   * @tparam Mode Accessor mode
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @tparam COMPUTECPP_ENABLE_IF This constructor is only available for
   *         access::address_space::global_space
   *         access::address_space::constant_space
   *         access::address_space::local_space
   * @param acc Accessor to retrieve the pointer from
   */
  template <typename ElementType, int dimensions, access::mode Mode,
            access::placeholder isPlaceholder,
            COMPUTECPP_ENABLE_IF(
                ElementType,
                (detail::address_space_trait<ElementType, asp>::hasTarget))>
  multi_ptr(accessor<ElementType, dimensions, Mode,
                     detail::address_space_trait<ElementType, asp>::target,
                     isPlaceholder>
                acc)
      : multi_ptr_base(acc.get_pointer().get()) {}

  /** @brief Explicit conversion to a multi_ptr<ElementType>
   * @return This pointer object converted to a multi_ptr<ElementType>
   */
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, asp>() const {
    using elem_ptr_t =
        typename detail::address_space_trait<ElementType, asp>::original_type*;
    return multi_ptr<ElementType, asp>(
        static_cast<elem_ptr_t>(this->get_pointer_internal()));
  }
};

/** @brief Generic pointer class, specialization for const void.
 * @tparam asp The address space the pointer class points to.
 */
template <cl::sycl::access::address_space asp>
class multi_ptr<const void, asp>
    : public detail::multi_ptr_base<const void, asp> {
 protected:
  using multi_ptr_base = detail::multi_ptr_base<const void, asp>;

 public:
  /// @cond COMPUTECPP_DEV

  /** Pointer type independent of the SYCL version
   * @note Internal use only
   */
  using ptr_t = typename multi_ptr_base::pointer;

  /// COMPUTECPP_DEV @endcond

  using element_type COMPUTECPP_DEPRECATED_BY_SYCL_VER(2020, "Use value_type") =
      typename multi_ptr_base::element_type;
  using difference_type = typename multi_ptr_base::difference_type;
  using pointer_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(2020, "Use pointer") =
      typename multi_ptr_base::pointer_t;
  using const_pointer_t COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      2020, "Use std::add_const_t<pointer>") =
      typename multi_ptr_base::const_pointer_t;

#if SYCL_LANGUAGE_VERSION >= 2020
  /// Underlying data type
  using value_type = typename multi_ptr_base::value_type;
  /// Underlying raw pointer type
  using pointer = typename multi_ptr_base::pointer;
  /// multi_ptr can be used as a random access iterator
  using iterator_category = typename multi_ptr_base::iterator_category;
#endif  // SYCL_LANGUAGE_VERSION

  //////////////////////////////////////////////////////////////////////////////
  // Constructors

  /** @brief Default constructor
   */
  multi_ptr() = default;

  /** @brief Initialize the object using the given pointer.
   * @param ptr Pointer that the class should manipulate.
   */
  multi_ptr(ptr_t ptr) : multi_ptr_base(ptr) {}

  /** @brief Initialize the object using an accessor
   * @tparam dimensions Accessor dimensions
   * @tparam Mode Accessor mode
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @tparam COMPUTECPP_ENABLE_IF This constructor is only available for
   *         access::address_space::global_space
   *         access::address_space::constant_space
   *         access::address_space::local_space
   * @param acc Accessor to retrieve the pointer from
   */
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder = access::placeholder::false_t,
            COMPUTECPP_ENABLE_IF(const void, (detail::address_space_trait<
                                                 const void, asp>::hasTarget))>
  multi_ptr(cl::sycl::accessor<
            const void, dimensions, Mode,
            detail::address_space_trait<const void, asp>::target, isPlaceholder>
                acc)
      : multi_ptr(acc.get_pointer()) {}

  /** @brief Initialize the object using the given
   *        non address space qualified pointer.
   *
   *        This conversion is defined by the device compiler.
   *
   * @tparam COMPUTECPP_ENABLE_IF Only available if the provided pointer
   *         is not address space qualified.
   * @param ptr Pointer that is not address space qualified
   *        that the class should manipulate
   * @note This constructor has to be declared in order for the device compiler
   *       to deduce address spaces, but it should not be defined because it
   *       should never actually be used.
   */
  template <COMPUTECPP_ENABLE_IF(const void,
                                 (!std::is_same<ptr_t, const void*>::value))>
  multi_ptr(const void* ptr);

  /** @brief Explicit conversion from a multi_ptr<ElementType>
   * @tparam ElementType Underlying type of the pointer to convert
   * @tparam COMPUTECPP_ENABLE_IF Only enabled if ElementType is not const void
   * @param ptr Pointer to convert to multi_ptr<const void>
   */
  template <typename ElementType,
            COMPUTECPP_ENABLE_IF(
                ElementType, (!std::is_same<ElementType, const void>::value))>
  explicit multi_ptr(const multi_ptr<ElementType, asp>& ptr)
      : multi_ptr_base(static_cast<ptr_t>(ptr.get())) {}

  /** @brief Initialize the object using an accessor
   * @tparam ElementType Underlying type of the accessor data
   * @tparam dimensions Accessor dimensions
   * @tparam Mode Accessor mode
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @tparam COMPUTECPP_ENABLE_IF This constructor is only available for
   *         access::address_space::global_space
   *         access::address_space::constant_space
   *         access::address_space::local_space
   * @param acc Accessor to retrieve the pointer from
   */
  template <typename ElementType, int dimensions, access::mode Mode,
            access::placeholder isPlaceholder,
            COMPUTECPP_ENABLE_IF(
                ElementType,
                (detail::address_space_trait<ElementType, asp>::hasTarget))>
  multi_ptr(accessor<ElementType, dimensions, Mode,
                     detail::address_space_trait<ElementType, asp>::target,
                     isPlaceholder>
                acc)
      : multi_ptr_base(acc.get_pointer().get()) {}

  /** @brief Explicit conversion to a multi_ptr<ElementType>
   * @return This pointer object converted to a multi_ptr<ElementType>
   */
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, asp>() const {
    using elem_ptr_t =
        typename detail::address_space_trait<ElementType, asp>::original_type*;
    return multi_ptr<ElementType, asp>(
        static_cast<elem_ptr_t>(this->get_pointer_internal()));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Explicit pointer class aliases

/** @brief global_ptr pointer class definition for data pointing to the OpenCL
 * global address space.
 * @tparam dataType Data type the object manipulates.
 */
template <typename dataType>
using global_ptr = multi_ptr<dataType, access::address_space::global_space>;

/** @brief local_ptr pointer class definition for data pointing to the OpenCL
 * local address space
 * @tparam dataType Data type the object manipulates.
 */
template <typename dataType>
using local_ptr = multi_ptr<dataType, access::address_space::local_space>;

/** @brief private_ptr, pointer class definition for data pointing to the OpenCL
 * private address space
 * @tparam dataType Data type the object manipulates.
 */
template <typename dataType>
using private_ptr = multi_ptr<dataType, access::address_space::private_space>;

/** @brief constant_ptr, pointer class definition for data living in the OpenCL
 * constant address space
 * @tparam dataType Data type the object manipulates.
 */
template <typename dataType>
using constant_ptr = multi_ptr<dataType, access::address_space::constant_space>;

namespace codeplay {
/** @brief subgroup_local_ptr, pointer class definition for data living in the
 * subgroup local address space.
 * @tparam dataType Data type the object manipulates.
 */
template <typename dataType>
using subgroup_local_ptr =
    multi_ptr<dataType, access::address_space::subgroup_local_space>;
}  // namespace codeplay

////////////////////////////////////////////////////////////////////////////////
// make_ptr

/** @brief Create a multi_ptr object from a raw pointer.
 * @tparam dataType Type of the pointed values.
 * @tparam Space Address space in which the data live.
 * @param ptr The raw pointer from which to create the multi_ptr.
 * @return A multi_ptr object pointing to the same address pointed as by ptr.
 */
template <typename dataType, cl::sycl::access::address_space Space>
typename std::enable_if<
    !std::is_same<typename multi_ptr<dataType, Space>::ptr_t, dataType*>::value,
    multi_ptr<dataType, Space>>::type
make_ptr(dataType* ptr) {
  return multi_ptr<dataType, Space>(ptr);
}

template <typename dataType, cl::sycl::access::address_space Space>
multi_ptr<dataType, Space> make_ptr(
    typename multi_ptr<dataType, Space>::ptr_t ptr) {
  return multi_ptr<dataType, Space>(ptr);
}

}  // namespace sycl
}  // namespace cl

/******************************************/
/* Operators definitions: explicit pointer */
/******************************************/

/// @cond COMPUTECPP_DEV

/**  Define logic operators (==, !=, <, >, <=, >=) for pointer classes: @ref
 * private_ptr, @ref gloabl_ptr, @ref constant_ptr, @ref constant_ptr
 */
#define COMPUTECPP_EXPLICIT_POINTER_OP_DEFINITION(explicit_ptr)            \
  template <typename dataType>                                             \
  bool operator==(const explicit_ptr<dataType>& lhs,                       \
                  const explicit_ptr<dataType>& rhs) {                     \
    return (lhs.get() == rhs.get());                                       \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator!=(const explicit_ptr<dataType>& lhs,                       \
                  const explicit_ptr<dataType>& rhs) {                     \
    return !(lhs == rhs);                                                  \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator<(const explicit_ptr<dataType>& lhs,                        \
                 const explicit_ptr<dataType>& rhs) {                      \
    return (lhs.get() < rhs.get());                                        \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator>(const explicit_ptr<dataType>& lhs,                        \
                 const explicit_ptr<dataType>& rhs) {                      \
    return (lhs.get() > rhs.get());                                        \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator>=(const explicit_ptr<dataType>& lhs,                       \
                  const explicit_ptr<dataType>& rhs) {                     \
    return (lhs == rhs) || (lhs > rhs);                                    \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator<=(const explicit_ptr<dataType>& lhs,                       \
                  const explicit_ptr<dataType>& rhs) {                     \
    return (lhs == rhs) || (lhs < rhs);                                    \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator==(const explicit_ptr<dataType>& lhs, std::nullptr_t rhs) { \
    return (lhs.get() == rhs);                                             \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator==(std::nullptr_t lhs, const explicit_ptr<dataType>& rhs) { \
    return rhs == lhs;                                                     \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator!=(const explicit_ptr<dataType>& lhs, std::nullptr_t rhs) { \
    return !(lhs == rhs);                                                  \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator!=(std::nullptr_t lhs, const explicit_ptr<dataType>& rhs) { \
    return !(lhs == rhs);                                                  \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator>(const explicit_ptr<dataType>& lhs, std::nullptr_t rhs) {  \
    /* anything > nullptr => nullptr != anything*/                         \
    return lhs != rhs;                                                     \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator>(std::nullptr_t lhs, const explicit_ptr<dataType>& rhs) {  \
    /*  nullptr > anything => false */                                     \
    return false;                                                          \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator<(const explicit_ptr<dataType>& lhs, std::nullptr_t rhs) {  \
    /*  anything < nullptr => false */                                     \
    return false;                                                          \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator<(std::nullptr_t lhs, const explicit_ptr<dataType>& rhs) {  \
    /*  nullptr < anything => nullptr != anything */                       \
    return lhs != rhs;                                                     \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator>=(const explicit_ptr<dataType>& lhs, std::nullptr_t rhs) { \
    return true;                                                           \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator>=(std::nullptr_t lhs, const explicit_ptr<dataType>& rhs) { \
    /*  nullptr >= anything => nullptr == anything */                      \
    return lhs == rhs;                                                     \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator<=(const explicit_ptr<dataType>& lhs, std::nullptr_t rhs) { \
    /*  anything <= nullptr => anything == nullptr */                      \
    return lhs == rhs;                                                     \
  }                                                                        \
  template <typename dataType>                                             \
  bool operator<=(std::nullptr_t lhs, const explicit_ptr<dataType>& rhs) { \
    /*  nullptr <= anything => anything == nullptr */                      \
    return true;                                                           \
  }

COMPUTECPP_EXPLICIT_POINTER_OP_DEFINITION(cl::sycl::private_ptr)
COMPUTECPP_EXPLICIT_POINTER_OP_DEFINITION(cl::sycl::global_ptr)
COMPUTECPP_EXPLICIT_POINTER_OP_DEFINITION(cl::sycl::constant_ptr)
COMPUTECPP_EXPLICIT_POINTER_OP_DEFINITION(cl::sycl::local_ptr)
COMPUTECPP_EXPLICIT_POINTER_OP_DEFINITION(
    cl::sycl::codeplay::subgroup_local_ptr)

#undef COMPUTECPP_EXPLICIT_POINTER_OP_DEFINITION

/// COMPUTECPP_DEV @endcond

/***********************************/
/* Operators definitions: multi_ptr */
/***********************************/

/** Equal operator.
 * \return True if both parameter points to the same memory location, false
 * otherwise.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator==(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs.get_pointer() == rhs.get_pointer();
}

/** Not equal operator.
 * \return True if the parameters does not point to the same memory location,
 * false
 * otherwise.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator!=(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs.get_pointer() != rhs.get_pointer();
}

/** Less than operator.
 * \return True if the lhs parameter address is strictly less than rhs, false
 * otherwise.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator<(const cl::sycl::multi_ptr<dataType, Space>& lhs,
               const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs.get_pointer() < rhs.get_pointer();
}

/** Greater than operator.
 * \return True if the lhs parameter address is strictly greater than rhs, false
 * otherwise.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator>(const cl::sycl::multi_ptr<dataType, Space>& lhs,
               const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs.get_pointer() > rhs.get_pointer();
}

/** Less or equal than operator.
 * \return True if the lhs parameter address is less or equal to rhs, false
 * otherwise.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator<=(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs.get_pointer() <= rhs.get_pointer();
}

/** Greater or equal than operator.
 * \return True if the lhs parameter address is greater or equal to rhs, false
 * otherwise.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator>=(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs.get_pointer() >= rhs.get_pointer();
}

/** Not equal operator.
 * \return True if the lhs parameter is not null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator!=(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                std::nullptr_t rhs) {
  return lhs.get_pointer() != rhs;
}

/** Not equal operator.
 * \return True if the rhs parameter is not null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator!=(std::nullptr_t lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs != rhs.get_pointer();
}

/** Equal operator.
 * \return True if the lhs parameter is null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator==(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                std::nullptr_t rhs) {
  return lhs.get_pointer() == rhs;
}

/** Equal operator.
 * \return True if the rhs parameter is null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator==(std::nullptr_t lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs == rhs.get_pointer();
}

/** Greater than operator.
 * \return True if the lhs parameter is strictly greater than null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator>(const cl::sycl::multi_ptr<dataType, Space>& lhs,
               std::nullptr_t rhs) {
  return lhs.get_pointer() > rhs;
}

/** Greater than operator.
 * \return False
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator>(std::nullptr_t lhs,
               const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs > rhs.get_pointer();
}

/** Less than operator.
 * \return True if the lhs parameter is strictly less than null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator<(const cl::sycl::multi_ptr<dataType, Space>& lhs,
               std::nullptr_t rhs) {
  return lhs.get_pointer() < rhs;
}

/** Less than operator.
 * \return True if the rhs parameter is strictly greater than null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator<(std::nullptr_t lhs,
               const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs < rhs.get_pointer();
}

/** Greater or equal than operator.
 * \return True if the lhs parameter is greater or equal to null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator>=(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                std::nullptr_t rhs) {
  return lhs.get_pointer() >= rhs;
}

/** Greater or equal than operator.
 * \return True if the rhs parameter is less or equal to null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator>=(std::nullptr_t lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs >= rhs.get_pointer();
}

/** Less or equal than operator.
 * \return True if the lhs parameter is less or equal to null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator<=(const cl::sycl::multi_ptr<dataType, Space>& lhs,
                std::nullptr_t rhs) {
  return lhs.get_pointer() <= rhs;
}

/** Less or equal than operator.
 * \return True if the rhs parameter is greater or equal to null.
 */
template <typename dataType, cl::sycl::access::address_space Space>
bool operator<=(std::nullptr_t lhs,
                const cl::sycl::multi_ptr<dataType, Space>& rhs) {
  return lhs <= rhs.get_pointer();
}

#endif  // RUNTIME_INCLUDE_SYCL_MULTI_POINTER_H_
