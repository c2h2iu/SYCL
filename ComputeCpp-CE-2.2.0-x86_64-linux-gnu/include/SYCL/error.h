/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 * @file error.h
 *
 * @brief Provides SYCL exception and exception list types.
 */
#ifndef RUNTIME_INCLUDE_SYCL_ERROR_H_
#define RUNTIME_INCLUDE_SYCL_ERROR_H_

#include "SYCL/base.h"
#include "SYCL/include_opencl.h"
#include "SYCL/predefines.h"

#include <exception>
#include <functional>
#include <iosfwd>
#include <memory>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
class context;
namespace detail {
struct sycl_log;
}  // namespace detail

/** @brief SYCL exception class, defined Section 3.2 of the specification,
 * for general SYCL error.
 *
 * This implementation adds extra methods to those defined in the
 * specification to provide additional information to the user.
 */
class COMPUTECPP_EXPORT exception {
 public:
  /// @cond COMPUTECPP_DEV
  /** @brief Constructs a exception from a sycl_log.
   * @param syclLog The sycl_log to be associated with the error.
   * @param context Shared pointer to a detail context if applies
   */
  explicit exception(std::unique_ptr<detail::sycl_log>&& syclLog,
                     dcontext_shptr context = nullptr);
  /// COMPUTECPP_DEV @endcond

  /** @brief Overload of std::runtime_error::what() which returns the message
   * associated with the error.
   * @return The message associated with the error.
   */
  const char* what() const noexcept;

  /** @brief Reports whether the exception has a context associated with it
   * @return True if a context is associated with this exception
   */
  bool has_context() const;

  /** @brief Returns the SYCL context that is associated with this SYCL
   * exception
   *
   * If no context is associated with this exception, it throws a new exception.
   *
   * @return Context that is associated with this exception
   */
  cl::sycl::context get_context() const;

  /** @brief Returns the OpenCL error code. Value extracted directly from
   * the OpenCL header.
   * @return int OpenCL error code
   */
  cl_int get_cl_code() const;

  /// @cond COMPUTECPP_DEV

  /** @brief Returns the SYCL error message.
   * @return The SYCL error message. The pointer is valid for the lifetime of
   * the exception; if required after that a copy of the null terminated string
   * must be made.
   */
  const char* get_description() const;

  /** @brief Returns the file name that trigger the error.
   * @return The file name.
   */
  const char* get_file_name() const;

  /** @brief Returns the line number that trigger the error.
   * @return The line number.
   */
  int get_line_number() const;

  /** @brief Returns an internal ComputeCpp error code from the error
   * @return The ComputeCpp specific error code representing the error
   */
  detail::cpp_error_code get_cpp_error_code() const;

  /** @brief Returns the name of the OpenCL error macro
   * @return const char * Name of the macro in human-readable-format
   */
  const char* get_cl_error_message() const;

  /// COMPUTECPP_DEV @endcond

 protected:
  /// @cond COMPUTECPP_DEV
  /** @brief Pointer to sycl_log containing the message and other information.
   *
   * Note: Either this must be a copyable pointer or an explicit copy
   * constructor needs to be provided for exceptions to allow the use of
   * `std::make_exception_ptr` which takes an excpetion by value.
   */
  std::shared_ptr<detail::sycl_log> m_syclLog;

  /* m_context.
   * If the SYCL exception was caused by a context, this will hold
   * a pointer to the context that caused the problem.
   */
  dcontext_shptr m_context;
  /// COMPUTECPP_DEV @endcond
};

/** @brief Class used to store exception objects and transfer them across
 * thread, equivalent to std::exception_ptr.
 */
using exception_ptr_class = std::exception_ptr;

/** @brief List of exceptions thrown asynchronously,
 * contains objects of type exception_ptr_class.
 *
 * The method add_exception has to be called from a derived or
 * friend class, it cannot be accessed directly by the user.
 */
class COMPUTECPP_EXPORT exception_list {
  friend COMPUTECPP_EXPORT exception_list* make_exception_list();

  friend COMPUTECPP_EXPORT void add_exception_to_list(
      exception_list* el, exception_ptr_class asyncExcp);

 private:
  using _exception_list = vector_class<exception_ptr_class>;

  _exception_list m_exceptionList;

 protected:
  /// @cond COMPUTECPP_DEV
  /** @brief Default constructor, not available to users
   */
  exception_list() = default;

  /** @brief Adds an exception to the list
   * @param asyncExcp exception to add.
   */
  void add_exception(exception_ptr_class asyncExcp);

  /// COMPUTECPP_DEV @endcond
 public:
  /** @brief Type of the list elements
   */
  using value_type = exception_ptr_class;
  /** @brief Reference type to a list element
   */
  using reference = value_type&;
  /** @brief Constant reference type to a list element
   */
  using const_reference = const value_type&;
  /** @brief Type of the size of the list
   */
  using size_type = std::size_t;
  /** @brief iterator definition
   */
  using iterator = _exception_list::iterator;
  /** @brief Constant iterator definition
   */
  using const_iterator = _exception_list::const_iterator;

  /** @brief Number of reported errors.
   * @return the number of errors
   */
  size_type size() const;
  /** @return The head of the error list
   */
  const_iterator begin() const;
  /** @return The sentinel value representing the end of the error list
   */
  const_iterator end() const;
};

/** @brief async_handler type definition. This is the type expected by a
 * \ref device to report asynchronous errors.
 */
using async_handler = cl::sycl::function_class<void(exception_list)>;

namespace detail {

enum class exception_types {
  runtime,
  kernel,
  accessor,
  nd_range,
  event,
  invalid_parameter,
  device,
  compile_program,
  link_program,
  invalid_object,
  memory_allocation,
  platform_error,
  profiling,
  feature_not_supported
};

template <exception_types type, typename Base>
class exception_implementation : public Base {
 public:
  using Base::Base;
};

}  // namespace detail

/** @brief Base SYCL runtime error group. Sub-classes of this error
 * represent a runtime specific error.
 */
class runtime_error
    : public detail::exception_implementation<detail::exception_types::runtime,
                                              exception> {
 public:
  using exception_implementation::exception_implementation;
};

/** @brief Represents an error that occurred before or while enqueuing a SYCL
 * kernel.
 */
class kernel_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error regarding \ref cl::sycl::accessor objects
 * defined.
 */

class accessor_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error related to a provided nd_range.
 */
class nd_range_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error related to a \ref cl::sycl::event.
 */
class event_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error related to SYCL kernel parameters.
 */
class invalid_parameter_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Base SYCL device error group. Sub-classes of this error
 * represent a device specific error.
 */
class device_error
    : public detail::exception_implementation<detail::exception_types::runtime,
                                              exception> {
 public:
  using exception_implementation::exception_implementation;
};

/** @brief Represents an error that happened during compilation.
 */
class compile_program_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an error that happened during linking.
 */
class link_program_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an error regarding any memory object being used by a
 * kernel.
 */
class invalid_object_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents a memory allocation error.
 */
class memory_allocation_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents a platform related error.
 */
class platform_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an issue related to profiling (can only be raised if
 * profiling is enabled).
 */
class profiling_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an exception when an optional feature
 * or extension is used in a kernel but its not
 * available on the device the SYCL kernel is
 * being enqueued on.
 */
class feature_not_supported : public device_error {
 public:
  using device_error::device_error;
};

}  // namespace sycl
}  // namespace cl
#endif  // RUNTIME_INCLUDE_SYCL_ERROR_H_
