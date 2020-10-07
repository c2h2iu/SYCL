/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file device_selector.h

  @brief This file contains the API for the @ref cl::sycl::device_selector class
*/
#ifndef RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_H_
#define RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_H_

#include "SYCL/common.h"
#include "SYCL/device.h"
#include "SYCL/offline_compilation.h"
#include "SYCL/predefines.h"

#include <memory>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
namespace detail {
// Implementation class declaration.
class device_selector;
}  // namespace detail

/**
  @brief Abstract class that can be implemented to tell the runtime how to
  perform device selection.

  The function call operator is a pure virtual function that needs to be
  implemented within derived classes.
*/
class COMPUTECPP_EXPORT device_selector {
 public:
  /** @brief Constructs a device_selector.
   */
  device_selector();

  /** @brief Constructs a device_selector from another device_selector.
   */
  device_selector(const device_selector& rhs);

  /** @brief Empty destructor.
   */
  virtual ~device_selector();

  /** @brief Performs a platform and device selection and returns a pointer to
   * the resulting cl::sycl::device object.
   * @return a pointer to the cl::sycl::device object that is selected.
   */
  COMPUTECPP_TEST_VIRTUAL device select_device() const;

  /** @brief Performs the scoring of a single device, called once for every
   * device discovered. Needs to be overloaded.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  virtual int operator()(const device& device) const = 0;

 protected:
  /** @brief Evaluates devices and returns the most suitable one.
   * @return chosenDevice The device selected via evaluation.
   */
  device evaluate_devices() const;

  /* @brief Pointer to the implementation object. */
  unique_ptr_class<detail::device_selector> m_impl;
};

/** @brief Implementation of a device_selector that selects either a CPU or a
  GPU, and falls back to a host mode device if none can be found.
*/
class COMPUTECPP_EXPORT default_selector : public device_selector {
 protected:
  /**
   * @copydoc default_selector(string_class target);
   */
  explicit default_selector(const char* target);
  /**
   * @brief Constructs a default_selector
   * @param target String representing a device target
   */
  explicit default_selector(string_class target)
      : default_selector{target.c_str()} {}

 public:
  /** @brief Constructs a default_selector
   */
  default_selector() : default_selector("") {}

  /** @brief Empty destructor.
   */
  ~default_selector() override = default;

  /** @brief Overload that scores both CPUs and GPUs positive if they have SPIR
    support, GPUs are scored higher, scores host mode devices as positive but
    lower than a non-host device. This should never fail.
  * @param device The device that is to be scored.
  * @return an integer representing the allocated score for the device.
  */
  int operator()(const device& Device) const override;

 protected:
  /** This function sets explicitly the m_compilationInfo member and it's used
   * as a helper for unit testing
   */
  void set_offline_backend(detail::offline_backend m) { m_compilationInfo = m; }

  /** @brief Get the cached offline compilation query result
   */
  inline detail::offline_backend get_offline_backend() const noexcept {
    return m_compilationInfo;
  }

 private:
  /** @brief Caches the offline compilation result from the offline compilation
   * query
   */
  detail::offline_backend m_compilationInfo;
};

/** @brief Implementation of an opencl_selector that selects either a CPU or a
 * GPU.
 */
class COMPUTECPP_EXPORT opencl_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  opencl_selector() = default;

  /** @brief Empty destructor.
   */
  ~opencl_selector() override = default;

  /** @brief Overload that scores both CPUs and GPUs positive if they have SPIR
    support, GPUs are scored higher. Will fail if no CPU or GPU is found.
  * @param device The device that is to be scored.
  * @return An integer representing the allocated score for the device.
  */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects a CPU device.
 */
class COMPUTECPP_EXPORT cpu_selector : public device_selector {
 public:
  cpu_selector() = default;

  ~cpu_selector() override = default;

  /** @brief Overload that scores CPUs positive if they have SPIR support. Fails
   * if a CPU cannot be found.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects a GPU device.
 */
class COMPUTECPP_EXPORT gpu_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  gpu_selector() = default;

  /** @brief Empty destructor.
   */
  ~gpu_selector() override = default;

  /** @brief Overload that scores GPUs positive if they have SPIR support. Fails
   * if a GPU cannot be found.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects an accelerator
 * device
 */
class COMPUTECPP_EXPORT accelerator_selector : public device_selector {
 public:
  /** @brief Default constructor
   */
  accelerator_selector() = default;

  /** @brief Overload that scores accelerators positive
   *        if they have SPIR support.
   *        Fails if an accelerator cannot be found.
   * @param device The device that is to be scored
   * @return an integer representing the allocated score for the device
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects an Intel platform.
 */
class COMPUTECPP_EXPORT intel_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  intel_selector() = default;

  /** @brief Empty destructor.
   */
  ~intel_selector() override = default;

  /** @brief Overload that scores devices with an Intel platform positive if
  they have SPIR support.
  * @param device The device that is to be scored.
  * @return an integer representing the allocated score for the device.
  */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects an AMD platform.
 */
class COMPUTECPP_EXPORT amd_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  amd_selector() = default;

  /** @brief Empty destructor.
   */
  ~amd_selector() override = default;

  /** @brief Overload that scores devices with an AMD platform positive if they
   * have SPIR support.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a host_selector that selects the host device.
 * This selector will always return a valid host device
 */
class COMPUTECPP_EXPORT host_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  host_selector() = default;

  /** @brief Empty destructor.
   */
  ~host_selector() override = default;

  /** @brief Overload that scores host mode devices positively.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @cond COMPUTECPP_DEV */

/** @brief Implementation of a device_selector that selects an ARM platform.
 */
class COMPUTECPP_EXPORT arm_selector : public device_selector {
 public:
  /**
    @brief Default constructor.
  */
  arm_selector() = default;

  /** @brief Empty destructor.
   */
  ~arm_selector() override = default;

  /** @brief Overload that scores devices with an ARM platform positive if they
   * have SPIR support.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** COMPUTECPP_DEV @endcond */

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_H_
