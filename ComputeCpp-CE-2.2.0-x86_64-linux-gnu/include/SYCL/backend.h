/*****************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

********************************************************************/

/**
  @file backend.h

  @brief This file contains the interface for enumerating backends
  and backend-specific functionality.
*/
#ifndef RUNTIME_INCLUDE_SYCL_BACKEND_H_
#define RUNTIME_INCLUDE_SYCL_BACKEND_H_

#define SYCL_BACKEND_OPENCL
#define SYCL_BACKEND_HOST

namespace cl {
namespace sycl {

/// Defines backends available in ComputeCpp
enum class backend {
  host,    ///< Native C++ device
  opencl,  ///< OpenCL device
};

}  // namespace sycl
}  // namespace cl

/** COMPUTECPP_DEV @endcond */

#endif  // RUNTIME_INCLUDE_SYCL_BACKEND_H_
