/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp.

*******************************************************************************/

/**
  @file include_opencl.h

  @brief This file provides a common point to include OpenCL header files
  on different platforms.
*/
#ifndef RUNTIME_INCLUDE_SYCL_INCLUDE_OPENCL_H_
#define RUNTIME_INCLUDE_SYCL_INCLUDE_OPENCL_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#elif (defined(WINDOWS) || defined(_WIN32) || defined(_WIN64)) && \
    defined(__SYCL_DEVICE_ONLY__)
// Used to suppress "calling convention __stdcall ignored for this target"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#pragma clang diagnostic pop
#else
// Used to suppress warning when using OpenCL 1.2 functions marked as
// deprecated (usually on Windows with OpenCL 2.0)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

/** @brief Declare functions for extensions that are used automatically
 * by ComputeCpp when available
 */

/** @brief Function pointer to clCreateProgramWithIL or clCreateProgramWithILKHR
 * Whenever the extension is available, it is used to create the SPIRV program.
 *
 * See OpenCL Extensions 9.21 : Intermediate Language Programs for details.
 */
using clCreateProgramWithIL_fn = cl_program(CL_API_CALL*)(
    cl_context /*cl_context*/, const void* /*binary*/,
    size_t /*length of binary*/, cl_int* /*error code*/);

#endif  // RUNTIME_INCLUDE_SYCL_INCLUDE_OPENCL_H_
