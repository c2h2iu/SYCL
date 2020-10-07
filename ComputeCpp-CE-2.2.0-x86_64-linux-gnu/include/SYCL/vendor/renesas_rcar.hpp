/******************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/*!
  @file renesas_rcar.hpp

  @brief This file includes Codeplay extensions for the Renesas R-Car platform.
*/

#ifndef RUNTIME_INCLUDE_SYCL_VENDOR_RENESAS_RCAR_HPP_
#define RUNTIME_INCLUDE_SYCL_VENDOR_RENESAS_RCAR_HPP_

#ifdef __SYCL_DEVICE_ONLY__

#pragma clang diagnostic push
#pragma clang diagnostic ignored \
    "-Wreturn-type-c-linkage"  // UDT return type in function with C linkage

extern "C" {

[[computecpp::builtin]] void _Z18begin_dma_transferPU3AS9hcjjjjj(
    __attribute__((address_space(9))) cl::sycl::cl_uchar* dst,
    cl::sycl::cl_char src_plane, cl::sycl::cl_uint src_offset,
    cl::sycl::cl_uint width, cl::sycl::cl_uint height, cl::sycl::cl_uint stride,
    cl::sycl::cl_uint transfer_desc);

[[computecpp::builtin]] void _Z18begin_dma_transfercjPU3AS9hjjjj(
    cl::sycl::cl_char dst_plane, cl::sycl::cl_uint dst_offset,
    const __attribute__((address_space(9))) cl::sycl::cl_uchar* src,
    cl::sycl::cl_uint width, cl::sycl::cl_uint height, cl::sycl::cl_uint stride,
    cl::sycl::cl_uint transfer_desc);

[[computecpp::builtin]] __attribute__((convergent)) void _Z16end_dma_transferj(
    cl::sycl::cl_uint transfer_desc);

}  // extern "C"

namespace cl {
namespace sycl {
namespace detail {
namespace rcar {

using dma_data_type = __attribute__((address_space(9))) cl::sycl::cl_uchar*;

inline void begin_dma_transfer(
    __attribute__((address_space(9))) cl::sycl::cl_uchar* dst,
    cl::sycl::cl_char src_plane, cl::sycl::cl_uint src_offset,
    cl::sycl::cl_uint width, cl::sycl::cl_uint height, cl::sycl::cl_uint stride,
    cl::sycl::cl_uint transfer_desc) noexcept {
  return void(::_Z18begin_dma_transferPU3AS9hcjjjjj(
      dst, src_plane, src_offset, width, height, stride, transfer_desc));
}

inline void begin_dma_transfer(
    cl::sycl::cl_char dest_plane, cl::sycl::cl_uint dest_offset,
    const __attribute__((address_space(9))) cl::sycl::cl_uchar* src,
    cl::sycl::cl_uint width, cl::sycl::cl_uint height, cl::sycl::cl_uint stride,
    cl::sycl::cl_uint transfer_desc) noexcept {
  return void(::_Z18begin_dma_transfercjPU3AS9hjjjj(
      dest_plane, dest_offset, src, width, height, stride, transfer_desc));
}

inline void end_dma_transfer(cl::sycl::cl_uint transfer_desc) noexcept {
  return void(::_Z16end_dma_transferj(transfer_desc));
}

}  // namespace rcar
}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
namespace detail {
namespace rcar {

/** @brief Asserts that the dimensionality of the accessors are valid when
 * used in begin_dma_transfer
 * @tparam N Number of dimensions
 */
template <int N>
inline void assert_plane_range() {
  static_assert(
      (N == 1) || (N == 2),
      "rcar_begin_dma_transfer only supports 1 or 2 dimensional accessors");
}

}  // namespace rcar
}  // namespace detail

/** Chunk size for DMA transfers invoked by rcar_begin_dma_transfer.
 */
enum class transfer_size : cl_uint {
  chunks_of_32 = 0,  // 0b00
  chunks_of_64 = 1,  // 0b01
  chunks_of_128 = 3  // 0b11
};

/** Thread that is to be used in DMA transfers triggered by
 * rcar_begin_dma_transfer.
 */
enum class transfer_thread : cl_uint { use_default = 0, use_current = 0x10 };

/** Whether to force a sub-group barrier in the wait for a DMA transfer to
 * complete invoked by rcar_await_dma_transfer.
 */
enum class force_sub_group_barrier : cl_uint { on = 0, off = 1 };

/** @brief Performs an asynchronous copy from global memory plane to subgroup
 * local memory.
 * @param source The region of memory to copy data from.
 * @param destination The region of memory to write the data to.
 * @param copyBounds The shape of the region.
 * @param offset The offset into the planar region of memory.
 * @param stride The subgroup local memory stride.
 */
template <class dataT, int sourceDim, int destinationDim,
          access::mode sourceMode, access::placeholder isPlaceholderSrc>
void rcar_begin_dma_transfer(
    const accessor<dataT, sourceDim, sourceMode, access::target::global_buffer,
                   isPlaceholderSrc>& source,
    const accessor<dataT, destinationDim, access::mode::read_write,
                   access::target::subgroup_local>& destination,
    const range<2> copyBounds, size_t offset, size_t stride,
    transfer_size transferSize = transfer_size::chunks_of_128,
    transfer_thread transferThread = transfer_thread::use_default) {
  detail::rcar::assert_plane_range<sourceDim>();
  detail::rcar::assert_plane_range<destinationDim>();
  detail::assert_read_mode<sourceMode>();
#ifdef __SYCL_DEVICE_ONLY__
  const auto offsetInBytes = offset * sizeof(dataT);
  const auto strideInBytes = stride * sizeof(dataT);
  const auto widthInBytes = copyBounds[0] * sizeof(dataT);
  const auto height = copyBounds[1];
  auto destPointer = reinterpret_cast<::cl::sycl::detail::rcar::dma_data_type>(
      destination.get_pointer().get());
  auto srcPlaneId = source.get_device_plane_id();

  ::cl::sycl::detail::rcar::begin_dma_transfer(
      destPointer, srcPlaneId, offsetInBytes, widthInBytes, height,
      strideInBytes,
      static_cast<cl_uint>(transferSize) |
          static_cast<cl_uint>(transferThread));
#else
  detail::trigger_sycl_log(
      log_type::not_implemented, __FILE__, __LINE__, CL_SUCCESS,
      detail::cpp_error_code::NOT_SUPPORTED_ERROR, nullptr,
      "ComputeCpp has not yet implemented rcar_begin_dma_transfer for host.");
#endif  // __SYCL_DEVICE_ONLY__
}

/** @brief Performs an asynchronous copy from a global memory plane to
 * subgroup local memory.
 * @param source The region of memory to copy data from.
 * @param destination The region of memory to write the data to.
 * @param copyBounds The shape of the region.
 * @param offset The offset into the planar region of memory.
 * @param stride The subgroup local memory stride.
 */
template <class dataT, int dim, access::mode sourceMode,
          access::placeholder isPlaceholderSrc>
void rcar_begin_dma_transfer(
    const accessor<dataT, dim, sourceMode, access::target::global_buffer,
                   isPlaceholderSrc>& source,
    const multi_ptr<dataT, access::address_space::subgroup_local_space>
        destination,
    const range<2> copyBounds, size_t offset, size_t stride,
    transfer_size transferSize = transfer_size::chunks_of_128,
    transfer_thread transferThread = transfer_thread::use_default) {
  detail::rcar::assert_plane_range<dim>();
  detail::assert_read_mode<sourceMode>();
#ifdef __SYCL_DEVICE_ONLY__
  const auto offsetInBytes = offset * sizeof(dataT);
  const auto strideInBytes = stride * sizeof(dataT);
  const auto widthInBytes = copyBounds[0] * sizeof(dataT);
  const auto height = copyBounds[1];
  auto destPointer =
      reinterpret_cast<::cl::sycl::detail::rcar::dma_data_type>(destination);
  auto srcPlaneId = source.get_device_plane_id();

  ::cl::sycl::detail::rcar::begin_dma_transfer(
      destPointer, srcPlaneId, offsetInBytes, widthInBytes, height,
      strideInBytes,
      static_cast<cl_uint>(transferSize) |
          static_cast<cl_uint>(transferThread));
#else
  detail::trigger_sycl_log(
      log_type::not_implemented, __FILE__, __LINE__, CL_SUCCESS,
      detail::cpp_error_code::NOT_SUPPORTED_ERROR, nullptr,
      "ComputeCpp has not yet implemented rcar_begin_dma_transfer for host.");
#endif  // __SYCL_DEVICE_ONLY__
}

/** @brief Performs an asynchronous copy from subgroup local memory to a
 * global memory plane.
 * @param source The region of memory to copy data from.
 * @param destination The region of memory to write the data to.
 * @param copyBounds The shape of the region.
 * @param offset The offset into the planar region of memory.
 * @param stride The subgroup local memory stride.
 */
template <class dataT, int sourceDim, int destinationDim,
          access::mode destinationMode, access::placeholder isPlaceholderDst>
void rcar_begin_dma_transfer(
    const accessor<dataT, sourceDim, access::mode::read_write,
                   access::target::subgroup_local>& source,
    const accessor<dataT, destinationDim, destinationMode,
                   access::target::global_buffer, isPlaceholderDst>&
        destination,
    const range<2> copyBounds, size_t offset, size_t stride,
    transfer_size transferSize = transfer_size::chunks_of_128,
    transfer_thread transferThread = transfer_thread::use_default) {
  detail::rcar::assert_plane_range<sourceDim>();
  detail::rcar::assert_plane_range<destinationDim>();
  detail::assert_write_mode<destinationMode>();
#ifdef __SYCL_DEVICE_ONLY__
  const auto offsetInBytes = offset * sizeof(dataT);
  const auto strideInBytes = stride * sizeof(dataT);
  const auto widthInBytes = copyBounds[0] * sizeof(dataT);
  const auto height = copyBounds[1];
  const auto destPlaneId = destination.get_device_plane_id();
  auto srcPointer = reinterpret_cast<::cl::sycl::detail::rcar::dma_data_type>(
      source.get_pointer().get());

  ::cl::sycl::detail::rcar::begin_dma_transfer(
      destPlaneId, offsetInBytes, srcPointer, widthInBytes, height,
      strideInBytes,
      static_cast<cl_uint>(transferSize) |
          static_cast<cl_uint>(transferThread));
#else
  detail::trigger_sycl_log(
      log_type::not_implemented, __FILE__, __LINE__, CL_SUCCESS,
      detail::cpp_error_code::NOT_SUPPORTED_ERROR, nullptr,
      "ComputeCpp has not yet implemented rcar_begin_dma_transfer for host.");
#endif  // __SYCL_DEVICE_ONLY__
}

/** @brief Performs an asynchronous copy from subgroup local memory to a
 * global memory plane.
 * @param source The region of memory to copy data from.
 * @param destination The region of memory to write the data to.
 * @param copyBounds The shape of the region.
 * @param offset The offset into the planar region of memory.
 * @param stride The subgroup local memory stride.
 */
template <class dataT, int dim, access::mode destinationMode,
          access::placeholder isPlaceholderDst>
void rcar_begin_dma_transfer(
    const multi_ptr<dataT, access::address_space::subgroup_local_space> source,
    const accessor<dataT, dim, destinationMode, access::target::global_buffer,
                   isPlaceholderDst>& destination,
    const range<2> copyBounds, size_t offset, size_t stride,
    transfer_size transferSize = transfer_size::chunks_of_128,
    transfer_thread transferThread = transfer_thread::use_default) {
  detail::rcar::assert_plane_range<dim>();
  detail::assert_write_mode<destinationMode>();
#ifdef __SYCL_DEVICE_ONLY__
  const auto offsetInBytes = offset * sizeof(dataT);
  const auto strideInBytes = stride * sizeof(dataT);
  const auto widthInBytes = copyBounds[0] * sizeof(dataT);
  const auto height = copyBounds[1];
  auto destPlaneId = destination.get_device_plane_id();
  auto srcPointer =
      reinterpret_cast<::cl::sycl::detail::rcar::dma_data_type>(source);

  ::cl::sycl::detail::rcar::begin_dma_transfer(
      destPlaneId, offsetInBytes, srcPointer, widthInBytes, height,
      strideInBytes,
      static_cast<cl_uint>(transferSize) |
          static_cast<cl_uint>(transferThread));
#else
  detail::trigger_sycl_log(
      log_type::not_implemented, __FILE__, __LINE__, CL_SUCCESS,
      detail::cpp_error_code::NOT_SUPPORTED_ERROR, nullptr,
      "ComputeCpp has not yet implemented rcar_begin_dma_transfer for host.");
#endif  // __SYCL_DEVICE_ONLY__
}

/** @brief Waits until the asynchronous DMA operation triggered by
 * rcar_begin_dma_transfer completes.
 */
void rcar_await_dma_transfer(force_sub_group_barrier forceSubGroupBarrier =
                                 force_sub_group_barrier::on) {
#ifdef __SYCL_DEVICE_ONLY__
  ::cl::sycl::detail::rcar::end_dma_transfer(
      static_cast<cl_uint>(forceSubGroupBarrier));
#else
  // Nothing to do on host
#endif  // __SYCL_DEVICE_ONLY__
}

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_VENDOR_RENESAS_RCAR_HPP_
