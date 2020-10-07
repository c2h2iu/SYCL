/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 * @file abacus_all.h
 * @brief Internal file used by the built-in maths functions.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ABACUS_ALL_H_
#define RUNTIME_INCLUDE_SYCL_ABACUS_ALL_H_

#include "abacus/abacus_config"

#include "abacus/abacus_type_traits"

#include "abacus/abacus_math"

#include "abacus_types.h"

#include "abacus/abacus_detail_cast"

#include "abacus/abacus_relational"

#include "abacus/abacus_detail_relational"

#include "abacus/abacus_integer"

#include "abacus/abacus_detail_integer"

#include "abacus/abacus_common"

#include "abacus/abacus_detail_common"

#include "abacus/abacus_geometric"

#include "abacus/abacus_detail_geometric"

namespace cl {
namespace sycl {
namespace detail {
template <typename T>
using abacus_type_t = typename abacus::convert_abacus_sycl<T>::abacus_type;
}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ABACUS_ALL_H_
