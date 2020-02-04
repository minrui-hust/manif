#ifndef _MANIF_MANIF_BUNDLETANGENT_MAP_H_
#define _MANIF_MANIF_BUNDLETANGENT_MAP_H_

#include "manif/impl/bundle/BundleTangent.h"

namespace manif {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename... _Args>
struct traits<Eigen::Map<BundleTangent<_Args...>, 0>> : public traits<BundleTangent<_Args...>> {
  using typename traits<BundleTangent<_Args...>>::Scalar;
  using traits<BundleTangent<_Args...>>::DoF;
  using Base = BundleTangentBase<Eigen::Map<BundleTangent<_Args...>, 0>>;
  using DataType = ::Eigen::Map<Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template <typename... _Args>
struct traits<Eigen::Map<const BundleTangent<_Args...>, 0>> : public traits<const BundleTangent<_Args...>> {
  using typename traits<const BundleTangent<_Args...>>::Scalar;
  using traits<const BundleTangent<_Args...>>::DoF;
  using Base = BundleTangentBase<Eigen::Map<const BundleTangent<_Args...>, 0>>;
  using DataType = ::Eigen::Map<const Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

} /* namespace internal */
} /* namespace manif */

namespace Eigen {

/**
 * @brief Specialization of Map for manif::SO3Tangent
 */
template <typename... _Args>
class Map<manif::BundleTangent<_Args...>, 0> : public manif::BundleTangentBase<Map<manif::BundleTangent<_Args...>, 0>> {
  using Base = manif::BundleTangentBase<Map<manif::BundleTangent<_Args...>, 0>>;
  using ListType = manif::TangentList<_Args...>;

 public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  Map(Scalar* coeffs) : data_(coeffs), list_(data_.data()) {}

  DataType& coeffs() { return data_; }
  const DataType& coeffs() const { return data_; }

 protected:
  friend Base;
  const ListType& list() const { return list_; }
  ListType& list() { return list_; }

  DataType data_;
  ListType list_;
};

/**
 * @brief Specialization of Map for const manif::SO3Tangent
 */
template <typename... _Args>
class Map<const manif::BundleTangent<_Args...>, 0>
    : public manif::BundleTangentBase<Map<const manif::BundleTangent<_Args...>, 0>> {
  using Base = manif::BundleTangentBase<Map<const manif::BundleTangent<_Args...>, 0>>;
  using ListType = manif::TangentList<_Args...>;

 public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  Map(const Scalar* coeffs) : data_(coeffs), list_(data_.data()) {}

  const DataType& coeffs() const { return data_; }

 protected:
  friend Base;
  const ListType& list() const { return list_; }

  const DataType data_;
  const ListType list_;
};

} /* namespace Eigen */

#endif
