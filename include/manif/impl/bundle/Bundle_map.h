#ifndef _MANIF_MANIF_BUNDLE_MAP_H_
#define _MANIF_MANIF_BUNDLE_MAP_H_

#include "manif/impl/bundle/Bundle.h"

namespace manif {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename... _Args>
struct traits<Eigen::Map<Bundle<_Args...>, 0>> : public traits<Bundle<_Args...>> {
  using typename traits<Bundle<_Args...>>::Scalar;
  using traits<Bundle<_Args...>>::RepSize;
  using Base = BundleBase<Eigen::Map<Bundle<_Args...>, 0>>;
  using DataType = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template <typename... _Args>
struct traits<Eigen::Map<const Bundle<_Args...>, 0>> : public traits<const Bundle<_Args...>> {
  using typename traits<const Bundle<_Args...>>::Scalar;
  using traits<const Bundle<_Args...>>::RepSize;
  using Base = BundleBase<Eigen::Map<const Bundle<_Args...>, 0>>;
  using DataType = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

} /* namespace internal */
} /* namespace manif */

namespace Eigen {

/**
 * @brief Specialization of Map for manif::Bundle
 */
template <typename... _Args>
class Map<manif::Bundle<_Args...>, 0> : public manif::BundleBase<Map<manif::Bundle<_Args...>, 0>> {
 private:
  using Base = manif::BundleBase<Map<manif::Bundle<_Args...>, 0>>;
  using ListType = manif::LieGroupList<_Args...>;

 public:
  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Map(Scalar* coeffs) : data_(coeffs), list_(data_.data()) {}

  const DataType& coeffs() const { return data_; }

 protected:
  friend struct manif::LieGroupBase<Map<manif::Bundle<_Args...>, 0>>;
  DataType& coeffs_nonconst() { return data_; }

  friend Base;
  const ListType& list() const { return list_; }
  ListType& list() { return list_; }

  DataType data_;
  ListType list_;
};

/**
 * @brief Specialization of Map for const manif::Bundle
 */
template <typename... _Args>
class Map<const manif::Bundle<_Args...>, 0> : public manif::BundleBase<Map<const manif::Bundle<_Args...>, 0>> {
 private:
  using Base = manif::BundleBase<Map<const manif::Bundle<_Args...>, 0>>;
  using ListType = manif::LieGroupList<_Args...>;

 public:
  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

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
