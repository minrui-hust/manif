#ifndef _MANIF_MANIF_BUNDLE_H_
#define _MANIF_MANIF_BUNDLE_H_

#include "manif/impl/bundle/Bundle_base.h"

namespace manif {
// Forward declare for type traits specialization
template <typename... _Args>
struct Bundle;

template <typename... _Agrs>
struct BundleTangent;

namespace internal {
//! Traits specialization
template <typename... _Args>
struct traits<Bundle<_Args...>> {
  using ListType = List<_Args...>;
  using Scalar = typename ListType::Scalar;

  using LieGroup = Bundle<_Args...>;
  using Tangent = BundleTangent<_Args...>;

  using Base = BundleBase<Bundle<_Args...>>;

  static constexpr int Dim = LieGroupProperties<Base>::Dim;
  static constexpr int DoF = LieGroupProperties<Base>::DoF;
  static constexpr int RepSize = LieGroupListInfo<ListType>::RepSize;

  using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  using Vector = Eigen::Matrix<Scalar, DoF, 1>;
};
} /* namespace internal */

//
// LieGroup
//

/**
 * @brief Represents an element of SO3.
 */
template <typename... _Args>
struct Bundle : BundleBase<Bundle<_Args...>> {
 private:
  using Base = BundleBase<Bundle<_Args...>>;
  using Type = Bundle<_Args...>;
  using ListType = List<_Args...>;

 public:
  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Bundle() = default;
  ~Bundle() = default;

  // Copy constructor
  Bundle(const Bundle& o);

  template <typename _DerivedOther>
  Bundle(const BundleBase<_DerivedOther>& o);

  template <typename _DerivedOther>
  Bundle(const LieGroupBase<_DerivedOther>& o);

  // Copy constructor given Eigen
  template <typename _EigenDerived>
  Bundle(const Eigen::MatrixBase<_EigenDerived>& data);

  // Copy constructor given a list of LieGroup elements
  Bundle(const _Args&... others);

  const DataType& coeffs() const;

 protected:
  friend struct LieGroupBase<Bundle<_Args...>>;
  DataType& coeffs_nonconst();
  ListType& list();

  DataType data_;
  ListType list_;
};

template <typename... _Args>
template <typename _EigenDerived>
Bundle<_Args...>::Bundle(const Eigen::MatrixBase<_EigenDerived>& data) : data_(data), list_(data_.data()) {
  //
}

template <typename... _Args>
Bundle<_Args...>::Bundle(const Bundle& o) : Bundle(o.coeffs()) {
  //
}

template <typename... _Args>
template <typename _DerivedOther>
Bundle<_Args...>::Bundle(const BundleBase<_DerivedOther>& o) : Bundle(o.coeffs()) {
  //
}

template <typename... _Args>
template <typename _DerivedOther>
Bundle<_Args...>::Bundle(const LieGroupBase<_DerivedOther>& o) : Bundle(o.coeffs()) {
  //
}

template <typename... _Args>
Bundle<_Args...>::Bundle(const _Args&... others) : list_(data_.data()) {
  list_.set(others...);
}

template <typename... _Args>
const typename Bundle<_Args...>::DataType& Bundle<_Args...>::coeffs() const {
  return data_;
}

template <typename... _Args>
typename Bundle<_Args...>::DataType& Bundle<_Args...>::coeffs_nonconst() {
  return data_;
}

template <typename... _Args>
typename Bundle<_Args...>::ListType& Bundle<_Args...>::list() {
  return list_;
}

}  // namespace manif

#endif
