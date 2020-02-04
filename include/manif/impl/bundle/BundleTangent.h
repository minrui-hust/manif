#ifndef _MANIF_MANIF_BUNDLETANGENT_H_
#define _MANIF_MANIF_BUNDLETANGENT_H_

#include <Eigen/Core>
#include <iostream>
#include "manif/impl/bundle/BundleTangent_base.h"

namespace manif {

namespace internal {
//! Traits specialization
template <typename... _Args>
struct traits<BundleTangent<_Args...>> {
  using ListType = TangentList<_Args...>;
  using Scalar = typename ListInfo<ListType>::Scalar;

  using LieGroup = Bundle<_Args...>;
  using Tangent = BundleTangent<_Args...>;

  using Base = BundleTangentBase<Tangent>;

  static constexpr int Dim = ListInfo<ListType>::Dim;
  static constexpr int DoF = ListInfo<ListType>::DoF;
  static constexpr int RepSize = DoF;

  using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  using LieAlg = LieAlgList<_Args...>;
};
}  // namespace internal

/**
 * @brief Represents an element of tangent space of Bundle.
 */
template <typename... _Args>
struct BundleTangent : BundleTangentBase<BundleTangent<_Args...>> {
 private:
  using Base = BundleTangentBase<BundleTangent<_Args...>>;
  using ListType = TangentList<_Args...>;

 public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND

  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  BundleTangent();

  // Copy constructor
  BundleTangent(const BundleTangent& o);

  template <typename _DerivedOther>
  BundleTangent(const BundleTangentBase<_DerivedOther>& o);

  template <typename _DerivedOther>
  BundleTangent(const TangentBase<_DerivedOther>& o);

  // Copy constructor given Eigen
  template <typename _EigenDerived>
  BundleTangent(const Eigen::MatrixBase<_EigenDerived>& v);

  // Copy constructor given a list of LieGroup elements
  template <typename... _Others>
  BundleTangent(const _Others&... others);

  // Tangent common API
  DataType& coeffs();
  const DataType& coeffs() const;

 protected:
  friend Base;
  const ListType& list() const;
  ListType& list();

  DataType data_;
  ListType list_;
};

template <typename... _Args>
BundleTangent<_Args...>::BundleTangent() : list_(data_.data()) {
  //
}

template <typename... _Args>
BundleTangent<_Args...>::BundleTangent(const BundleTangent& o) : BundleTangent(o.coeffs()) {
  //
}

template <typename... _Args>
template <typename _DerivedOther>
BundleTangent<_Args...>::BundleTangent(const BundleTangentBase<_DerivedOther>& o) : BundleTangent(o.coeffs()) {
  //
}

template <typename... _Args>
template <typename _DerivedOther>
BundleTangent<_Args...>::BundleTangent(const TangentBase<_DerivedOther>& o) : BundleTangent(o.coeffs()) {
  //
}

template <typename... _Args>
template <typename _EigenDerived>
BundleTangent<_Args...>::BundleTangent(const Eigen::MatrixBase<_EigenDerived>& v) : data_(v), list_(data_.data()) {
  //
}

template <typename... _Args>
typename BundleTangent<_Args...>::DataType& BundleTangent<_Args...>::coeffs() {
  return data_;
}

template <typename... _Args>
const typename BundleTangent<_Args...>::DataType& BundleTangent<_Args...>::coeffs() const {
  return data_;
}

template <typename... _Args>
typename BundleTangent<_Args...>::ListType& BundleTangent<_Args...>::list() {
  return list_;
}

template <typename... _Args>
const typename BundleTangent<_Args...>::ListType& BundleTangent<_Args...>::list() const {
  return list_;
}

}  // namespace manif

#endif
