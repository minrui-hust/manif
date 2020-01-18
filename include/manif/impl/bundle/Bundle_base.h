#ifndef _MANIF_MANIF_BUNDLE_BASE_H_
#define _MANIF_MANIF_BUNDLE_BASE_H_

#include "manif/impl/bundle/Bundle_list.h"
#include "manif/impl/bundle/Bundle_properties.h"
#include "manif/impl/lie_group_base.h"
#include "manif/impl/utils.h"

namespace manif {
template <typename _Derived>
struct BundleBase : LieGroupBase<_Derived> {
 private:
  using Base = LieGroupBase<_Derived>;
  using Type = BundleBase<_Derived>;

  using ListType = typename internal::traits<_Derived>::ListType;

  template <unsigned int _id>
  using ListElementType = typename ListType::template ListElementType<_id>;

 public:
  MANIF_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_AUTO_API
  MANIF_INHERIT_GROUP_OPERATOR

  // LieGroup common API
  LieGroup inverse(OptJacobianRef J_minv_m = {}) const;

  Tangent log(OptJacobianRef J_t_m = {}) const;

  template <typename _DerivedOther>
  LieGroup compose(const LieGroupBase<_DerivedOther>& m, OptJacobianRef J_mc_ma = {},
                   OptJacobianRef J_mc_mb = {}) const;

  template <typename _EigenDerived>
  Eigen::Matrix<Scalar, 3, 1> act(const Eigen::MatrixBase<_EigenDerived>& v,
                                  tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_m = {},
                                  tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v = {}) const;

  Jacobian adj() const;

  // Bundle specific API

  template <unsigned int _id>
  const ListElementType<_id>& get() const;

  template <unsigned int _id>
  ListElementType<_id>& get();

  struct Range {
    unsigned int start;
    unsigned int size;
  };

  template <unsigned int _id>
  Range dof_range() const;

  template <unsigned int _id>
  Range rep_range() const;

 protected:
  ListType& list() { return static_cast<_Derived&>(*this).list(); }
};

template <typename _Derived>
typename BundleBase<_Derived>::LieGroup BundleBase<_Derived>::inverse(OptJacobianRef J_minv_m) const {
  LieGroup inversed;
  LieGroupListOperation<ListType>::BundleInverse((*this).list(), inversed.list(), J_minv_m);
  return inversed;
}

template <typename _Derived>
typename BundleBase<_Derived>::Tangent BundleBase<_Derived>::log(OptJacobianRef J_t_m) const {}

template <typename _Derived>
template <typename _DerivedOther>
typename BundleBase<_Derived>::LieGroup BundleBase<_Derived>::compose(const LieGroupBase<_DerivedOther>& m,
                                                                      OptJacobianRef J_mc_ma,
                                                                      OptJacobianRef J_mc_mb) const {}

template <typename _Derived>
template <typename _EigenDerived>
Eigen::Matrix<typename BundleBase<_Derived>::Scalar, 3, 1> BundleBase<_Derived>::act(
    const Eigen::MatrixBase<_EigenDerived>& v, tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_m,
    tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>> J_vout_v) const {}

template <typename _Derived>
typename BundleBase<_Derived>::Jacobian BundleBase<_Derived>::adj() const {}

}  // namespace manif

#endif
