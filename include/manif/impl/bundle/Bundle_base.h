#ifndef _MANIF_MANIF_BUNDLE_BASE_H_
#define _MANIF_MANIF_BUNDLE_BASE_H_

#include "manif/impl/bundle/Bundle_list.h"
#include "manif/impl/bundle/Bundle_properties.h"
#include "manif/impl/lie_group_base.h"
#include "manif/impl/utils.h"

namespace manif {

struct Range {
  unsigned int start;
  unsigned int size;
};

template <typename _Derived>
struct BundleBase : LieGroupBase<_Derived> {
 private:
  using Base = LieGroupBase<_Derived>;
  using Type = BundleBase<_Derived>;

  using ListType = typename internal::traits<_Derived>::ListType;

 public:
  MANIF_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_AUTO_API
  MANIF_INHERIT_GROUP_OPERATOR

  template <unsigned int _id>
  using ElementType = typename ElementInfo<_id, ListType>::Type;

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
  const ElementType<_id>& get() const {
    return list().template get<_id>();
  }

  template <unsigned int _id>
  ElementType<_id>& get() {
    return list().template get<_id>();
  }

  template <unsigned int _id>
  static constexpr Range DofRange() {
    return Range{ElementInfo<_id, ListType>::DoFIndex, ElementType<_id>::DoF};
  }

  template <unsigned int _id>
  static constexpr Range RepRange() {
    return Range{ElementInfo<_id, ListType>::RepIndex, ElementType<_id>::RepSize};
  }

 protected:
  ListType& list() { return static_cast<_Derived&>(*this).list(); }
  const ListType& list() const { return static_cast<const _Derived&>(*this).list(); }
};

template <typename _Bundle>
struct InverseFunctor {
  using OptJacobianRef = typename _Bundle::OptJacobianRef;

  // constructor
  InverseFunctor(_Bundle& b, OptJacobianRef& j) : bundle(b), jac(j) {}

  // templated operator handle different kind of manifold element
  template <unsigned int _id, typename _LieGroup>
  void operator()(const _LieGroup& m) {
    using ThisOptJacobianRef = typename _Bundle::template ElementType<_id>::OptJacobianRef;
    static constexpr Range range = _Bundle::template DofRange<_id>();

    ThisOptJacobianRef this_jac;
    if (jac) {
      this_jac.emplace(jac->template block<range.size, range.size>(range.start, range.start));
    }

    bundle.template get<_id>() = m.inverse(this_jac);
  }

 protected:
  _Bundle& bundle;
  OptJacobianRef& jac;
};

template <typename _Derived>
typename BundleBase<_Derived>::LieGroup BundleBase<_Derived>::inverse(OptJacobianRef j_minv_m) const {
  LieGroup inversed;
  (*this).list().traverse(InverseFunctor<LieGroup>(inversed, j_minv_m));
  return inversed;
}

template <typename _Derived>
typename BundleBase<_Derived>::Tangent BundleBase<_Derived>::log(OptJacobianRef J_t_m) const {
  // TODO
}

template <typename _OtherBundle, typename _ResultBundle>
struct ComposeFunctor {
  using OptJacobianRef = typename _ResultBundle::OptJacobianRef;

  // constructor
  ComposeFunctor(const _OtherBundle& other, _ResultBundle& res, OptJacobianRef& j_c_a, OptJacobianRef& j_c_b)
      : other_(other), res_(res), jac_c_a_(j_c_a), jac_c_b_(j_c_b) {}

  // templated operator handle different kind of manifold element
  template <unsigned int _id, typename _LieGroup>
  void operator()(const _LieGroup& m) {
    using ThisOptJacobianRef = typename _ResultBundle::template ElementType<_id>::OptJacobianRef;
    static constexpr Range range = _ResultBundle::template DofRange<_id>();

    ThisOptJacobianRef this_jac_c_a;
    if (jac_c_a_) {
      this_jac_c_a.emplace(jac_c_a_->template block<range.size, range.size>(range.start, range.start));
    }

    ThisOptJacobianRef this_jac_c_b;
    if (jac_c_b_) {
      this_jac_c_b.emplace(jac_c_b_->template block<range.size, range.size>(range.start, range.start));
    }

    res_.template get<_id>() = m.compose(other_.template get<_id>(), this_jac_c_a, this_jac_c_b);
  }

 protected:
  const _OtherBundle& other_;
  _ResultBundle& res_;
  OptJacobianRef& jac_c_a_;
  OptJacobianRef& jac_c_b_;
};

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
