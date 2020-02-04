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
  using ListType = typename internal::traits<_Derived>::ListType;

 public:
  MANIF_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_AUTO_API
  MANIF_INHERIT_GROUP_OPERATOR

  // LieGroup common API

  /**
   * @brief
   * @param J_minv_m
   * @return
   */
  LieGroup inverse(OptJacobianRef J_minv_m = {}) const;

  /**
   * @brief
   * @param J_t_m
   * @return
   */
  Tangent log(OptJacobianRef J_t_m = {}) const;

  /**
   * @brief
   * @param m
   * @param J_mc_ma
   * @return
   */
  template <typename _DerivedOther>
  LieGroup compose(const LieGroupBase<_DerivedOther>& m, OptJacobianRef J_mc_ma = {},
                   OptJacobianRef J_mc_mb = {}) const;

  /**
   * @brief
   * @param v
   * @param J_vout_m
   * @return
   */
  template <typename _EigenDerived>
  Vector act(const Eigen::MatrixBase<_EigenDerived>& v,
             tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> J_vout_m = {},
             tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> J_vout_v = {}) const;

  /**
   * @brief
   * @return
   */
  Jacobian adj() const;

  // Bundle specific functions

  /**
   * @brief Get the type of the component in Bundle
   * @tparam _id The component to get
   */
  template <unsigned int _id>
  using ElementType = typename ElementInfo<_id, ListType>::Type;

  /**
   * @brief Get the component in Bundle, mutable version
   * @tparam _id The component to get
   * @return An reference to the component
   */
  template <unsigned int _id>
  ElementType<_id>& get() {
    return list().template get<_id>();
  }

  /**
   * @brief Get the component in Bundle, const version
   * @tparam _id The component to get
   * @return An const reference to the component
   */
  template <unsigned int _id>
  const ElementType<_id>& get() const {
    return list().template get<_id>();
  }

  /**
   * @brief Get the DoF range of a component in Bundle
   * @tparam _id The component to get
   * @return The range
   */
  template <unsigned int _id>
  static constexpr Range DofRange() {
    return Range{ElementInfo<_id, ListType>::DoFIndex, ElementType<_id>::DoF};
  }

  /**
   * @brief Get the component's representation range in Bundle
   * @tparam _id The component to get
   * @return The range
   */
  template <unsigned int _id>
  static constexpr Range RepRange() {
    return Range{ElementInfo<_id, ListType>::RepIndex, ElementType<_id>::RepSize};
  }

  /**
   * @brief Get the component's Dim range in Bundle
   * @tparam _id The component to get
   * @return The range
   */
  template <unsigned int _id>
  static constexpr Range DimRange() {
    return Range{ElementInfo<_id, ListType>::DimIndex, ElementType<_id>::Dim};
  }

 protected:
  friend internal::RandomEvaluatorImpl<BundleBase>;

  // Get the underlying list storage, which is a member of derived struct
  ListType& list() { return static_cast<_Derived&>(*this).list(); }                    // mutable version
  const ListType& list() const { return static_cast<const _Derived&>(*this).list(); }  // const version
};

template <typename _Bundle>
struct InverseFunctor {
  using OptJacobianRef = typename _Bundle::OptJacobianRef;

  // constructor
  InverseFunctor(_Bundle& res, OptJacobianRef& jac) : res_(res), jac_(jac) {}

  // templated operator handle different kind of manifold element
  template <unsigned int _id, typename _LieGroup>
  void operator()(const _LieGroup& m) {
    using ThisOptJacobianRef = typename _Bundle::template ElementType<_id>::OptJacobianRef;
    static constexpr Range range = _Bundle::template DofRange<_id>();

    ThisOptJacobianRef this_jac;
    if (jac_) {
      this_jac.emplace(jac_->template block<range.size, range.size>(range.start, range.start));
    }

    res_.template get<_id>() = m.inverse(this_jac);
  }

 protected:
  _Bundle& res_;
  OptJacobianRef& jac_;
};

template <typename _Derived>
typename BundleBase<_Derived>::LieGroup BundleBase<_Derived>::inverse(OptJacobianRef j_inv_m) const {
  if (j_inv_m) j_inv_m->setZero();
  LieGroup inversed;
  (*this).list().traverse(InverseFunctor<LieGroup>(inversed, j_inv_m));
  return inversed;
}

template <typename _Bundle>
struct LogFunctor {
  using Tangent = typename _Bundle::Tangent;
  using OptJacobianRef = typename _Bundle::OptJacobianRef;

  // constructor
  LogFunctor(Tangent& res, OptJacobianRef& jac) : res_(res), jac_(jac) {}

  // templated operator handle different kind of manifold element
  template <unsigned int _id, typename _LieGroup>
  void operator()(const _LieGroup& m) {
    using ThisOptJacobianRef = typename _Bundle::template ElementType<_id>::OptJacobianRef;
    static constexpr Range range = _Bundle::template DofRange<_id>();

    ThisOptJacobianRef this_jac;
    if (jac_) {
      this_jac.emplace(jac_->template block<range.size, range.size>(range.start, range.start));
    }

    res_.template get<_id>() = m.log(this_jac);
  }

 protected:
  Tangent& res_;
  OptJacobianRef& jac_;
};

template <typename _Derived>
typename BundleBase<_Derived>::Tangent BundleBase<_Derived>::log(OptJacobianRef j_t_m) const {
  if (j_t_m) j_t_m->setZero();
  Tangent tangent;
  (*this).list().traverse(LogFunctor<LieGroup>(tangent, j_t_m));
  return tangent;
}

template <typename _OtherBundle, typename _ResultBundle>
struct ComposeFunctor {
  using OptJacobianRef = typename _ResultBundle::OptJacobianRef;

  ComposeFunctor(const _OtherBundle& other, _ResultBundle& res, OptJacobianRef& j_c_a, OptJacobianRef& j_c_b)
      : other_(other), res_(res), jac_c_a_(j_c_a), jac_c_b_(j_c_b) {}

  // Templated operator handle different kind of manifold element
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
                                                                      OptJacobianRef j_c_a,
                                                                      OptJacobianRef j_c_b) const {
  if (j_c_a) j_c_a->setZero();
  if (j_c_b) j_c_b->setZero();

  LieGroup composed;
  (*this).list().traverse(
      ComposeFunctor<_DerivedOther, LieGroup>(static_cast<const _DerivedOther&>(m), composed, j_c_a, j_c_b));

  return composed;
}

template <typename _Bundle, typename _EigenVector>
struct ActFunctor {
  using Scalar = typename _Bundle::Scalar;
  using Vector = typename _Bundle::Vector;
  using OptJacobianBundleRef = tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _Bundle::Dim, _Bundle::DoF>>>;
  using OptJacobianVectorRef = tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _Bundle::Dim, _Bundle::Dim>>>;

  ActFunctor(const _EigenVector& v, Vector& res, OptJacobianBundleRef& j_o_m, OptJacobianVectorRef& j_o_v)
      : v_(v), res_(res), j_o_m_(j_o_m), j_o_v_(j_o_v) {}

  // Templated operator handle different kind of manifold element
  template <unsigned int _id, typename _LieGroup>
  void operator()(const _LieGroup& m) {
    using ThisOptJacobianBundleRef = tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _LieGroup::Dim, _LieGroup::DoF>>>;
    using ThisOptJacobianVectorRef = tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _LieGroup::Dim, _LieGroup::Dim>>>;
    static constexpr Range dim_range = _Bundle::template DimRange<_id>();
    static constexpr Range dof_range = _Bundle::template DofRange<_id>();

    ThisOptJacobianBundleRef this_j_o_m;
    if (j_o_m_) {
      this_j_o_m.emplace(j_o_m_->template block<dim_range.size, dof_range.size>(dim_range.start, dof_range.start));
    }

    ThisOptJacobianVectorRef this_j_o_v;
    if (j_o_v_) {
      this_j_o_v.emplace(j_o_v_->template block<dim_range.size, dim_range.size>(dim_range.start, dim_range.start));
    }

    res_.template middleRows<dim_range.size>(dim_range.start) =
        m.act(v_.template middleRows<dim_range.size>(dim_range.start), this_j_o_m, this_j_o_v);
  }

 protected:
  const _EigenVector& v_;
  Vector& res_;
  OptJacobianBundleRef& j_o_m_;
  OptJacobianVectorRef& j_o_v_;
};

template <typename _Derived>
template <typename _EigenDerived>
typename BundleBase<_Derived>::Vector BundleBase<_Derived>::act(
    const Eigen::MatrixBase<_EigenDerived>& v, tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> j_o_m,
    tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> j_o_v) const {
  if (j_o_m) j_o_m->setZero();
  if (j_o_v) j_o_v->setZero();

  Vector res;
  (*this).list().traverse(ActFunctor<_Derived, _EigenDerived>(static_cast<const _EigenDerived&>(v), res, j_o_m, j_o_v));
  return res;
}

template <typename _Bundle>
struct AdjFunctor {
  using Jacobian = typename _Bundle::Jacobian;

  AdjFunctor(Jacobian& jac) : jac_(jac) {}

  // Templated operator handle different kind of manifold element
  template <unsigned int _id, typename _LieGroup>
  void operator()(const _LieGroup& m) {
    static constexpr Range range = _Bundle::template DofRange<_id>();
    jac_.template block<range.size, range.size>(range.start, range.start) = m.adj();
  }

 protected:
  Jacobian& jac_;
};

template <typename _Derived>
typename BundleBase<_Derived>::Jacobian BundleBase<_Derived>::adj() const {
  Jacobian jac;
  jac.setZero();
  (*this).list().traverse(AdjFunctor<_Derived>(jac));
  return jac;
}

//
namespace internal {

//! @brief Random specialization for BundleBase objects.

template <typename _Derived>
struct RandomEvaluatorImpl<BundleBase<_Derived>> {
  using Base = BundleBase<_Derived>;

  struct RandomFunctor {
    // Templated operator handle different kind of manifold element
    template <unsigned int _id, typename _LieGroup>
    void operator()(_LieGroup& m) {
      m.setRandom();
    }
  };

  static void run(Base& m) { m.list().traverse(RandomFunctor()); }
};

} /* namespace internal */

}  // namespace manif

#endif
