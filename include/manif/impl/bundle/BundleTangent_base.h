#ifndef _MANIF_MANIF_BUNDLETANGENT_BASE_H_
#define _MANIF_MANIF_BUNDLETANGENT_BASE_H_

#include "manif/impl/bundle/Bundle_properties.h"
#include "manif/impl/tangent_base.h"

namespace manif {

/**
 * @brief The base class of the Bundle tangent.
 */
template <typename _Derived>
struct BundleTangentBase : TangentBase<_Derived> {
 private:
  using Base = TangentBase<_Derived>;
  using Type = BundleTangentBase<_Derived>;

  using ListType = typename internal::traits<_Derived>::ListType;

 public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_OPERATOR

  // Tangent common API

  using Base::coeffs;

  BundleTangentBase() = default;
  ~BundleTangentBase() = default;

  /**
   * @brief Hat operator of Bundle.
   * @return An element of the Bundle Lie algebra.
   */
  LieAlg hat() const;

  /**
   * @brief Get the Bundle element.
   * @param[out] -optional- J_m_t Jacobian of the SO3 element wrt this.
   * @return The Bundle element.
   */
  LieGroup exp(OptJacobianRef J_m_t = {}) const;

  /**
   * Get the right Jacobian of Bundle.
   */
  Jacobian rjac() const;

  /**
   * Get the left Jacobian of Bundle.
   */
  Jacobian ljac() const;

  /**
   * Get the inverse of the right Jacobian of Bundle.
   */
  Jacobian rjacinv() const;

  /**
   * Get the inverse of the left Jacobian of Bundle.
   */
  Jacobian ljacinv() const;

  /**
   * @brief
   * @return
   */
  Jacobian smallAdj() const;

  // BundleTangent specific API

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
  friend internal::RandomEvaluatorImpl<BundleTangentBase>;

  // Get the underlying list storage, which is a member of derived struct
  ListType& list() { return static_cast<_Derived&>(*this).list(); }                    // mutable version
  const ListType& list() const { return static_cast<const _Derived&>(*this).list(); }  // const version
};

template <typename _BundleTangent>
struct HatFunctor {
  using LieAlg = typename _BundleTangent::LieAlg;

  HatFunctor(LieAlg& alg) : alg_(alg) {}

  template <unsigned int _id, typename _Tangent>
  void operator()(const _Tangent& tangent) {
    alg_.template get<_id>() = tangent.hat();
  }

 protected:
  LieAlg& alg_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::LieAlg BundleTangentBase<_Derived>::hat() const {
  LieAlg alg;
  (*this).list().traverse(HatFunctor<_Derived>(alg));
  return alg;
}

template <typename _BundleTangent>
struct ExpFunctor {
  using LieGroup = typename _BundleTangent::LieGroup;
  using OptJacobianRef = typename _BundleTangent::OptJacobianRef;

  ExpFunctor(LieGroup& bundle, OptJacobianRef& jac) : bundle_(bundle), jac_(jac) {}

  template <unsigned int _id, typename _Tangent>
  void operator()(const _Tangent& tangent) {
    using ThisOptJacobianRef = typename _Tangent::OptJacobianRef;
    static constexpr Range range = _BundleTangent::template DofRange<_id>();

    ThisOptJacobianRef this_jac;
    if (jac_) {
      this_jac.emplace(jac_->template block<range.size, range.size>(range.start, range.start));
    }

    bundle_.template get<_id>() = tangent.exp(this_jac);
  }

 protected:
  LieGroup& bundle_;
  OptJacobianRef& jac_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::LieGroup BundleTangentBase<_Derived>::exp(OptJacobianRef j_m_t) const {
  if (j_m_t) j_m_t->setZero();

  LieGroup bundle;
  (*this).list().traverse(ExpFunctor<_Derived>(bundle, j_m_t));
  return bundle;
}

template <typename _BundleTangent>
struct RjacFunctor {
  using Jacobian = typename _BundleTangent::Jacobian;

  RjacFunctor(Jacobian& jac) : jac_(jac) {}

  template <unsigned int _id, typename _Tangent>
  void operator()(const _Tangent& tangent) {
    static constexpr Range range = _BundleTangent::template DofRange<_id>();

    jac_.template block<range.size, range.size>(range.start, range.start) = tangent.rjac();
  }

 protected:
  Jacobian& jac_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian BundleTangentBase<_Derived>::rjac() const {
  Jacobian jac;
  (*this).list().traverse(RjacFunctor<_Derived>(jac));
  return jac;
}

template <typename _BundleTangent>
struct LjacFunctor {
  using Jacobian = typename _BundleTangent::Jacobian;

  LjacFunctor(Jacobian& jac) : jac_(jac) {}

  template <unsigned int _id, typename _Tangent>
  void operator()(const _Tangent& tangent) {
    static constexpr Range range = _BundleTangent::template DofRange<_id>();

    jac_.template block<range.size, range.size>(range.start, range.start) = tangent.ljac();
  }

 protected:
  Jacobian& jac_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian BundleTangentBase<_Derived>::ljac() const {
  Jacobian jac;
  (*this).list().traverse(LjacFunctor<_Derived>(jac));
  return jac;
}

template <typename _BundleTangent>
struct RjacInvFunctor {
  using Jacobian = typename _BundleTangent::Jacobian;

  RjacInvFunctor(Jacobian& jac) : jac_(jac) {}

  template <unsigned int _id, typename _Tangent>
  void operator()(const _Tangent& tangent) {
    static constexpr Range range = _BundleTangent::template DofRange<_id>();

    jac_.template block<range.size, range.size>(range.start, range.start) = tangent.rjacinv();
  }

 protected:
  Jacobian& jac_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian BundleTangentBase<_Derived>::rjacinv() const {
  Jacobian jac;
  (*this).list().traverse(RjacInvFunctor<_Derived>(jac));
  return jac;
}

template <typename _BundleTangent>
struct LjacInvFunctor {
  using Jacobian = typename _BundleTangent::Jacobian;

  LjacInvFunctor(Jacobian& jac) : jac_(jac) {}

  template <unsigned int _id, typename _Tangent>
  void operator()(const _Tangent& tangent) {
    static constexpr Range range = _BundleTangent::template DofRange<_id>();

    jac_.template block<range.size, range.size>(range.start, range.start) = tangent.ljacinv();
  }

 protected:
  Jacobian& jac_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian BundleTangentBase<_Derived>::ljacinv() const {
  Jacobian jac;
  (*this).list().traverse(LjacInvFunctor<_Derived>(jac));
  return jac;
}

template <typename _BundleTangent>
struct SmallAdjFunctor {
  using Jacobian = typename _BundleTangent::Jacobian;

  SmallAdjFunctor(Jacobian& jac) : jac_(jac) {}

  template <unsigned int _id, typename _Tangent>
  void operator()(const _Tangent& tangent) {
    static constexpr Range range = _BundleTangent::template DofRange<_id>();

    jac_.template block<range.size, range.size>(range.start, range.start) = tangent.smallAdj();
  }

 protected:
  Jacobian& jac_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::Jacobian BundleTangentBase<_Derived>::smallAdj() const {
  Jacobian jac;
  (*this).list().traverse(SmallAdjFunctor<_Derived>(jac));
  return jac;
}

//
namespace internal {

//! @brief Random specialization for BundleTangentBase objects.

template <typename _Derived>
struct RandomEvaluatorImpl<BundleTangentBase<_Derived>> {
  using Base = BundleTangentBase<_Derived>;

  struct RandomFunctor {
    // Templated operator handle different kind of manifold element
    template <unsigned int _id, typename _Tangent>
    void operator()(_Tangent& t) {
      t.setRandom();
    }
  };

  static void run(Base& t) { t.list().traverse(RandomFunctor()); }
};

} /* namespace internal */

}  // namespace manif

#endif
