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
  LieAlg alg_;
};

template <typename _Derived>
typename BundleTangentBase<_Derived>::LieAlg BundleTangentBase<_Derived>::hat() const {
  LieAlg alg;
  (*this).list().traverse(HatFunctor<_Derived>(alg));
  return alg;
}

}  // namespace manif

#endif
