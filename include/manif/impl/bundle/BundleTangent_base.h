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
};

}  // namespace manif

#endif
