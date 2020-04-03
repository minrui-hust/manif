#ifndef _MANIF_MANIF_ARRAYTANGENT_BASE_H_
#define _MANIF_MANIF_ARRAYTANGENT_BASE_H_

#include "manif/impl/array/Array_properties.h"
#include "manif/impl/tangent_base.h"

namespace manif {

/**
 * @brief The base class of the Bundle tangent.
 */
template <typename _Derived>
struct ArrayTangentBase : TangentBase<_Derived> {
 private:
  using Base = TangentBase<_Derived>;
  using ElementType = typename internal::traits<_Derived>::ElementType;
  using ArrayType = typename internal::traits<_Derived>::ArrayType;

  using ElementOptJacobianRef = typename ElementType::OptJacobianRef;

  static constexpr int ElementDoF = ElementType::DoF;

 public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_OPERATOR

  static constexpr int Size = internal::traits<_Derived>::Size;

  using Base::coeffs;

  ArrayTangentBase() = default;
  ~ArrayTangentBase() = default;

  /**
   * @brief Hat operator of Bundle.
   * @return An element of the Bundle Lie algebra.
   */
  LieAlg hat() const {
    LieAlg alg;
    for (auto i = 0; i < (*this).size(); ++i) {
      alg[i] = (*this).get(i).hat();
    }
    return alg;
  }

  /**
   * @brief Get the Bundle element.
   * @param[out] -optional- J_m_t Jacobian of the SO3 element wrt this.
   * @return The Bundle element.
   */
  LieGroup exp(OptJacobianRef J_m_t = {}) const {
    if (j_m_t) j_m_t->setZero();
    LieGroup array;
    for (auto i = 0; i < (*this).size(); ++i) {
      ElementOptJacobianRef element_jac;
      if (j_m_t) element_jac.emplace(j_m_t->template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF));
      array.get(i) = (*this).get(i).exp(element_jac);
    }
    return array;
  }

  /**
   * Get the right Jacobian of Bundle.
   */
  Jacobian rjac() const {
    Jacobian jac;
    jac.setZero();
    for (auto i = 0; i < (*this).size(); ++i) {
      jac.template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF) = (*this).get(i).rjac();
    }
    return jac;
  }

  /**
   * Get the left Jacobian of Bundle.
   */
  Jacobian ljac() const {
    Jacobian jac;
    jac.setZero();
    for (auto i = 0; i < (*this).size(); ++i) {
      jac.template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF) = (*this).get(i).ljac();
    }
    return jac;
  }

  /**
   * Get the inverse of the right Jacobian of Bundle.
   */
  Jacobian rjacinv() const {
    Jacobian jac;
    jac.setZero();
    for (auto i = 0; i < (*this).size(); ++i) {
      jac.template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF) = (*this).get(i).rjacinv();
    }
    return jac;
  }

  /**
   * Get the inverse of the left Jacobian of Bundle.
   */
  Jacobian ljacinv() const {
    Jacobian jac;
    jac.setZero();
    for (auto i = 0; i < (*this).size(); ++i) {
      jac.template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF) = (*this).get(i).ljacinv();
    }
    return jac;
  }

  /**
   * @brief
   * @return
   */
  Jacobian smallAdj() const {
    Jacobian jac;
    jac.setZero();
    for (auto i = 0; i < (*this).size(); ++i) {
      jac.template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF) = (*this).get(i).smallAdj();
    }
    return jac;
  }

  // BundleTangent specific API

  /**
   * @brief Get the size of the array
   * @tparam
   * @return The size of the array
   */
  int size() { return array.size(); }

  /**
   * @brief Get the element in array, mutable version
   * @param id The element to get
   * @return An reference to the element
   */
  ElementType& get(int id) { return array()[id]; }

  /**
   * @brief Get the element in array, const version
   * @param id The element to get
   * @return An const reference to the element
   */
  const ElementType& get(int id) const { return array()[id]; }

 protected:
  friend internal::RandomEvaluatorImpl<ArrayTangentBase>;

  // Get the underlying array storage, which is a member of derived struct
  ArrayType& array() { return static_cast<_Derived&>(*this).array(); }                    // mutable version
  const ArrayType& array() const { return static_cast<const _Derived&>(*this).array(); }  // const version
};

namespace internal {
//! @brief Random specialization for ArrayTangentBase objects.
template <typename _Derived>
struct RandomEvaluatorImpl<BundleTangentBase<_Derived>> {
  using Base = ArrayTangentBase<_Derived>;
  static void run(Base& t) {
    for (auto i = 0; i < t.size(); ++i) {
      t.get(i).setRandom();
    }
  }
};
} /* namespace internal */

}  // namespace manif

#endif
