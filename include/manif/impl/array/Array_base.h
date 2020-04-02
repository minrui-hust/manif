#ifndef _MANIF_MANIF_ARRAY_BASE_H_
#define _MANIF_MANIF_ARRAY_BASE_H_

#include "manif/impl/array/Bundle_list.h"
#include "manif/impl/bundle/Bundle_properties.h"
#include "manif/impl/lie_group_base.h"
#include "manif/impl/utils.h"

namespace manif {

template <typename _Derived>
struct ArrayBase : LieGroupBase<_Derived> {
 private:
  using Base = LieGroupBase<_Derived>;
  using ElementType = typename internal::traits<_Derived>::ElementType;
  using ArrayType = typename internal::traits<_Derived>::ArrayType;

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

  // Array Specific functions

  /**
   * @brief Get the size of the array
   * @tparam
   * @return The size of the array
   */
  size_t size() { return array.size(); }

  /**
   * @brief Get the element of the array
   * @tparam element index
   * @return The reference of the element
   */
  ElementType& get(size_t id) { return array()[id]; }

  /**
   * @brief Get the element of the array (const version)
   * @tparam Element index
   * @return The const reference of the element
   */
  const ElementType& get(size_t id) const { return array()[id]; }

 protected:
  ArrayType& array() { return static_cast<_Derived&>(*this).array(); }
  const ArrayType& array() const { return static_cast<const _Derived&>(*this).array(); }
};

// Implemention of inverse
template <typename _Derived>
typename ArrayBase<_Derived>::LieGroup ArrayBase<_Derived>::inverse(OptJacobianRef j_inv_m) const {
  using ElementOptJacobianRef = ElementType::OptjacobianRef;
  static constexpr size_t ElementDoF = ElementType::DoF;

  if (j_inv_m) j_inv_m->setZero();

  LieGroup inversed;
  for (auto i = 0u; i < (*this).size(); ++i) {
    ElementOptJacobianRef element_jac;
    if (j_inv_m) {
      element_jac.emplace(j_inv_m->template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF));
    }

    inversed.get(i) = (*this).get(i).inverse(element_jac);
  }

  return inversed;
}

// Implemention of log
template <typename _Derived>
typename ArrayBase<_Derived>::Tangent ArrayBase<_Derived>::log(OptJacobianRef j_t_m) const {
  using ElementOptJacobianRef = ElementType::OptjacobianRef;
  static constexpr size_t ElementDoF = ElementType::DoF;

  if (j_t_m) j_t_m->setZero();

  Tangent tangent;
  for (auto i = 0u; i < (*this).size(); ++i) {
    ElementOptJacobianRef element_jac;
    if (j_inv_m) {
      element_jac.emplace(j_inv_m->template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF));
    }

    tangent.get(i) = (*this).get(i).log(element_jac);
  }

  return tangent;
}

// Implemention of compose
template <typename _Derived>
template <typename _DerivedOther>
typename ArrayBase<_Derived>::LieGroup ArrayBase<_Derived>::compose(const LieGroupBase<_DerivedOther>& m,
                                                                    OptJacobianRef j_c_a, OptJacobianRef j_c_b) const {
  using ElementOptJacobianRef = ElementType::OptjacobianRef;
  static constexpr size_t ElementDoF = ElementType::DoF;

  if (j_c_a) j_c_a->setZero();
  if (j_c_b) j_c_b->setZero();

  LieGroup composed;
  for (auto i = 0u; i < (*this).size(); ++i) {
    ElementOptJacobianRef element_jac_a;
    ElementOptJacobianRef element_jac_b;
    if (j_c_a) {
      element_jac_a.emplace(j_c_a->template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF));
    }
    if (j_c_b) {
      element_jac_b.emplace(j_c_b->template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF));
    }

    composed.get(i) = (*this).get(i).compose(m.get(i), element_jac_a, element_jac_b);
  }

  return composed;
}

// Implemention of act
template <typename _Derived>
template <typename _EigenDerived>
typename ArrayBase<_Derived>::Vector ArrayBase<_Derived>::act(
    const Eigen::MatrixBase<_EigenDerived>& v, tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> j_o_m,
    tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> j_o_v) const {
  using ElementOptJacobianLieGroupRef =
      tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, ElementType::Dim, ElementType::DoF>>>;
  using ElementOptJacobianVectorRef =
      tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, ElementType::Dim, ElementType::Dim>>>;
  static constexpr size_t ElementDoF = ElementType::DoF;
  static constexpr size_t ElementDim = ElementType::Dim;

  if (j_o_m) j_o_m->setZero();
  if (j_o_v) j_o_v->setZero();

  Vector res;
  for (auto i = 0u; i < (*this).size(); ++i) {
    ElementOptJacobianLieGroupRef element_j_o_m;
    ElementOptJacobianVectorRef element_j_o_v;
    if (j_o_m) {
      element_j_o_m.emplace(j_o_m->template block<ElementDim, ElementDoF>(i * ElementDim, i * ElementDoF));
    }
    if (j_o_v) {
      element_j_o_v.emplace(j_o_v->template block<ElementDim, ElementDim>(i * ElementDim, i * ElementDim));
    }

    res.template middleRows<ElementDim>(i * ElementDim) =
        (*this).get(i).act(v.template middleRows<ElementDim>(i * ElementDim), element_j_o_m, element_j_o_v);
  }

  return res;
}

// Implemention of adj
template <typename _Derived>
typename ArrayBase<_Derived>::Jacobian ArrayBase<_Derived>::adj() const {
  static constexpr size_t ElementDoF = ElementType::DoF;

  Jacobian jac;
  jac.setZero();

  for (auto i = 0u; i < (*this).size(); ++i) {
    jac.template block<ElementDoF, ElementDoF>(i * ElementDoF, i * ElementDoF) = (*this).get(i).adj();
  }

  return jac;
}

}  // namespace manif

#endif
