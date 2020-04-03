#ifndef _MANIF_MANIF_ARRAY_PROPERTIES_H_
#define _MANIF_MANIF_ARRAY_PROPERTIES_H_

#include "manif/impl/traits.h"

namespace manif {

// Forward declaration
template <typename _ElementType, int _N>
struct Array;

template <typename _ElementType, int _N>
struct ArrayTangent;

template <typename _Derived>
struct ArrayBase;

template <typename _Derived>
struct ArrayTangentBase;

namespace internal {

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<ArrayBase<_Derived>> {
  static constexpr int Dim = traits<_Derived>::Dim;  /// @brief Space dimension
  static constexpr int DoF = traits<_Derived>::DoF;  /// @brief Degrees of freedom
};

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<ArrayTangentBase<_Derived>> {
  static constexpr int Dim = traits<_Derived>::Dim;  /// @brief Space dimension
  static constexpr int DoF = traits<_Derived>::DoF;  /// @brief Degrees of freedom
};

}  // namespace internal
}  // namespace manif

#endif
