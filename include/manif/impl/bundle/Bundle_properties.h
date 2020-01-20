#ifndef _MANIF_MANIF_BUNDLE_PROPERTIES_H_
#define _MANIF_MANIF_BUNDLE_PROPERTIES_H_

#include "manif/impl/bundle/Bundle_list.h"
#include "manif/impl/traits.h"

namespace manif {

// Forward declaration
template <typename... _Args>
struct Bundle;

template <typename... _Args>
struct BundleTangent;

template <typename _Derived>
struct BundleBase;

template <typename _Derived>
struct BundleTangentBase;

namespace internal {

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<BundleBase<_Derived>> {
  static constexpr int Dim = traits<_Derived>::Dim;  /// @brief Space dimension
  static constexpr int DoF = traits<_Derived>::DoF;  /// @brief Degrees of freedom
};

//! traits specialization
template <typename _Derived>
struct LieGroupProperties<BundleTangentBase<_Derived>> {
  static constexpr int Dim = traits<_Derived>::Dim;  /// @brief Space dimension
  static constexpr int DoF = traits<_Derived>::DoF;  /// @brief Degrees of freedom
};

}  // namespace internal
}  // namespace manif

#endif
