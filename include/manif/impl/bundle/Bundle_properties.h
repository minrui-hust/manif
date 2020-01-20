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
template <typename... _Args>
struct LieGroupProperties<BundleBase<Bundle<_Args...>>> {
  static constexpr int Dim = ListInfo<List<_Args...>>::Dim;  /// @brief Space dimension
  static constexpr int DoF = ListInfo<List<_Args...>>::DoF;  /// @brief Degrees of freedom
};

template <typename... _Args>
struct LieGroupProperties<BundleBase<Eigen::Map<Bundle<_Args...>>>> {
  static constexpr int Dim = ListInfo<List<_Args...>>::Dim;  /// @brief Space dimension
  static constexpr int DoF = ListInfo<List<_Args...>>::DoF;  /// @brief Degrees of freedom
};

//! traits specialization
template <typename... _Args>
struct LieGroupProperties<BundleTangentBase<BundleTangent<_Args...>>> {
  static constexpr int Dim = ListInfo<List<_Args...>>::Dim;  /// @brief Space dimension
  static constexpr int DoF = ListInfo<List<_Args...>>::DoF;  /// @brief Degrees of freedom
};

template <typename... _Args>
struct LieGroupProperties<BundleTangentBase<Eigen::Map<BundleTangent<_Args...>>>> {
  static constexpr int Dim = ListInfo<List<_Args...>>::Dim;  /// @brief Space dimension
  static constexpr int DoF = ListInfo<List<_Args...>>::DoF;  /// @brief Degrees of freedom
};

}  // namespace internal
}  // namespace manif

#endif
