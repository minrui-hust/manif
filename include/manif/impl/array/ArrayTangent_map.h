#ifndef _MANIF_MANIF_ARRAYTANGENT_MAP_H_
#define _MANIF_MANIF_ARRAYTANGENT_MAP_H_

#include "manif/impl/array/ArrayTangent.h"

namespace manif {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _ElementType, int _N>
struct traits<Eigen::Map<ArrayTangent<_ElementType, _N>, 0>> : public traits<ArrayTangent<_ElementType, _N>> {
  using typename traits<ArrayTangent<_ElementType, _N>>::Scalar;
  using traits<ArrayTangent<_ElementType, _N>>::DoF;
  using Base = ArrayTangentBase<Eigen::Map<ArrayTangent<_ElementType, _N>, 0>>;
  using DataType = ::Eigen::Map<Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template <typename _ElementType, int _N>
struct traits<Eigen::Map<const ArrayTangent<_ElementType, _N>, 0>>
    : public traits<const ArrayTangent<_ElementType, _N>> {
  using typename traits<const ArrayTangent<_ElementType, _N>>::Scalar;
  using traits<const ArrayTangent<_ElementType, _N>>::DoF;
  using Base = ArrayTangentBase<Eigen::Map<const ArrayTangent<_ElementType, _N>, 0>>;
  using DataType = ::Eigen::Map<const Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

} /* namespace internal */
} /* namespace manif */

namespace Eigen {

/**
 * @brief Specialization of Map for manif::SO3Tangent
 */
template <typename _ElementType, int _N>
class Map<manif::ArrayTangent<_ElementType, _N>, 0>
    : public manif::ArrayTangentBase<Map<manif::ArrayTangent<_ElementType, _N>, 0>> {
  using Base = manif::ArrayTangentBase<Map<manif::ArrayTangent<_ElementType, _N>, 0>>;
  using ElementType = typename Base::ElementType;
  using ArrayType = typename Base::ArrayType;

  static constexpr bool IsDynamic = Base::IsDynamic;
  static constexpr int Size = Base::Size;
  static constexpr int ElementRepSize = ElementType::RepSize;

 public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  Map(Scalar* coeffs) : data_(coeffs) {
    static_assert(!IsDynamic, "Dynamic size Map should be construct with size");
    array_.resize(Size, ElementType(nullptr));
    for (auto i = 0; i < Size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  Map(Scalar* coeffs, int size) : data_(coeffs) {
    static_assert(IsDynamic, "Static size Map should not construct with size");
    array_.resize(size, ElementType(nullptr));
    for (auto i = 0; i < size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  DataType& coeffs() { return data_; }
  const DataType& coeffs() const { return data_; }

 protected:
  friend Base;
  const ArrayType& array() const { return array_; }
  ArrayType& array() { return array_; }

  DataType data_;
  ArrayType array_;
};

/**
 * @brief Specialization of Map for const manif::SO3Tangent
 */
template <typename _ElementType, int _N>
class Map<const manif::ArrayTangent<_ElementType, _N>, 0>
    : public manif::ArrayTangentBase<Map<const manif::ArrayTangent<_ElementType, _N>, 0>> {
  using Base = manif::ArrayTangentBase<Map<const manif::ArrayTangent<_ElementType, _N>, 0>>;
  using ElementType = typename Base::ElementType;
  using ArrayType = typename Base::ArrayType;

  static constexpr bool IsDynamic = Base::IsDynamic;
  static constexpr int Size = Base::Size;
  static constexpr int ElementRepSize = ElementType::RepSize;

 public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  Map(const Scalar* coeffs) : data_(coeffs) {
    static_assert(!IsDynamic, "Dynamic size Map should be construct with size");
    array_.resize(Size, ElementType(nullptr));
    for (auto i = 0; i < Size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  Map(const Scalar* coeffs) : data_(coeffs) {
    static_assert(IsDynamic, "Static size Map should not construct with size");
    array_.resize(size, ElementType(nullptr));
    for (auto i = 0; i < size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  const DataType& coeffs() const { return data_; }

 protected:
  friend Base;
  const ArrayType& array() const { return array_; }

  const DataType data_;
  const ArrayType array_;
};

} /* namespace Eigen */

#endif
