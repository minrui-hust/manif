#ifndef _MANIF_MANIF_ARRAY_MAP_H_
#define _MANIF_MANIF_ARRAY_MAP_H_

#include "manif/impl/array/Array.h"

namespace manif {
namespace internal {

//! @brief traits specialization for Eigen Map
template <typename _ElementType, int _N>
struct traits<Eigen::Map<Array<_ElementType, _N>, 0>> : public traits<Array<_ElementType, _N>> {
  using typename traits<Array<_ElementType, _N>>::Scalar;
  using traits<Array<_ElementType, _N>>::RepSize;
  using Base = ArrayBase<Eigen::Map<Array<_ElementType, _N>, 0>>;
  using DataType = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template <typename _ElementType, int _N>
struct traits<Eigen::Map<const Array<_ElementType, _N>, 0>> : public traits<const Array<_ElementType, _N>> {
  using typename traits<const Array<_ElementType, _N>>::Scalar;
  using traits<const Array<_ElementType, _N>>::RepSize;
  using Base = ArrayBase<Eigen::Map<const Array<_ElementType, _N>, 0>>;
  using DataType = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

} /* namespace internal */
} /* namespace manif */

namespace Eigen {

/**
 * @brief Specialization of Map for manif::Array
 */
template <typename _ElementType, int _N>
class Map<manif::Array<_ElementType, _N>, 0> : public manif::ArrayBase<Map<manif::Array<_ElementType, _N>, 0>> {
 private:
  using Base = manif::ArrayBase<Map<manif::Array<_ElementType, _N>, 0>>;
  using ElementType = typename Base::ElementType;
  using ArrayType = typename Base::ArrayType;

  static constexpr bool IsDynamic = Base::IsDynamic;
  static constexpr int Size = Base::Size;
  static constexpr int ElementRepSize = ElementType::RepSize;

 public:
  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

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

  const DataType& coeffs() const { return data_; }

  const ArrayType& array() const { return array_; }

  ArrayType& array() { return array_; }

 protected:
  friend struct manif::LieGroupBase<Map<manif::Array<_ElementType, _N>, 0>>;
  DataType& coeffs_nonconst() { return data_; }

  DataType data_;
  ArrayType array_;
};

/**
 * @brief Specialization of Map for const manif::Bundle
 */
template <typename _ElementType, int _N>
class Map<const manif::Array<_ElementType, _N>, 0>
    : public manif::ArrayBase<Map<const manif::Array<_ElementType, _N>, 0>> {
 private:
  using Base = manif::ArrayBase<Map<const manif::Array<_ElementType, _N>, 0>>;
  using ElementType = typename Base::ElementType;
  using ArrayType = typename Base::ArrayType;

  static constexpr bool IsDynamic = Base::IsDynamic;
  static constexpr int Size = Base::Size;
  static constexpr int ElementRepSize = ElementType::RepSize;

 public:
  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Map(const Scalar* coeffs) : data_(coeffs) {
    static_assert(!IsDynamic, "Dynamic Map should be construct with size");
    array_.resize(Size, ElementType(nullptr));
    for (auto i = 0; i < Size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  Map(const Scalar* coeffs, int size) : data_(coeffs) {
    static_assert(IsDynamic, "Static Map should not construct with size");
    array_.resize(size, ElementType(nullptr));
    for (auto i = 0; i < size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  const DataType& coeffs() const { return data_; }

  const ArrayType& array() const { return array_; }

 protected:
  const DataType data_;
  const ArrayType array_;
};

} /* namespace Eigen */

#endif
