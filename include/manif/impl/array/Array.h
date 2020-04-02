#ifndef _MANIF_MANIF_ARRAY_H_
#define _MANIF_MANIF_ARRAY_H_

#include "manif/impl/array/Array_base.h"

namespace manif {

template <typename _ElementType, int _N>
struct Array;

template <typename _ElementType, int _N>
struct ArrayTangent;

namespace internal {
//! Traits specialization
template <typename _ElementType, int _N>
struct traits<Array<_ElementType, _N>> {
  using ElementType = _ElementType;
  using Scalar = typename ElementType::Scalar;

  // If is fixed size, std::array is used, if is dynamic size, std::vector is used
  using ArrayType = std::vector<Eigen::Map<typename ElementType::LieGroup>>;

  using LieGroup = Array<ElementType, _N>;
  using Tangent = ArrayTangent<ElementType, _N>;

  using Base = ArrayBase<Array<ElementType, _N>>;

  static constexpr bool IsDynamic = (_N == Dynamic);

  static constexpr int Dim = IsDynamic ? _N : ElementType::Dim * _N;
  static constexpr int DoF = IsDynamic ? _N : ElementType::DoF * _N;
  static constexpr int RepSize = IsDynamic ? _N : ElementType::RepSize * _N;

  static constexpr int Size = _N;

  using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  using Vector = Eigen::Matrix<Scalar, Dim, 1>;
};
}  // namespace internal

/**
 * @brief Represents a Array of LieGroup elements
 */
template <typename _ElementType, int _N>
struct Array : ArrayBase<Array<_ElementType, _N>> {
 private:
  using Base = ArrayBase<Array<_ElementType, _N>>;
  using Type = Array<_ElementType, _N>;
  using ElementType = typename Base::ElementType;
  using ArrayType = typename Base::ArrayType;

  static constexpr bool IsDynamic = Base::IsDynamic;
  static constexpr int ElementRepSize = ElementType::RepSize;

 public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND

  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Array();

  Array(int size);

  // Copy constructor
  Array(const Bundle& o);

  template <typename _DerivedOther>
  Array(const BundleBase<_DerivedOther>& o);

  template <typename _DerivedOther>
  Array(const LieGroupBase<_DerivedOther>& o);

  // Copy constructor given Eigen
  template <typename _EigenDerived>
  Array(const Eigen::MatrixBase<_EigenDerived>& data);

  const DataType& coeffs() const { return data_; }

  const ArrayType& array() const { return array_; }

  ArrayType& array() { return array_; }

 protected:
  friend struct LieGroupBase<Array<_ElementType, _N>>;
  DataType& coeffs_nonconst() { return data_; }

  DataType data_;
  ArrayType array_;
};

template <typename _ElementType, int _N>
Array<_ElementType, _N>::Array() {
  if (!IsDynamic) {
    array_.resize(_N, Eigen::Map<typename ElementType::LieGroup>(nullptr));
    for (auto i = 0; i < _N; ++i) {
      array_[i] = Eigen::Map<typename ElementType::LieGroup>(data_.data() + i * ElementRepSize);
    }
  }
}

template <typename _ElementType, int _N>
Array<_ElementType, _N>::Array(int size) : data_(size * ElementRepSize) {
  int real_size = IsDynamic ? size : _N;
  array_.resize(_N, Eigen::Map<typename ElementType::LieGroup>(nullptr));
  for (auto i = 0; i < real_size; ++i) {
    array_[i] = Eigen::Map<typename ElementType::LieGroup>(data_.data() + i * ElementRepSize);
  }
}

template <typename _ElementType, int _N>
template <typename _EigenDerived>
Array<_ElementType, _N>::Array(const Eigen::MatrixBase<_EigenDerived>& data) : data_(data) {
  int real_size = IsDynamic ? data.rows() / ElementRepSize : _N;
  array_.resize(_N, Eigen::Map<typename ElementType::LieGroup>(nullptr));
  for (auto i = 0; i < real_size; ++i) {
    array_[i] = Eigen::Map<typename ElementType::LieGroup>(data_.data() + i * ElementRepSize);
  }
}

template <typename _ElementType, int _N>
Array<_ElementType, _N>::Array(const Array& o) : Array(o.coeffs()) {
  //
}

template <typename _ElementType, int _N>
template <typename _DerivedOther>
Array<_ElementType, _N>::Array(const ArrayBase<_DerivedOther>& o) : Array(o.coeffs()) {
  //
}

template <typename _ElementType, int _N>
template <typename _DerivedOther>
Array<_ElementType, _N>::Array(const LieGroupBase<_DerivedOther>& o) : Array(o.coeffs()) {
  //
}

}  // namespace manif

#endif
