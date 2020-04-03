#ifndef _MANIF_MANIF_ARRAY_H_
#define _MANIF_MANIF_ARRAY_H_

#include "manif/impl/array/Array_base.h"

namespace manif {

template <typename _ElementType, int _N>
struct Array;

template <typename _ElementType, int _N>
struct ArrayTangent;

namespace internal {
template <typename _ElementType, int _N>
struct traits<Array<_ElementType, _N>> {
  using Scalar = typename ElementType_::Scalar;
  using ElementType = Eigen::Map<typename _ElementType::LieGroup>;
  using ArrayType = std::vector<ElementType>;  // ElementType is an Map, so aligened allocator is not required

  using LieGroup = Array<_ElementType, _N>;
  using Tangent = ArrayTangent<_ElementType, _N>;

  using Base = ArrayBase<LieGroup>;

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
  using ElementType = typename Base::ElementType;
  using ArrayType = typename Base::ArrayType;

  static constexpr bool IsDynamic = Base::IsDynamic;
  static constexpr int Size = Base::Size;
  static constexpr int ElementRepSize = ElementType::RepSize;

 public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND

  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Array() {
    if (!IsDynamic) {
      array_.resize(Size, ElementType(nullptr));
      for (auto i = 0; i < Size; ++i) {
        array_[i] = ElementType(data_.data() + i * ElementRepSize);
      }
    }
  }

  Array(int size) {
    int real_size = IsDynamic ? size : Size;  // parameter size is ignored in fixed size
    array_.resize(Size, ElementType(nullptr));
    for (auto i = 0; i < real_size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  template <typename _EigenDerived>
  Array(const Eigen::MatrixBase<_EigenDerived>& data) : data_(data) {
    // Todo: 1.make sure data.rows() is integer times of ElementRepSize in dynamic mode
    //       2.make sure data.rows() equal Size*ElementReqSize in static mode
    int real_size = IsDynamic ? data.rows() / ElementRepSize : Size;
    array_.resize(_N, ElementType(nullptr));
    for (auto i = 0; i < real_size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  Array(const Array& o) : Array(o.coeffs()) {}

  template <typename _DerivedOther>
  Array(const BundleBase<_DerivedOther>& o) : Array(o.coeffs()) {}

  template <typename _DerivedOther>
  Array(const LieGroupBase<_DerivedOther>& o) : Array(o.coeffs()) {}

  const DataType& coeffs() const { return data_; }

  const ArrayType& array() const { return array_; }

  ArrayType& array() { return array_; }

 protected:
  friend struct LieGroupBase<Array<_ElementType, _N>>;
  DataType& coeffs_nonconst() { return data_; }

  DataType data_;
  ArrayType array_;
};

}  // namespace manif

#endif
