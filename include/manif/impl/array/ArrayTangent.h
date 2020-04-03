#ifndef _MANIF_MANIF_ARRAYTANGENT_H_
#define _MANIF_MANIF_ARRAYTANGENT_H_

#include <Eigen/Core>
#include "manif/impl/array/ArrayTangent_base.h"

namespace manif {

namespace internal {
template <typename _ElementType, int _N>
struct traits<ArrayTangent<_ElementType, _N>> {
  using Scalar = typename _ElementType::Scalar;
  using ElementType = Eigen::Map<typename _ElementType::Tangent>;
  using ArrayType = std::vector<ElementType>;  // ElementType is an Map, so aligned allocator is not required

  using LieGroup = Array<_ElementType, _N>;
  using Tangent = ArrayTangent<_ElementType, _N>;

  using Base = ArrayTangentBase<Tangent>;

  static constexpr bool IsDynamic = (_N == Dynamic);

  static constexpr int Dim = IsDynamic ? _N : ElementType::Dim * _N;
  static constexpr int DoF = IsDynamic ? _N : ElementType::DoF * _N;
  static constexpr int RepSize = DoF;

  static constexpr int Size = _N;

  using DataType = Eigen::Matrix<Scalar, RepSize, 1>;

  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  using LieAlg = std::vector<_ElementType::LieAlg>;
};
}  // namespace internal

/**
 * @brief Represents an element of tangent space of Bundle.
 */
template <typename _ElementType, int _N>
struct ArrayTangent : ArrayTangentBase<ArrayTangent<_ElementType, _N>> {
 private:
  using Base = ArrayTangentBase<ArrayTangent<_ElementType, _N>>;
  using ElementType = typename Base::ElementType;
  using ArrayType = typename Base::ArrayType;

  static constexpr bool IsDynamic = Base::IsDynamic;
  static constexpr int Size = Base::Size;
  static constexpr int ElementRepSize = ElementType::RepSize;

 public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND

  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  ArrayTangent() {
    if (!IsDynamic) {
      array_.resize(Size, ElementType(nullptr));
      for (auto i = 0; i < Size; ++i) {
        array_[i] = ElementType(data_.data() + i * ElementRepSize);
      }
    }
  }

  ArrayTangent(int size) {
    int real_size = IsDynamic ? size : Size;  // parameter size is ignored in fixed size
    array_.resize(Size, ElementType(nullptr));
    for (auto i = 0; i < real_size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  template <typename _EigenDerived>
  ArrayTangent(const Eigen::MatrixBase<_EigenDerived>& v) : data_(v) {
    // Todo: 1.make sure data.rows() is integer times of ElementRepSize in dynamic mode
    //       2.make sure data.rows() equal Size*ElementReqSize in static mode
    int real_size = IsDynamic ? data.rows() / ElementRepSize : Size;
    array_.resize(Size, ElementType(nullptr));
    for (auto i = 0; i < real_size; ++i) {
      array_[i] = ElementType(data_.data() + i * ElementRepSize);
    }
  }

  ArrayTangent(const BundleTangent& o) : ArrayTangent(o.coeffs()) {}

  template <typename _DerivedOther>
  ArrayTangent(const BundleTangentBase<_DerivedOther>& o) : ArrayTangent(o.coeffs()) {}

  template <typename _DerivedOther>
  ArrayTangent(const TangentBase<_DerivedOther>& o) : ArrayTangent(o.coeffs()) {}

  DataType& coeffs() { return data_; }
  const DataType& coeffs() const { return data_; }

 protected:
  friend Base;
  const ArrayType& array() const { return array_; }
  ArrayType& array() { return array_; }

  DataType data_;
  ArrayType array_;
};

}  // namespace manif

#endif
