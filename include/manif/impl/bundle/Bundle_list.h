#ifndef _MANIF_MANIF_BUNDLE_LIST_H_
#define _MANIF_MANIF_BUNDLE_LIST_H_

#include "manif/Rn.h"
#include "manif/SE2.h"
#include "manif/SE3.h"
#include "manif/SO2.h"
#include "manif/SO3.h"

namespace manif {
struct nulltype;

template <typename... _Args>
struct List {};

template <unsigned int _id, typename... _Args>
struct ListHelper {};

template <unsigned int _id, typename _Head, typename... _Tails>
struct ListHelper<_id, List<_Head, _Tails...>> {
  using DataType = typename ListHelper<_id - 1, List<_Tails...>>::DataType;
  using ListType = typename ListHelper<_id - 1, List<_Tails...>>::ListType;
};

template <typename _Head, typename... _Tails>
struct ListHelper<0, List<_Head, _Tails...>> {
  using DataType = Eigen::Map<_Head>;
  using ListType = List<_Head, _Tails...>;
};

template <unsigned int _id>
struct ListHelper<_id, List<>> {
  static_assert(_id < 0, "Index out of bound");
};

template <typename _Head, typename... _Tails>
struct List<_Head, _Tails...> : public List<_Tails...> {
  using Base = List<_Tails...>;
  using Data = Eigen::Map<_Head>;

  using Scalar = typename _Head::Scalar;
  static_assert(std::is_same<Scalar, typename Base::Scalar>::value ||
                    std::is_same<typename Base::Scalar, nulltype>::value,
                "Scalar mismatch in bundle");

  template <unsigned int _id>
  using ListElementType = typename ListHelper<_id, List>::DataType;

  template <unsigned int _id>
  using SubListType = typename ListHelper<_id, List>::ListType;

  template <unsigned int _id>
  ListElementType<_id>& get() {
    return static_cast<SubListType<_id>&>(*this).head();
  }

  template <unsigned int _id>
  const ListElementType<_id>& get() const {
    return static_cast<const SubListType<_id>&>(*this).head();
  }

  void set(const _Head& head, const _Tails&... tails) {
    head_ = head;
    static_cast<Base&>(*this).set(tails...);
  }

  // the only constructor
  List(Scalar* data) : head_(data), Base(data + _Head::RepSize) {}

  Data& head() { return head_; }
  const Data& head() const { return head_; }

 protected:
  Data head_;
};

template <>
struct List<> {
  using Scalar = nulltype;
};

template <typename _List>
struct LieGroupListInfo {};

template <unsigned int _id, typename _List>
struct LieGroupListElementInfo {};

template <typename _List>
struct LieGroupListOperation {};

template <typename _Head, typename... _Tails>
struct LieGroupListInfo<List<_Head, _Tails...>> {
  static constexpr unsigned int RepSize = _Head::RepSize + LieGroupListInfo<List<_Tails...>>::RepSize;
  static constexpr unsigned int DoF = _Head::DoF + LieGroupListInfo<List<_Tails...>>::DoF;
  static constexpr unsigned int Dim = _Head::Dim + LieGroupListInfo<List<_Tails...>>::Dim;
  static constexpr unsigned int Size = 1 + LieGroupListInfo<List<_Tails...>>::Size;
};

template <>
struct LieGroupListInfo<List<>> {
  static constexpr unsigned int RepSize = 0;
  static constexpr unsigned int DoF = 0;
  static constexpr unsigned int Dim = 0;
  static constexpr unsigned int Size = 0;
};

template <unsigned int _id, typename _Head, typename... _Tails>
struct LieGroupListElementInfo<_id, List<_Head, _Tails...>> {
  static constexpr unsigned int RepIndex = _Head::RepSize + LieGroupListElementInfo<_id - 1, List<_Tails...>>::RepIndex;
  static constexpr unsigned int DoFIndex = _Head::DoF + LieGroupListElementInfo<_id - 1, List<_Tails...>>::DoFIndex;
  static constexpr unsigned int DimIndex = _Head::Dim + LieGroupListElementInfo<_id - 1, List<_Tails...>>::DimIndex;
};

template <typename _Head, typename... _Tails>
struct LieGroupListElementInfo<0, List<_Head, _Tails...>> {
  static constexpr unsigned int RepIndex = 0;
  static constexpr unsigned int DoFIndex = 0;
  static constexpr unsigned int DimIndex = 0;
};

template <unsigned int _id>
struct LieGroupListElementInfo<_id, List<>> {
  static_assert(_id < 0, "Index out of bound");
};

template <typename _Head, typename... _Tails>
struct LieGroupListOperation<List<_Head, _Tails...>> {
  using ThisList = List<_Head, _Tails...>;
  using NextList = List<_Tails...>;
  using NextOperation = LieGroupListOperation<NextList>;
  using HeadOptJacobianRef = typename _Head::OptJacobianRef;

  static constexpr unsigned int HeadDoF = _Head::DoF;
  static constexpr unsigned int ThisDoF = LieGroupListInfo<ThisList>::DoF;
  static constexpr unsigned int NextDoF = LieGroupListInfo<NextList>::DoF;

  template <unsigned int _index = 0, typename _OptJacobianRef>
  static void BundleInverse(const ThisList& origin, ThisList& inversed, _OptJacobianRef jac_minv_m) {
    // process head
    HeadOptJacobianRef head_jac_minv_m;
    if (!jac_minv_m.empty()) {
      head_jac_minv_m.emplace(jac_minv_m->template block<_Head::DoF, _Head::DoF>(_index, _index));
    }
    inversed.head() = origin.head().inverse(head_jac_minv_m);

    // process tails recursively
    NextOperation::BundleInverse<_index + HeadDoF>(origin, inversed, jac_minv_m);
  }
};

template <>
struct LieGroupListOperation<List<>> {
  template <unsigned int _index, typename ThisList, typename ThisOptJacobianRef>
  static void BundleInverse(const ThisList&, ThisList&, ThisOptJacobianRef) {
    // termimator
  }
};

template <typename _List>
struct TangentListInfo {};

template <unsigned int _id, typename _List>
struct TangentListElementInfo {};

template <typename _List>
struct TangentListOperation {};

}  // namespace manif

#endif
