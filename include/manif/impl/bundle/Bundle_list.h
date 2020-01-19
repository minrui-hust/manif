#ifndef _MANIF_MANIF_BUNDLE_LIST_H_
#define _MANIF_MANIF_BUNDLE_LIST_H_

#include <Eigen/Core>

namespace manif {

struct nulltype;

template <typename... _Args>
struct List {};

template <unsigned int _id, typename... _Args>
struct ListHelper {};

template <typename _List>
struct ListInfo {};

template <unsigned int _id, typename _List>
struct ElementInfo {};

template <unsigned int _id, typename _Head, typename... _Tails>
struct ListHelper<_id, List<_Head, _Tails...>> {
  using HeadType = typename ListHelper<_id - 1, List<_Tails...>>::HeadType;
  using ListType = typename ListHelper<_id - 1, List<_Tails...>>::ListType;
};

template <typename _Head, typename... _Tails>
struct ListHelper<0, List<_Head, _Tails...>> {
  using HeadType = Eigen::Map<_Head>;
  using ListType = List<_Head, _Tails...>;
};

template <unsigned int _id>
struct ListHelper<_id, List<>> {
  static_assert(_id < 0, "Index out of bound");
};

template <typename _Head, typename... _Tails>
struct List<_Head, _Tails...> : public List<_Tails...> {
 private:
  using Base = List<_Tails...>;
  using Head = Eigen::Map<_Head>;
  using Scalar = typename ListInfo<List>::Scalar;

  template <unsigned int _id>
  using SubListType = typename ListHelper<_id, List>::ListType;

  template <unsigned int _id>
  using ElementType = typename ListHelper<_id, List>::HeadType;

 public:
  List(Scalar* data) : Base(data + _Head::RepSize), head_(data) {}

  template <unsigned int _id>
  ElementType<_id>& get() {
    return static_cast<SubListType<_id>&>(*this).head();
  }

  template <unsigned int _id>
  const ElementType<_id>& get() const {
    return static_cast<const SubListType<_id>&>(*this).head();
  }

  void set(const _Head& head, const _Tails&... tails) {
    head_ = head;
    static_cast<Base&>(*this).set(tails...);
  }

  template <unsigned int _id = 0, typename _Functor>
  void traverse(_Functor&& func) {
    func.template operator()<_id>(head());
    static_cast<Base&>(*this).traverse<_id + 1>(std::forward<decltype(func)>(func));
  }

  template <unsigned int _id = 0, typename _Functor>
  void traverse(_Functor&& func) const {
    func.template operator()<_id>(head());
    static_cast<const Base&>(*this).traverse<_id + 1>(std::forward<decltype(func)>(func));
  }

  Head& head() { return head_; }
  const Head& head() const { return head_; }

 protected:
  Head head_;
};

template <>
struct List<> {
  // terminator
  template <typename _Scalar>
  List(_Scalar* data) {}

  // terminator
  template <unsigned int _id = 0, typename _Functor>
  void traverse(_Functor&&) {}

  // terminator
  template <unsigned int _id = 0, typename _Functor>
  void traverse(_Functor&&) const {}
};

template <typename _Head, typename... _Tails>
struct ListInfo<List<_Head, _Tails...>> {
  static constexpr unsigned int RepSize = _Head::RepSize + ListInfo<List<_Tails...>>::RepSize;
  static constexpr unsigned int DoF = _Head::DoF + ListInfo<List<_Tails...>>::DoF;
  static constexpr unsigned int Dim = _Head::Dim + ListInfo<List<_Tails...>>::Dim;
  static constexpr unsigned int Size = 1 + ListInfo<List<_Tails...>>::Size;

  using Scalar = typename _Head::Scalar;
  static_assert(std::is_same<Scalar, typename ListInfo<List<_Tails...>>::Scalar>::value ||
                    std::is_same<nulltype, typename ListInfo<List<_Tails...>>::Scalar>::value,
                "Scalar type in list should be identical");
};

template <>
struct ListInfo<List<>> {
  static constexpr unsigned int RepSize = 0;
  static constexpr unsigned int DoF = 0;
  static constexpr unsigned int Dim = 0;
  static constexpr unsigned int Size = 0;

  using Scalar = nulltype;
};

template <unsigned int _id, typename _Head, typename... _Tails>
struct ElementInfo<_id, List<_Head, _Tails...>> {
  static constexpr unsigned int RepIndex = _Head::RepSize + ElementInfo<_id - 1, List<_Tails...>>::RepIndex;
  static constexpr unsigned int DoFIndex = _Head::DoF + ElementInfo<_id - 1, List<_Tails...>>::DoFIndex;
  static constexpr unsigned int DimIndex = _Head::Dim + ElementInfo<_id - 1, List<_Tails...>>::DimIndex;

  using Type = typename ElementInfo<_id - 1, List<_Tails...>>::Type;
};

template <typename _Head, typename... _Tails>
struct ElementInfo<0, List<_Head, _Tails...>> {
  static constexpr unsigned int RepIndex = 0;
  static constexpr unsigned int DoFIndex = 0;
  static constexpr unsigned int DimIndex = 0;

  using Type = Eigen::Map<_Head>;
};

template <unsigned int _id>
struct ElementInfo<_id, List<>> {
  static_assert(_id < 0, "Index out of bound");
};

}  // namespace manif

#endif
