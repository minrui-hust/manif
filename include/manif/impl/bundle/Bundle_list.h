#ifndef _MANIF_MANIF_BUNDLE_LIST_H_
#define _MANIF_MANIF_BUNDLE_LIST_H_

#include <Eigen/Core>
#include "manif/impl/traits.h"

namespace manif {

struct nulltype;

struct Range {
  unsigned int start;
  unsigned int size;
};

enum {
  LieGroup = 0,
  Tangent = 1,
  LieAlg = 2,
};

// helper class to get the dest type from raw type
template <unsigned int _type, typename _RawType>
struct TypeHelper {};

namespace internal {
template <typename _RawType>
struct traits<TypeHelper<LieGroup, _RawType>> {
  using Type = Eigen::Map<typename _RawType::LieGroup>;
};

template <typename _RawType>
struct traits<TypeHelper<Tangent, _RawType>> {
  using Type = Eigen::Map<typename _RawType::Tangent>;
};

template <typename _RawType>
struct traits<TypeHelper<LieAlg, _RawType>> {
  using Type = typename _RawType::Tangent::LieAlg;
};

}  // namespace internal

template <unsigned int _type, typename... _Args>
struct List {};

template <unsigned int _id, typename _List>
struct ListHelper {};

template <typename _List>
struct ListInfo {};

template <unsigned int _id, typename _List>
struct ElementInfo {};

template <unsigned int _id, unsigned int _type, typename _Head, typename... _Tails>
struct ListHelper<_id, List<_type, _Head, _Tails...>> {
  using HeadType = typename ListHelper<_id - 1, List<_type, _Tails...>>::HeadType;
  using ListType = typename ListHelper<_id - 1, List<_type, _Tails...>>::ListType;
};

template <unsigned int _type, typename _Head, typename... _Tails>
struct ListHelper<0, List<_type, _Head, _Tails...>> {
  using HeadType = typename internal::traits<TypeHelper<_type, _Head>>::Type;
  using ListType = List<_type, _Head, _Tails...>;
};

template <unsigned int _id, unsigned int _type>
struct ListHelper<_id, List<_type>> {
  static_assert(_id < 0, "Index out of bound");
};

template <unsigned int _type, typename _Head, typename... _Tails>
struct List<_type, _Head, _Tails...> : public List<_type, _Tails...> {
 private:
  using Base = List<_type, _Tails...>;
  using Head = typename internal::traits<TypeHelper<_type, _Head>>::Type;
  using Scalar = typename ListInfo<List>::Scalar;

  template <unsigned int _id>
  using SubListType = typename ListHelper<_id, List>::ListType;

  template <unsigned int _id>
  using ElementType = typename ListHelper<_id, List>::HeadType;

 public:
  List() = default;
  List(Scalar* data) : Base(data + Head::RepSize), head_(data) {}

  template <unsigned int _id>
  ElementType<_id>& get() {
    return static_cast<SubListType<_id>&>(*this).head();
  }

  template <unsigned int _id>
  const ElementType<_id>& get() const {
    return static_cast<const SubListType<_id>&>(*this).head();
  }

  template <typename _First, typename... _Remains>
  void set(const _First& first, const _Remains&... remains) {
    head_ = first;
    static_cast<Base&>(*this).set(remains...);
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

template <unsigned int _type>
struct List<_type> {
  List() = default;

  // terminator
  template <typename _Scalar>
  List(_Scalar* data) {}

  // teminator
  void set() {}

  // terminator
  template <unsigned int _id = 0, typename _Functor>
  void traverse(_Functor&&) {}

  // terminator
  template <unsigned int _id = 0, typename _Functor>
  void traverse(_Functor&&) const {}
};

template <unsigned int _type, typename _Head, typename... _Tails>
struct ListInfo<List<_type, _Head, _Tails...>> {
  static constexpr unsigned int RepSize = _Head::RepSize + ListInfo<List<_type, _Tails...>>::RepSize;
  static constexpr unsigned int DoF = _Head::DoF + ListInfo<List<_type, _Tails...>>::DoF;
  static constexpr unsigned int Dim = _Head::Dim + ListInfo<List<_type, _Tails...>>::Dim;
  static constexpr unsigned int Size = 1 + ListInfo<List<_type, _Tails...>>::Size;

  using Scalar = typename _Head::Scalar;
  static_assert(std::is_same<Scalar, typename ListInfo<List<_type, _Tails...>>::Scalar>::value ||
                    std::is_same<nulltype, typename ListInfo<List<_type, _Tails...>>::Scalar>::value,
                "Scalar type in list should be identical");
};

template <unsigned int _type>
struct ListInfo<List<_type>> {
  static constexpr unsigned int RepSize = 0;
  static constexpr unsigned int DoF = 0;
  static constexpr unsigned int Dim = 0;
  static constexpr unsigned int Size = 0;

  using Scalar = nulltype;
};

template <unsigned int _id, unsigned int _type, typename _Head, typename... _Tails>
struct ElementInfo<_id, List<_type, _Head, _Tails...>> {
  static constexpr unsigned int RepIndex = _Head::RepSize + ElementInfo<_id - 1, List<_type, _Tails...>>::RepIndex;
  static constexpr unsigned int DoFIndex = _Head::DoF + ElementInfo<_id - 1, List<_type, _Tails...>>::DoFIndex;
  static constexpr unsigned int DimIndex = _Head::Dim + ElementInfo<_id - 1, List<_type, _Tails...>>::DimIndex;

  using Type = typename ElementInfo<_id - 1, List<_type, _Tails...>>::Type;
};

template <unsigned int _type, typename _Head, typename... _Tails>
struct ElementInfo<0, List<_type, _Head, _Tails...>> {
  static constexpr unsigned int RepIndex = 0;
  static constexpr unsigned int DoFIndex = 0;
  static constexpr unsigned int DimIndex = 0;

  using Type = typename internal::traits<TypeHelper<_type, _Head>>::Type;
};

template <unsigned int _id, unsigned int _type>
struct ElementInfo<_id, List<_type>> {
  static_assert(_id < 0, "Index out of bound");
};

template <typename... _Args>
using LieGroupList = List<LieGroup, _Args...>;

template <typename... _Args>
using TangentList = List<Tangent, _Args...>;

template <typename... _Args>
using LieAlgList = List<LieAlg, _Args...>;

}  // namespace manif

#endif
