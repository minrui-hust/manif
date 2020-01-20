#ifndef _MANIF_MANIF_BUNDLELIEALG_H_
#define _MANIF_MANIF_BUNDLELIEALG_H_

namespace manif {
namespace internal {

template <typename... _Args>
struct BundleLieAlg;

template <unsigned int _id, typename... _Args>
struct BundleLieAlgHelper {};

template <unsigned int _id, typename _Head, typename... _Tails>
struct BundleLieAlgHelper<_id, BundleLieAlg<_Head, _Tails...>> {
  using LieAlgType = typename BundleLieAlgHelper<_id - 1, BundleLieAlg<_Tails...>>::LieAlgType;
  using BundleLieAlgType = typename BundleLieAlgHelper<_id - 1, BundleLieAlg<_Tails...>>::BundleLieAlgType;
};

template <typename _Head, typename... _Tails>
struct BundleLieAlgHelper<0, BundleLieAlg<_Head, _Tails...>> {
  using LieALgType = typename _Head::LieAlg;
  using BundleLieALgType = BundleLieAlg<_Head, _Tails...>;
};

template <typename _Head, typename... _Tails>
struct BundleLieAlg<_Head, _Tails...> : public BundleLieAlg<_Tails...> {
 private:
  using Base = BundleLieAlg<_Tails...>;
  using Head = typename _Head::LieAlg;

  template <unsigned int _id>
  using SubBundleLieAlgType = typename BundleLieAlgHelper<_id, BundleLieAlg>::BundleLieAlgType;

  template <unsigned int _id>
  using LieAlgType = typename BundleLieAlgHelper<_id, BundleLieAlg>::LieAlgType;

 public:
  template <unsigned int _id>
  LieAlgType<_id>& get() {
    return static_cast<SubBundleLieAlgType<_id>&>(*this).head();
  }

  template <unsigned int _id>
  const LieAlgType<_id>& get() const {
    return static_cast<const SubBundleLieAlgType<_id>&>(*this).head();
  }

  Head& head() { return head_; }
  const Head& head() const { return head_; }

 protected:
  typename Head::LieAlg head_;
};

}  // namespace internal
}  // namespace manif

#endif
