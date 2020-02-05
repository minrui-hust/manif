#include "manif/Bundle.h"
#include "manif/Rn.h"
#include "manif/SE2.h"
#include "manif/SE3.h"
#include "manif/SO2.h"
#include "manif/SO3.h"

#include "../common_tester.h"

using namespace manif;

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_COEFFS) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;

  double data[BundleTangentMap::RepSize];

  TestBundleTangent t;
  BundleTangentMap t_map(data);

  for (int i = 0; i < 10; ++i) {
    t.setRandom();
    t_map = t;

    EXPECT_EIGEN_NEAR(t.coeffs(), t_map.coeffs());
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_NEST_SIZE) {
  using TestBundleTangent = BundleTangent<Bundle<SO3d, R3d>, R3d>;

  EXPECT_EQ(9, TestBundleTangent::Dim);
  EXPECT_EQ(9, TestBundleTangent::DoF);
  EXPECT_EQ(9, TestBundleTangent::RepSize);
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_NEST_COMMON) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using TestNestBundleTangent = BundleTangent<Bundle<SO3d, R3d>, SO2d>;

  TestBundleTangent t;
  SO2d::Tangent so2_t;
  TestNestBundleTangent nt;

  for (int i = 0; i < 10; ++i) {
    t.setRandom();
    so2_t.setRandom();

    nt.get<0>() = t;
    nt.get<1>() = so2_t;

    EXPECT_EIGEN_NEAR(nt.get<0>().coeffs(), t.coeffs());
    EXPECT_EIGEN_NEAR(nt.get<1>().coeffs(), so2_t.coeffs());
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_HAT) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;

  double data[BundleTangentMap::RepSize];
  BundleTangentMap bt(data);
  bt.get<0>() = Eigen::Vector3d(1.0, 2.0, 3.0);
  bt.get<1>() = Eigen::Vector3d(4.0, 5.0, 6.0);

  for (int i = 0; i < 10; ++i) {
    TestBundleTangent::LieAlg bt_alg = bt.hat();

    SO3d::Tangent::LieAlg so3_alg = bt.get<0>().hat();
    R3d::Tangent::LieAlg r3_alg = bt.get<1>().hat();

    EXPECT_EIGEN_NEAR(bt_alg.get<0>(), so3_alg);
    EXPECT_EIGEN_NEAR(bt_alg.get<1>(), r3_alg);

    bt.setRandom();
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_EXP) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;
  using BundleJacobian = TestBundleTangent::Jacobian;

  double data[BundleTangentMap::RepSize];
  BundleTangentMap bt(data);
  bt.get<0>() = Eigen::Vector3d(1.0, 2.0, 3.0);
  bt.get<1>() = Eigen::Vector3d(4.0, 5.0, 6.0);

  for (int i = 0; i < 10; ++i) {
    BundleJacobian bundle_jac;
    bundle_jac.setRandom();
    TestBundleTangent::LieGroup bt_m = bt.exp(bundle_jac);

    SO3d::Tangent::Jacobian so3_jac;
    R3d::Tangent::Jacobian r3_jac;
    SO3d::Tangent::LieGroup so3_m = bt.get<0>().exp(so3_jac);
    R3d::Tangent::LieGroup r3_m = bt.get<1>().exp(r3_jac);

    EXPECT_EIGEN_NEAR(bt_m.get<0>().coeffs(), so3_m.coeffs());
    EXPECT_EIGEN_NEAR(bt_m.get<1>().coeffs(), r3_m.coeffs());

    BundleJacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bundle_jac, predict_jac);

    bt.setRandom();
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_RJAC) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;

  double data[BundleTangentMap::RepSize];
  BundleTangentMap bt(data);
  bt.get<0>() = Eigen::Vector3d(1.0, 2.0, 3.0);
  bt.get<1>() = Eigen::Vector3d(4.0, 5.0, 6.0);

  for (int i = 0; i < 10; ++i) {
    TestBundleTangent::Jacobian bt_jac = bt.rjac();

    SO3d::Tangent::Jacobian so3_jac = bt.get<0>().rjac();
    R3d::Tangent::Jacobian r3_jac = bt.get<1>().rjac();

    TestBundleTangent::Jacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bt_jac, predict_jac);

    bt.setRandom();
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_LJAC) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;

  double data[BundleTangentMap::RepSize];
  BundleTangentMap bt(data);
  bt.get<0>() = Eigen::Vector3d(1.0, 2.0, 3.0);
  bt.get<1>() = Eigen::Vector3d(4.0, 5.0, 6.0);

  for (int i = 0; i < 10; ++i) {
    TestBundleTangent::Jacobian bt_jac = bt.ljac();

    SO3d::Tangent::Jacobian so3_jac = bt.get<0>().ljac();
    R3d::Tangent::Jacobian r3_jac = bt.get<1>().ljac();

    TestBundleTangent::Jacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bt_jac, predict_jac);

    bt.setRandom();
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_RJACINV) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;

  double data[BundleTangentMap::RepSize];
  BundleTangentMap bt(data);
  bt.get<0>() = Eigen::Vector3d(1.0, 2.0, 3.0);
  bt.get<1>() = Eigen::Vector3d(4.0, 5.0, 6.0);

  for (int i = 0; i < 10; ++i) {
    TestBundleTangent::Jacobian bt_jac = bt.rjacinv();

    SO3d::Tangent::Jacobian so3_jac = bt.get<0>().rjacinv();
    R3d::Tangent::Jacobian r3_jac = bt.get<1>().rjacinv();

    TestBundleTangent::Jacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bt_jac, predict_jac);

    bt.setRandom();
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_LJACINV) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;

  double data[BundleTangentMap::RepSize];
  BundleTangentMap bt(data);
  bt.get<0>() = Eigen::Vector3d(1.0, 2.0, 3.0);
  bt.get<1>() = Eigen::Vector3d(4.0, 5.0, 6.0);

  for (int i = 0; i < 10; ++i) {
    TestBundleTangent::Jacobian bt_jac = bt.ljacinv();

    SO3d::Tangent::Jacobian so3_jac = bt.get<0>().ljacinv();
    R3d::Tangent::Jacobian r3_jac = bt.get<1>().ljacinv();

    TestBundleTangent::Jacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bt_jac, predict_jac);

    bt.setRandom();
  }
}

TEST(TEST_BUNDLE_TANGENT_MAP, TEST_BUNDLE_TANGENT_MAP_SMALLADJ) {
  using TestBundleTangent = BundleTangent<SO3d, R3d>;
  using BundleTangentMap = Eigen::Map<TestBundleTangent>;

  double data[BundleTangentMap::RepSize];
  BundleTangentMap bt(data);
  bt.get<0>() = Eigen::Vector3d(1.0, 2.0, 3.0);
  bt.get<1>() = Eigen::Vector3d(4.0, 5.0, 6.0);

  for (int i = 0; i < 10; ++i) {
    TestBundleTangent::Jacobian bt_jac = bt.smallAdj();

    SO3d::Tangent::Jacobian so3_jac = bt.get<0>().smallAdj();
    R3d::Tangent::Jacobian r3_jac = bt.get<1>().smallAdj();

    TestBundleTangent::Jacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bt_jac, predict_jac);

    bt.setRandom();
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
