#include "manif/Bundle.h"
#include "manif/Rn.h"
#include "manif/SE2.h"
#include "manif/SE3.h"
#include "manif/SO2.h"
#include "manif/SO3.h"

#include "../common_tester.h"

using namespace manif;

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_COEFFS) {
  using TestBundle = Bundle<SO3d, R3d>;
  using TestBundleMap = Eigen::Map<TestBundle>;

  double data[TestBundle::RepSize];

  TestBundle b;
  TestBundleMap b_map(data);

  for (int i = 0; i < 10; ++i) {
    b.setRandom();
    b_map = b;

    EXPECT_EIGEN_NEAR(b.coeffs(), b_map.coeffs());
  }
}

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_NEST_SIZE) {
  using TestBundle = Bundle<Bundle<SO3d, R3d>, R3d>;

  EXPECT_EQ(9, TestBundle::Dim);
  EXPECT_EQ(9, TestBundle::DoF);
  EXPECT_EQ(10, TestBundle::RepSize);
}

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_NEST_COMMON) {
  using TestBundle = Bundle<SO3d, R3d>;
  using TestNestBundle = Bundle<TestBundle, SO2d>;

  TestBundle b;
  SO2d so2;
  TestNestBundle nb;

  for (int i = 0; i < 10; ++i) {
    b.setRandom();
    so2.setRandom();

    nb.get<0>() = b;
    nb.get<1>() = so2;

    EXPECT_EIGEN_NEAR(nb.get<0>().coeffs(), b.coeffs());
    EXPECT_EIGEN_NEAR(nb.get<1>().coeffs(), so2.coeffs());
  }
}

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_INVERSE) {
  using TestBundle = Bundle<SO3d, R3d>;
  using TestBundleMap = Eigen::Map<TestBundle>;
  using BundleJacobian = TestBundle::Jacobian;

  double data[TestBundleMap::RepSize];
  TestBundleMap b(data);
  b.get<0>() = SO3d(0, 0, 0.6, 0.8);
  b.get<1>() = R3d(Eigen::Vector3d(1, 2, 3));

  for (int i = 0; i < 10; ++i) {
    BundleJacobian bundle_jac;
    bundle_jac.setRandom();
    TestBundle b_inv = b.inverse(bundle_jac);

    SO3d::Jacobian so3_jac;
    R3d::Jacobian r3_jac;
    SO3d so3_inv = b.get<0>().inverse(so3_jac);
    R3d r3_inv = b.get<1>().inverse(r3_jac);

    EXPECT_EIGEN_NEAR(b_inv.get<0>().coeffs(), so3_inv.coeffs());
    EXPECT_EIGEN_NEAR(b_inv.get<1>().coeffs(), r3_inv.coeffs());

    BundleJacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bundle_jac, predict_jac);

    b.setRandom();
  }
}

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_LOG) {
  using TestBundle = Bundle<SO3d, R3d>;
  using TestBundleMap = Eigen::Map<TestBundle>;
  using BundleJacobian = TestBundle::Jacobian;

  double data[TestBundleMap::RepSize];
  TestBundleMap b(data);
  b.get<0>() = SO3d(0, 0, 0.6, 0.8);
  b.get<1>() = R3d(Eigen::Vector3d(1, 2, 3));

  for (int i = 0; i < 10; ++i) {
    BundleJacobian bundle_jac;
    bundle_jac.setRandom();
    TestBundle::Tangent b_tangent;
    b_tangent = b.log(bundle_jac);

    SO3d::Jacobian so3_jac;
    R3d::Jacobian r3_jac;
    SO3d::Tangent so3_tangent = b.get<0>().log(so3_jac);
    R3d::Tangent r3_tangent = b.get<1>().log(r3_jac);

    EXPECT_EIGEN_NEAR(b_tangent.get<0>().coeffs(), so3_tangent.coeffs());
    EXPECT_EIGEN_NEAR(b_tangent.get<1>().coeffs(), r3_tangent.coeffs());

    BundleJacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bundle_jac, predict_jac);

    b.setRandom();
  }
}

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_COMPOSE) {
  using TestBundle = Bundle<SO3d, R3d>;
  using TestBundleMap = Eigen::Map<TestBundle>;
  using BundleJacobian = TestBundle::Jacobian;
  using SO3Jacobian = SO3d::Jacobian;
  using R3Jacobian = R3d::Jacobian;

  double data_a[TestBundleMap::RepSize];
  TestBundleMap a(data_a);
  a.get<0>() = SO3d(0, 0, 0.6, 0.8);
  a.get<1>() = R3d(Eigen::Vector3d(1, 2, 3));

  double data_b[TestBundleMap::RepSize];
  TestBundleMap b(data_b);
  b.get<0>() = SO3d(0, 0.6, 0, 0.8);
  b.get<1>() = R3d(Eigen::Vector3d(4, 5, 6));

  for (int i = 0; i < 10; ++i) {
    BundleJacobian bundle_jac_a, bundle_jac_b;
    bundle_jac_a.setRandom();
    bundle_jac_b.setRandom();
    TestBundle bundle_composed = a.compose(b, bundle_jac_a, bundle_jac_b);

    SO3Jacobian so3_jac_a, so3_jac_b;
    R3Jacobian r3_jac_a, r3_jac_b;
    SO3d so3_composed = a.get<0>().compose(b.get<0>(), so3_jac_a, so3_jac_b);
    R3d r3_composed = a.get<1>().compose(b.get<1>(), r3_jac_a, r3_jac_b);

    EXPECT_EIGEN_NEAR(bundle_composed.get<0>().coeffs(), so3_composed.coeffs());
    EXPECT_EIGEN_NEAR(bundle_composed.get<1>().coeffs(), r3_composed.coeffs());

    BundleJacobian predict_jac_a, predict_jac_b;
    predict_jac_a.setZero();
    predict_jac_b.setZero();

    predict_jac_a.block<3, 3>(0, 0) = so3_jac_a;
    predict_jac_b.block<3, 3>(0, 0) = so3_jac_b;

    predict_jac_a.block<3, 3>(3, 3) = r3_jac_a;
    predict_jac_b.block<3, 3>(3, 3) = r3_jac_b;

    EXPECT_EIGEN_NEAR(bundle_jac_a, predict_jac_a);
    EXPECT_EIGEN_NEAR(bundle_jac_b, predict_jac_b);

    a.setRandom();
    b.setRandom();
  }
}

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_ACT) {
  using TestBundle = Bundle<SO2d, R2d>;
  using TestBundleMap = Eigen::Map<TestBundle>;
  using BundleVector = TestBundle::Vector;
  using SO2Vector = SO2d::Vector;
  using R2Vector = R2d::Vector;
  using JacobianBundle = Eigen::Matrix<double, 4, 3>;
  using JacobianBundleVector = Eigen::Matrix<double, 4, 4>;
  using JacobianSO2 = Eigen::Matrix<double, 2, 1>;
  using JacobianSO2Vector = Eigen::Matrix<double, 2, 2>;
  using JacobianR2 = Eigen::Matrix<double, 2, 2>;
  using JacobianR2Vector = Eigen::Matrix<double, 2, 2>;

  double data[TestBundleMap::RepSize];
  TestBundleMap b(data);
  b.get<0>() = SO2d(0.6, 0.8);
  b.get<1>() = R2d(Eigen::Vector2d(1, 2));

  BundleVector v;
  v << 1, 2, 3, 4;

  for (int i = 0; i < 10; ++i) {
    JacobianBundle bundle_jac_m;
    JacobianBundleVector bundle_jac_v;
    bundle_jac_m.setRandom();
    bundle_jac_v.setRandom();
    BundleVector bundle_vector = b.act(v, bundle_jac_m, bundle_jac_v);

    JacobianSO2 so2_jac_m;
    JacobianSO2Vector so2_jac_v;
    JacobianR2 r2_jac_m;
    JacobianR2Vector r2_jac_v;
    SO2Vector so2_vector = b.get<0>().act(v.topRows<2>(), so2_jac_m, so2_jac_v);
    R2Vector r2_vector = b.get<1>().act(v.bottomRows<2>(), r2_jac_m, r2_jac_v);

    EXPECT_EIGEN_NEAR(bundle_vector.topRows<2>(), so2_vector);
    EXPECT_EIGEN_NEAR(bundle_vector.bottomRows<2>(), r2_vector);

    JacobianBundle predict_jac_m;
    JacobianBundleVector predict_jac_v;
    predict_jac_m.setZero();
    predict_jac_v.setZero();

    predict_jac_m.block<2, 1>(0, 0) = so2_jac_m;
    predict_jac_v.block<2, 2>(0, 0) = so2_jac_v;

    predict_jac_m.block<2, 2>(2, 1) = r2_jac_m;
    predict_jac_v.block<2, 2>(2, 2) = r2_jac_v;

    EXPECT_EIGEN_NEAR(bundle_jac_m, predict_jac_m);
    EXPECT_EIGEN_NEAR(bundle_jac_v, predict_jac_v);

    b.setRandom();
    v.setRandom();
  }
}

TEST(TEST_BUNDLE_MAP, TEST_BUNDLE_MAP_ADJ) {
  using TestBundle = Bundle<SO3d, R3d>;
  using TestBundleMap = Eigen::Map<TestBundle>;
  using BundleJacobian = TestBundle::Jacobian;

  double data[TestBundleMap::RepSize];
  TestBundleMap b(data);
  b.get<0>() = SO3d(0, 0, 0.6, 0.8);
  b.get<1>() = R3d(Eigen::Vector3d(1, 2, 3));

  for (int i = 0; i < 10; ++i) {
    BundleJacobian bundle_jac = b.adj();

    SO3d::Jacobian so3_jac = b.get<0>().adj();
    R3d::Jacobian r3_jac = b.get<1>().adj();

    BundleJacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bundle_jac, predict_jac);

    b.setRandom();
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
