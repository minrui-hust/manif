#include "manif/Bundle.h"
#include "manif/Rn.h"
#include "manif/SE2.h"
#include "manif/SE3.h"
#include "manif/SO2.h"
#include "manif/SO3.h"

#include "../common_tester.h"

using namespace manif;

TEST(TEST_BUNDLE, TEST_BUNDLE_PROPS) {
  using TestBundle = Bundle<SO3d, R3d>;

  EXPECT_EQ(6, TestBundle::Dim);
  EXPECT_EQ(6, TestBundle::DoF);
  EXPECT_EQ(7, TestBundle::RepSize);
}

TEST(TEST_BUNDLE, TEST_BUNDLE_INVERSE) {
  using TestBundle = Bundle<SO3d, R3d>;
  using BundleJacobian = TestBundle::Jacobian;

  TestBundle b(SO3d(0, 0, 0.6, 0.8), R3d(Eigen::Vector3d(1, 2, 3)));

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

    b = TestBundle(SO3d::Random(), R3d::Random());
  }
}

TEST(TEST_BUNDLE, TEST_BUNDLE_COMPOSE) {
  using TestBundle = Bundle<SO3d, R3d>;
  using BundleJacobian = TestBundle::Jacobian;
  using SO3Jacobian = SO3d::Jacobian;
  using R3Jacobian = R3d::Jacobian;

  TestBundle a(SO3d(0, 0, 0.6, 0.8), R3d(Eigen::Vector3d(1, 2, 3)));
  TestBundle b(SO3d(0, 0.6, 0, 0.8), R3d(Eigen::Vector3d(4, 5, 6)));

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

    a = TestBundle(SO3d::Random(), R3d::Random());
    b = TestBundle(SO3d::Random(), R3d::Random());
  }
}

TEST(TEST_BUNDLE, TEST_BUNDLE_ACT) {
  using TestBundle = Bundle<SO2d, R2d>;
  using BundleVector = TestBundle::Vector;
  using SO2Vector = SO2d::Vector;
  using R2Vector = R2d::Vector;
  using JacobianBundle = Eigen::Matrix<double, 4, 3>;
  using JacobianBundleVector = Eigen::Matrix<double, 4, 4>;
  using JacobianSO2 = Eigen::Matrix<double, 2, 1>;
  using JacobianSO2Vector = Eigen::Matrix<double, 2, 2>;
  using JacobianR2 = Eigen::Matrix<double, 2, 2>;
  using JacobianR2Vector = Eigen::Matrix<double, 2, 2>;

  TestBundle b(SO2d(0.6, 0.8), R2d(Eigen::Vector2d(1, 2)));
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

    b = TestBundle(SO2d::Random(), R2d::Random());
    v = Eigen::Matrix<double, 4, 1>::Random();
  }
}

TEST(TEST_BUNDLE, TEST_BUNDLE_ADJ) {
  using TestBundle = Bundle<SO3d, R3d>;
  using BundleJacobian = TestBundle::Jacobian;

  TestBundle b(SO3d(0, 0, 0.6, 0.8), R3d(Eigen::Vector3d(1, 2, 3)));

  for (int i = 0; i < 10; ++i) {
    BundleJacobian bundle_jac = b.adj();

    SO3d::Jacobian so3_jac = b.get<0>().adj();
    R3d::Jacobian r3_jac = b.get<1>().adj();

    BundleJacobian predict_jac;
    predict_jac.setZero();
    predict_jac.block<3, 3>(0, 0) = so3_jac;
    predict_jac.block<3, 3>(3, 3) = r3_jac;

    EXPECT_EIGEN_NEAR(bundle_jac, predict_jac);

    b = TestBundle(SO3d::Random(), R3d::Random());
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
