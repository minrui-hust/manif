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

  TestBundle b;
  TestBundle b_inv;

  b.get<0>() = SO3d(0, 0, 0.6, 0.8);
  b.get<1>() = R3d(Eigen::Vector3d(1, 2, 3));

  SO3d b_so3_inv = b.get<0>().inverse();
  R3d b_r3_inv = b.get<1>().inverse();

  b_inv = b.inverse();
  SO3d b_inv_so3 = b_inv.get<0>();
  R3d b_inv_r3 = b_inv.get<1>();

  EXPECT_EIGEN_NEAR(b_so3_inv.coeffs(), b_inv_so3.coeffs());
  EXPECT_EIGEN_NEAR(b_r3_inv.coeffs(), b_inv_r3.coeffs());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
