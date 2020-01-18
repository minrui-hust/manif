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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
