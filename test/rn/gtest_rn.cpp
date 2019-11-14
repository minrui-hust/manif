#include <gtest/gtest.h>

#include "manif/Rn.h"

#include "../common_tester.h"

#include <Eigen/StdVector>

using namespace manif;

// specialize std::vector for 'fixed-size vectorizable' Eigen object
// that are multiple of 32 bytes
// @todo: investigate why only this alignment is troublesome
// especially, SO3 wasn't an issue despite being Eigen::Vector4d too...
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(R4d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(R8d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(R8f)

#ifdef MANIF_COVERAGE_ENABLED

MANIF_TEST(R4d);
MANIF_TEST_JACOBIANS(R4d);

#else

// This is a little too heavy for coverage and not relevant...

MANIF_TEST(R1f);
MANIF_TEST(R2f);
MANIF_TEST(R3f);
MANIF_TEST(R4f);
MANIF_TEST(R5f);
MANIF_TEST(R6f);
MANIF_TEST(R7f);
MANIF_TEST(R8f);
MANIF_TEST(R9f);

MANIF_TEST_JACOBIANS(R1f);
MANIF_TEST_JACOBIANS(R2f);
MANIF_TEST_JACOBIANS(R3f);
MANIF_TEST_JACOBIANS(R4f);
MANIF_TEST_JACOBIANS(R5f);
MANIF_TEST_JACOBIANS(R6f);
MANIF_TEST_JACOBIANS(R7f);
MANIF_TEST_JACOBIANS(R8f);
MANIF_TEST_JACOBIANS(R9f);

MANIF_TEST(R1d);
MANIF_TEST(R2d);
MANIF_TEST(R3d);
MANIF_TEST(R4d);
MANIF_TEST(R5d);
MANIF_TEST(R6d);
MANIF_TEST(R7d);
MANIF_TEST(R8d);
MANIF_TEST(R9d);

MANIF_TEST_JACOBIANS(R1d);
MANIF_TEST_JACOBIANS(R2d);
MANIF_TEST_JACOBIANS(R3d);
MANIF_TEST_JACOBIANS(R4d);
MANIF_TEST_JACOBIANS(R5d);
MANIF_TEST_JACOBIANS(R6d);
MANIF_TEST_JACOBIANS(R7d);
MANIF_TEST_JACOBIANS(R8d);
MANIF_TEST_JACOBIANS(R9d);

#endif

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
