# Bundle tests

# Bundle tests
manif_add_gtest(gtest_bundle gtest_bundle.cpp)

# Bundle Eigen::Map tests
manif_add_gtest(gtest_bundle_map gtest_bundle_map.cpp)

# Bundle tangent tests
manif_add_gtest(gtest_bundle_tangent gtest_bundle_tangent.cpp)

# Bundle tangent Eigen::Map tests
manif_add_gtest(gtest_bundle_tangent_map gtest_bundle_tangent_map.cpp)

set(CXX_11_TEST_TARGETS

  ${CXX_11_TEST_TARGETS}

  # Bundle
  gtest_bundle
  gtest_bundle_map
  gtest_bundle_tangent
  gtest_bundle_tangent_map

  PARENT_SCOPE
)
