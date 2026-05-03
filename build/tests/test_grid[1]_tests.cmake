add_test([=[TestBuildGrid.ShapeDefault]=]  /home/sebas/Desktop/ie_dev/cuKan/build/tests/test_grid [==[--gtest_filter=TestBuildGrid.ShapeDefault]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TestBuildGrid.ShapeDefault]=]  PROPERTIES WORKING_DIRECTORY /home/sebas/Desktop/ie_dev/cuKan/build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_grid_TESTS TestBuildGrid.ShapeDefault)
