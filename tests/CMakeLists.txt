file(GLOB TESTS *.cpp)
set(TEST_TARGETS)

foreach(test IN LISTS TESTS)
  get_filename_component(TEST_NAME "${test}" NAME_WE)
  add_executable(test-${TEST_NAME} ${test})
  list(APPEND TEST_TARGETS test-${TEST_NAME})
  target_link_libraries(test-${TEST_NAME} retro-base gmock_main)
  add_test(${TEST_NAME} test-${TEST_NAME})
endforeach()

add_custom_target(build-tests DEPENDS ${TEST_TARGETS})
