find_package(PythonLibs ${USE_PYTHON_VERSION})
find_package(PythonInterp ${USE_PYTHON_VERSION})
list(APPEND DEPENDENCY_LIB ${PYTHON_LIBRARIES})
include_directories(AFTER ${PYTHON_INCLUDE_DIRS})

get_property(INCLUDE_DIRECTORIES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
set(INCLUDE_FLAGS)
foreach(DIR IN LISTS INCLUDE_DIRECTORIES)
	list(APPEND INCLUDE_FLAGS "-I${DIR}")
endforeach()

file(GLOB PYTHON_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

if(NOT GIT_TAG)
	if(GIT_BRANCH STREQUAL "master" OR NOT GIT_BRANCH)
		set(PYLIB_VERSION -b .dev${GIT_REV}+g${GIT_COMMIT_SHORT})
	else()
		set(PYLIB_VERSION -b .dev${GIT_REV}+${GIT_BRANCH}.g${GIT_COMMIT_SHORT})
	endif()
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lib.c
                  COMMAND BINDIR=${CMAKE_CURRENT_BINARY_DIR}/.. CPPFLAGS="${INCLUDE_FLAGS}" ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/_builder.py
                  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/lib.c
                  DEPENDS ${PYTHON_HEADERS}
                  DEPENDS ${BINARY_NAME}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/_builder.py)

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/lib.c PROPERTIES GENERATED ON)

file(GLOB PYTHON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
add_library(${BINARY_NAME}-pylib STATIC ${CMAKE_CURRENT_BINARY_DIR}/lib.c ${PYTHON_SRC})
set_target_properties(${BINARY_NAME}-pylib PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR};${INCLUDE_DIRECTORIES}")
set_target_properties(${BINARY_NAME}-pylib PROPERTIES COMPILE_DEFINITIONS "${OS_DEFINES};${FEATURE_DEFINES};${FUNCTION_DEFINES}")

add_custom_target(${BINARY_NAME}-py ALL
                  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py egg_info -e ${CMAKE_CURRENT_BINARY_DIR} ${PYLIB_VERSION}
                  COMMAND BINDIR=${CMAKE_CURRENT_BINARY_DIR}/.. CPPFLAGS="${INCLUDE_FLAGS}" ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py build -b ${CMAKE_CURRENT_BINARY_DIR}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  DEPENDS ${BINARY_NAME}
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/setup.py
                  DEPENDS ${PYTHON_HEADERS}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/_builder.py
                  DEPENDS ${BINARY_NAME}-pylib)

file(GLOB BASE_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.py)
file(GLOB SUBTESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*/test_*.py)
foreach(TEST IN LISTS BASE_TESTS SUBTESTS)
	if(APPLE)
		set(PATH DYLD_LIBRARY_PATH)
	elseif(WIN32)
		set(PATH PATH)
	else()
		set(PATH LD_LIBRARY_PATH)
	endif()
	string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/(tests/.*/)?test_" "" TEST_NAME "${TEST}")
	string(REPLACE ".py" "" TEST_NAME "${TEST_NAME}")
	add_test(python-${TEST_NAME} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py build -b ${CMAKE_CURRENT_BINARY_DIR} pytest --extras --addopts ${TEST})
	set_tests_properties(python-${TEST_NAME} PROPERTIES ENVIRONMENT "${PATH}=${CMAKE_CURRENT_BINARY_DIR}/..")
endforeach()
