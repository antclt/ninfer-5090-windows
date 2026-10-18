# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops SHARED)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_library(ninfer_ops)
# Export all symbols so consumers of this shared library (via ninfer_engine, now PUBLIC) can
# resolve the nvfp4 TMA kernel launchers and other Op entry points that live in the
# ninfer_ops_non_rdc static archive. Without this, MSVC does not export DLL symbols by default
# and ninfer-serve.exe fails to link with 16 unresolved externals.
set_target_properties(ninfer_ops PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(ninfer_ops PUBLIC ninfer_core ninfer_ops_non_rdc)

if(WIN32)
  # On Windows, copy ninfer_ops.dll beside executables so loaders locate it.
  add_custom_command(TARGET ninfer_ops POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/apps"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:ninfer_ops>
      "${CMAKE_BINARY_DIR}/apps/$<TARGET_FILE_NAME:ninfer_ops>"
  )
  if(BUILD_TESTING)
    add_custom_command(TARGET ninfer_ops POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/tests"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ninfer_ops>
        "${CMAKE_BINARY_DIR}/tests/$<TARGET_FILE_NAME:ninfer_ops>"
    )
  endif()
endif()

# Non-RDC kernels keep instruction and register-allocation choices within one translation unit.
# This private PIC archive joins the shared library through host launchers.
add_library(ninfer_ops_non_rdc STATIC)
ninfer_internal_includes(ninfer_ops_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_ops_non_rdc)
target_link_libraries(ninfer_ops_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

target_sources(ninfer_ops PRIVATE weight_input.cpp)

# Perturbation seam, scoped to the test tree. An Op's kernel may carry small mutation branches behind
# NINFER_OP_TEST_MUTATIONS, selected per test run by the launcher from NINFER_OP_MUTATION, and
# tools/release/check_test_mutation.py asserts the suite actually goes red when one is selected.
#
# The scope is the whole point. `build` and `build-test` are separate configurations and BUILD_TESTING
# is OFF in the former, so the define -- and the branches it guards -- exist only in a tree that also
# builds tests. A shipping binary compiles the unmutated code, so the seam cannot be reached in a
# release. PUBLIC rather than PRIVATE so the test translation units can detect that the binary they
# are linked against supports mutations at all; a build without the seam must fail the gate loudly
# rather than pass it by never going red.
if(BUILD_TESTING)
  target_compile_definitions(ninfer_ops PUBLIC NINFER_OP_TEST_MUTATIONS=1)
endif()
include("${CMAKE_CURRENT_LIST_DIR}/common/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/elementwise/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/moe/sources.cmake")
