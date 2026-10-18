add_library(ninfer_op_test_support STATIC "${CMAKE_CURRENT_LIST_DIR}/suite.cpp")
ninfer_test_includes(ninfer_op_test_support)
ninfer_op_oracle_options(ninfer_op_test_support)
target_link_libraries(ninfer_op_test_support PUBLIC ninfer_cpu CUDA::cudart)

include("${CMAKE_CURRENT_LIST_DIR}/elementwise/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/tests.cmake")

# The port's DFlash2 NVFP4 route coverage. The source is this port's own -- upstream
# does not have it -- and the 2026-10-17 merge left it registered nowhere, so the test
# could not run. Restored from f89e05fc.
ninfer_add_op_test(ninfer_dflash2_nvfp4_routes_test
  SOURCES ops/test_dflash2_nvfp4_routes.cpp
  LIBRARIES ninfer_ops)

# The port's context-KV materialize and speculative-round coverage. Upstream deleted
# both files in the 2026-10-17 merge, taking the port's batch-splitting sweep and the
# sparse-accept isolation test with them; restore them so the coverage isn't lost
# silently. Restored from f89e05fc.
set(ninfer_context_kv_materialize_batch_groups "8" "1" "2,3,4" "5,6,7")
add_executable(ninfer_context_kv_materialize_test
  "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp")
ninfer_test_includes(ninfer_context_kv_materialize_test)
target_link_libraries(ninfer_context_kv_materialize_test PRIVATE ninfer_ops)
ninfer_op_oracle_options(ninfer_context_kv_materialize_test)
foreach(batch_group IN LISTS ninfer_context_kv_materialize_batch_groups)
  string(REPLACE "," "_" batch_suffix "${batch_group}")
  add_test(NAME "ninfer_context_kv_materialize_test_b${batch_suffix}"
    COMMAND ninfer_context_kv_materialize_test --batches "${batch_group}")
  set_tests_properties("ninfer_context_kv_materialize_test_b${batch_suffix}"
    PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/moe/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/tests.cmake")
