# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninfer_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

foreach(part IN ITEMS system task_pool)
  ninfer_add_test(ninfer_cpu_${part}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_${part}.cpp"
    LIBRARIES ninfer_cpu)
  set_tests_properties(ninfer_cpu_${part}_test PROPERTIES TIMEOUT 30)
endforeach()

if(NINFER_TARGET STREQUAL "rtx5090")
  foreach(part IN ITEMS x86)
    ninfer_add_test(ninfer_cpu_${part}_test
      SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_${part}.cpp"
      LIBRARIES ninfer_cpu_executor)
    set_tests_properties(ninfer_cpu_${part}_test PROPERTIES TIMEOUT 30)
  endforeach()
  # test_executor.cpp locates its worker CPU through sched_getcpu() (<sched.h>), which MSVC does
  # not provide. The executor itself still builds and runs on Windows; only this test's CPU
  # identity check is POSIX, so it is gated the same way as the --wrap cases below.
  if(NOT MSVC)
    ninfer_add_test(ninfer_cpu_executor_test
      SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_executor.cpp"
      LIBRARIES ninfer_cpu_executor)
    set_tests_properties(ninfer_cpu_executor_test PROPERTIES TIMEOUT 30)
  endif()
  ninfer_add_test(ninfer_cpu_tensor_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_tensor.cpp"
    LIBRARIES ninfer_core ninfer_cpu_executor)
endif()

ninfer_add_test(ninfer_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(ninfer_device_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:")
set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test ${flags})
  set_tests_properties(ninfer_device_sync_${mode}_test PROPERTIES
    ENVIRONMENT "NINFER_CUDA_SYNC=${mode}" SKIP_RETURN_CODE 77)
endforeach()
add_test(NAME ninfer_device_sync_invalid_test COMMAND ninfer_device_test --invalid-sync)
set_tests_properties(ninfer_device_sync_invalid_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=invalid")

# The empty case must not go through ctest's own ENVIRONMENT. Measured 2026-09-25 on CMake 4.3.1:
# decorating this test with ENVIRONMENT "NINFER_CUDA_SYNC=" makes the test itself pass and poisons
# every test that follows it -- ninfer_decode_graph_test fails with "NINFER_CUDA_SYNC must be spin,
# blocking, yield, or auto" immediately after it, and passes when the invalid-value test precedes it
# or when it runs alone. Most of the suite binds a device, so 36 tests failed that way. Setting the
# variable inside the command instead, through `cmake -E env`, produces the same set-but-empty value
# the case is about without putting it into ctest's environment.
add_test(NAME ninfer_device_sync_empty_test
  COMMAND ${CMAKE_COMMAND} -E env "NINFER_CUDA_SYNC=" ninfer_device_test --invalid-sync)

ninfer_add_test(ninfer_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_memory_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_memory.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_nf4_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_nf4.cpp"
  LIBRARIES ninfer_core)
add_test(NAME ninfer_nf4_exact_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_nf4_exact.py"
    $<TARGET_FILE:ninfer_nf4_test>)

if(NOT WIN32)
  ninfer_add_test(ninfer_read_queue_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../io/test_read_queue.cpp"
    LIBRARIES ninfer_core PkgConfig::URING)
  target_link_options(ninfer_read_queue_test PRIVATE "LINKER:--wrap=io_uring_submit")
  set_tests_properties(ninfer_read_queue_test PROPERTIES TIMEOUT 30)
endif()

ninfer_add_test(ninfer_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_host_context_arena_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_context_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_kv_cache_test
  ninfer_host_context_arena_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES ninfer_core)

# The 128-bit integer the cost-model arithmetic is written against. Its two branches are selected by
# the compiler rather than by this project -- the builtin on GCC and Clang, the fallback struct on
# MSVC -- so the same assertions are the only thing that can show the fallback agrees with the
# builtin, and the divisor case is the bug the header records as fixed.
ninfer_add_test(ninfer_uint128_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_uint128.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_weight_bank_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_weight_bank.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_cache.cu"
          "${CMAKE_CURRENT_LIST_DIR}/../expert_cache_cuda_errors.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_expert_cache_test PROPERTIES TIMEOUT 60)

# The injected-failure cases need GNU --wrap, which MSVC does not provide. The test source reports
# that as SKIP rather than passing silently; see expert_cache_cuda_errors.cpp, and
# tests/artifact/tests.cmake for the same arrangement on the artifact side.
if(NOT MSVC)
  foreach(symbol IN ITEMS
      cudaEventRecord cudaMemcpyAsync cudaMemcpyBatchAsync cudaStreamSynchronize cudaStreamWaitEvent
      cudaFree cudaFreeHost)
    target_link_options(ninfer_expert_cache_test PRIVATE "LINKER:--wrap=${symbol}")
  endforeach()
endif()
