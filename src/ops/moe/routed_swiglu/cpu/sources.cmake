if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/unavailable.cpp")
  return()
endif()

# Baseline entry/grouping code never inherits worker ISA flags.
set(ninfer_moe_cpu_sources
  "${CMAKE_CURRENT_LIST_DIR}/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q6/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/h2560_i640_e512_k10/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/groupwise_int/launch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/nvfp4/launch.cpp")

foreach(isa IN ITEMS avx2 avx512 avx512_bf16)
  add_library(ninfer_moe_cpu_${isa} OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/groupwise_int/${isa}.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/nvfp4/${isa}.cpp")
  ninfer_internal_includes(ninfer_moe_cpu_${isa})
  set_target_properties(ninfer_moe_cpu_${isa} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(ninfer_moe_cpu_${isa} PRIVATE ninfer_core)
  # Each variant is reached only after select_isa() has confirmed the CPU reports its features, so
  # compiling one for its own ISA is safe. MSVC accepts the -m flags below and ignores them without
  # a word (measured: no D9002, where an option it does not know is reported), so the whole per-variant
  # split would silently collapse into MSVC's default codegen; /arch is what actually applies it.
  if(MSVC)
    if(isa STREQUAL "avx2")
      target_compile_options(ninfer_moe_cpu_${isa} PRIVATE /arch:AVX2)
    else()
      target_compile_options(ninfer_moe_cpu_${isa} PRIVATE /arch:AVX512)
    endif()
  else()
    target_compile_options(ninfer_moe_cpu_${isa} PRIVATE -mavx2 -mfma -mf16c)
    if(NOT isa STREQUAL "avx2")
      target_compile_options(ninfer_moe_cpu_${isa} PRIVATE
        -mavx512f -mavx512bw -mavx512dq -mavx512vl)
    endif()
    if(isa STREQUAL "avx512_bf16")
      target_compile_options(ninfer_moe_cpu_${isa} PRIVATE -mavx512bf16)
    endif()
  endif()
endforeach()

add_library(ninfer_moe_cpu SHARED ${ninfer_moe_cpu_sources})
set_target_properties(ninfer_moe_cpu PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
ninfer_internal_includes(ninfer_moe_cpu)
target_link_libraries(ninfer_moe_cpu PUBLIC ninfer_core PRIVATE ninfer_cpu_executor
  ninfer_moe_cpu_avx2 ninfer_moe_cpu_avx512 ninfer_moe_cpu_avx512_bf16)
target_link_libraries(ninfer_ops PUBLIC ninfer_moe_cpu)

if(WIN32)
  add_custom_command(TARGET ninfer_moe_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/apps"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:ninfer_moe_cpu>
      "${CMAKE_BINARY_DIR}/apps/$<TARGET_FILE_NAME:ninfer_moe_cpu>"
  )
  if(BUILD_TESTING)
    add_custom_command(TARGET ninfer_moe_cpu POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/tests"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ninfer_moe_cpu>
        "${CMAKE_BINARY_DIR}/tests/$<TARGET_FILE_NAME:ninfer_moe_cpu>"
    )
  endif()
endif()
