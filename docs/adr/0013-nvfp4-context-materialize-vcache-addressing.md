# ADR-0013: V-cache stride addressing in NVFP4 context materialize and runtime DLL staging

**Status:** accepted (fixed in `ec17f64e`)

## Context

When running DFlash2 speculative decoding lanes (e.g. `start_quasar_v3_dflash2_vision.bat`), OpenCode Desktop received garbled, repetitive output (e.g., repeating `<think><think><think>` and `WeWe`). In server request logs, DFlash2 draft token acceptance plummeted from the recorded 45%–55% down to ~0.4% (e.g. 1/240 accepted), forcing fallback to single-token AR decoding from a poisoned KV context cache.

When switched to non-speculative or MTP lanes (such as `start_quasar_v3_mtp4_vision.bat`), the engine generated coherent, natural English.

## Defect 1: Broken V-Cache Stride Indexing in `materialize_nvfp4.cu`

In [`src/ops/kv_cache/context_materialize/materialize_nvfp4.cu`](../../src/ops/kv_cache/context_materialize/materialize_nvfp4.cu), the store logic for `layer.cache_v` previously used a handcrafted index formula:

```cuda
const auto dst = row % 128 + 128LL * ((positions[column] & 2047) +
                                      (long long)layer.padded_capacity *
                                          (row / 128 + 8 * slots[request]));
layer.cache_v[dst] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(result)));
```

### Why this failed

1. **Ignored `value_slot_stride`**: The layout of slotted multi-head KV caches does not follow flat arithmetic (`8 * slots[request]`). The true slot offset is governed by `layer.value_slot_stride * slot`.
2. **Ignored `cyclic_offset` mapping**: In `epilogue.cuh`, `store_value()` correctly resolves cyclic layout offsets:
   ```cuda
   const auto dst = layer.value_slot_stride * slot +
                    kv::cyclic_offset<128, 2048>(positions[column], row / 128,
                                                 layer.padded_capacity, row % 128);
   layer.cache_v[dst] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(result)));
   ```
3. During prompt prefill, the V-cache was populated at corrupted memory addresses. As a result, every speculative draft round read nonsense values, leading to complete draft rejection and degenerate token loops.

### Resolution

Replaced the ad-hoc store calculation in `materialize_nvfp4.cu` with `store_value(row, column, slots[request], positions, layer, result);` from [`epilogue.cuh`](../../src/ops/kv_cache/context_materialize/epilogue.cuh).

## Defect 2: Stale Windows DLL Shadowing in `build\apps\`

Even after `materialize_nvfp4.cu` was fixed in `src/ops/`, launching `start_quasar_v3_dflash2_vision.bat` continued to produce garbled text.

### Root cause

- On Windows, CMake puts shared libraries (`ninfer_ops.dll`, `ninfer_moe_cpu.dll`) into `${CMAKE_BINARY_DIR}/src/ops/`.
- Executables (`ninfer-serve.exe`, `ninfer.exe`) live in `${CMAKE_BINARY_DIR}/apps/`.
- Windows dynamic link library search order searches the application directory (`build\apps\`) **first**.
- A stale copy of `ninfer_ops.dll` from an earlier build sat in `build\apps\`. When `ninfer-serve.exe` started, Windows loaded this stale DLL rather than the recompiled DLL in `build\src\ops\`.

### Resolution

Added CMake `POST_BUILD` copy commands in [`src/ops/CMakeLists.txt`](../../src/ops/CMakeLists.txt) and [`src/ops/moe/routed_swiglu/cpu/sources.cmake`](../../src/ops/moe/routed_swiglu/cpu/sources.cmake):

```cmake
if(WIN32)
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
```

Every build now automatically guarantees that runtime DLLs beside executables in `build\apps\` and `build-test\tests\` remain bit-identical to the newly compiled library.

## Verification Evidence

1. **Unit tests (`build-test/`)**:
   - `ninfer_context_kv_materialize_test.exe`: 94/94 cases passed (100%).
   - `ninfer_speculative_round_test.exe`: PASS (isolation, Chi-squared distribution match).
   - `ninfer_sampling_test.exe`: 391/391 cases passed (100%).
   - `ninfer_dflash2_nvfp4_routes_test.exe`: PASS.
   - `ninfer_qwen3_5_dflash2_real_test.exe 7 1 1 1 k8v4 1 1`: `ok K=7 B=1 graph=1 optimized=1 accepted=20/20`.

2. **End-to-End Live Server (`ninfer-serve.exe` on port 8086)**:
   - **Request 1 (math reasoning)**: Natural English reasoning + proof; **44.9% DFlash2 acceptance** (151/336 accepted), **291.7 tok/s**.
   - **Request 2 (factual QA)**: Clean answer; **73.8% DFlash2 acceptance** (31/42 accepted), **417.5 tok/s**, total latency **141 ms**.
