# The CPU execution layer is x86-64 only, and upstream implements it against hwloc on Linux. The
# port supplies a native Windows implementation of the same core/cpu/system.h interface in
# cpu/system_windows.cpp (Win32 processor groups and MSVC intrinsics), so the architecture ceiling
# stands while the platform one widens. CMAKE_SYSTEM_PROCESSOR is empty under some Windows
# generators, so an 8-byte pointer stands in for it when it is.
set(NINFER_CPU_X86_64 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(NINFER_CPU_X86_64 TRUE)
elseif(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(NINFER_CPU_X86_64 TRUE)
endif()
if(NOT NINFER_CPU_X86_64)
  message(FATAL_ERROR "NInfer CPU execution currently requires x86-64")
endif()

if(WIN32)
  set(NINFER_CPU_SYSTEM_SOURCE cpu/system_windows.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(NINFER_CPU_SYSTEM_SOURCE cpu/system.cpp)
else()
  message(FATAL_ERROR "NInfer CPU execution requires Linux x86-64 or Windows x86-64")
endif()

add_library(ninfer_cpu STATIC
  ${NINFER_CPU_SYSTEM_SOURCE}
  cpu/task_pool.cpp
  cpu/executor.cpp
)
ninfer_internal_includes(ninfer_cpu)
if(WIN32)
  # hwloc backs upstream's Linux implementation only; the Windows path needs no external library.
  target_link_libraries(ninfer_cpu PRIVATE Threads::Threads)
else()
  target_link_libraries(ninfer_cpu PRIVATE PkgConfig::HWLOC Threads::Threads)
endif()

add_library(ninfer_core STATIC
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  dtype.cpp
  nf4.cpp
  io/file.cpp
  io/read_queue.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  weight_bank.cpp
  expert_cache.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
if(NOT WIN32)
  target_link_libraries(ninfer_core PRIVATE PkgConfig::URING)
endif()
