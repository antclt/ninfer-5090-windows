#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
enum class Failure { CompletionEvent, IndexCopy, ConsumerWait };

struct Trace {
    bool active = false, failed = false, pending = false, premature_free = false;
    std::size_t copies = 0;
    Failure failure    = Failure::CompletionEvent;
};

Trace trace;
} // namespace

// Link-time wrappers affect only this test executable. The production path has no fault hooks.
//
// GNU ld --wrap is what makes this work: references to the CUDA entry points are rewritten to
// __wrap_*, and __real_* reaches the genuine function. MSVC has no --wrap, and an import-address
// patch cannot substitute here for the reason tests/artifact/materialization_cuda_errors.cpp
// records: the CUDA runtime resolves statically in this build, so call sites reference the plain
// symbol directly. The injected-failure cases are therefore skipped on MSVC, and reported rather
// than silently skipped -- see expert_cache_cuda_errors below.
#ifndef _MSC_VER
extern "C" {
cudaError_t CUDARTAPI __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t CUDARTAPI __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind,
                                             cudaStream_t);
cudaError_t CUDARTAPI __real_cudaMemcpyBatchAsync(void* const*, const void* const*,
                                                  const std::size_t*, std::size_t,
                                                  cudaMemcpyAttributes*, std::size_t*, std::size_t,
                                                  cudaStream_t);
cudaError_t CUDARTAPI __real_cudaStreamSynchronize(cudaStream_t);
cudaError_t CUDARTAPI __real_cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t CUDARTAPI __real_cudaFree(void*);
cudaError_t CUDARTAPI __real_cudaFreeHost(void*);

cudaError_t CUDARTAPI __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (trace.active && !trace.failed && trace.failure == Failure::CompletionEvent) {
        trace.failed = true;
        return cudaErrorInvalidResourceHandle;
    }
    return __real_cudaEventRecord(event, stream);
}

cudaError_t CUDARTAPI __wrap_cudaMemcpyAsync(void* to, const void* from, std::size_t bytes,
                                             cudaMemcpyKind kind, cudaStream_t stream) {
    const auto result = __real_cudaMemcpyAsync(to, from, bytes, kind, stream);
    if (trace.active && result == cudaSuccess) {
        trace.pending = true;
        ++trace.copies;
    }
    return result;
}

cudaError_t CUDARTAPI __wrap_cudaMemcpyBatchAsync(void* const* to, const void* const* from,
                                                  const std::size_t* bytes, std::size_t count,
                                                  cudaMemcpyAttributes* attrs, std::size_t* indices,
                                                  std::size_t attr_count, cudaStream_t stream) {
    if (trace.active && !trace.failed && trace.failure == Failure::IndexCopy && trace.copies == 1) {
        trace.failed = true;
        return cudaErrorInvalidValue;
    }
    const auto result =
        __real_cudaMemcpyBatchAsync(to, from, bytes, count, attrs, indices, attr_count, stream);
    if (trace.active && result == cudaSuccess) {
        trace.pending = true;
        ++trace.copies;
    }
    return result;
}

cudaError_t CUDARTAPI __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto result = __real_cudaStreamSynchronize(stream);
    if (trace.active && result == cudaSuccess) { trace.pending = false; }
    return result;
}

cudaError_t CUDARTAPI __wrap_cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                 unsigned flags) {
    if (trace.active && !trace.failed && trace.failure == Failure::ConsumerWait) {
        trace.failed = true;
        return cudaErrorInvalidResourceHandle;
    }
    return __real_cudaStreamWaitEvent(stream, event, flags);
}

cudaError_t CUDARTAPI __wrap_cudaFree(void* p) {
    if (trace.active && trace.pending) { trace.premature_free = true; }
    return __real_cudaFree(p);
}

cudaError_t CUDARTAPI __wrap_cudaFreeHost(void* p) {
    if (trace.active && trace.pending) { trace.premature_free = true; }
    return __real_cudaFreeHost(p);
}
}
#endif // !_MSC_VER

namespace ninfer::test {
void expert_cache_cuda_errors(const ExpertBankSource& source, const DeviceContext& device) {
#if defined(_MSC_VER)
    // Link-time injection needs GNU ld --wrap and a dynamically resolved CUDA runtime; neither is
    // available in this build, so the injected cases are not exercised here. Reported rather than
    // silently skipped so the gap stays visible.
    (void)source;
    (void)device;
    std::cerr << "SKIP: CUDA failure injection is unavailable on MSVC (no --wrap, static cudart); "
                 "injected-failure cases not exercised\n";
#else
    const std::array sources{source};
    auto req = describe_expert_cache(sources, {.capacity_bytes = 65536, .max_batch_bytes = 65536});
    for (const auto failure :
         {Failure::CompletionEvent, Failure::IndexCopy, Failure::ConsumerWait}) {
        bool caught = false, unusable = false;
        {
            ExpertCache cache(sources, req, device, device.transfer_view());
            cache.replace({0, req.config.capacity_bytes, 256},
                          std::array{ExpertPlacement{{{0}, 0}, 0}});
            cache.prepare({std::array{ExpertKey{{0}, 0}}});
            trace.active  = true;
            trace.failure = failure;
            try {
                cache.progress();
                if (failure == Failure::ConsumerWait) {
                    (void)cache.acquire(std::array{ExpertKey{{0}, 0}}, device.execution_view());
                }
            } catch (const std::runtime_error&) { caught = true; }
            try {
                cache.progress();
            } catch (const std::runtime_error&) { unusable = true; }
        }
        const auto result = std::exchange(trace, {});
        if (!caught || !unusable || !result.failed || !result.copies || result.pending ||
            result.premature_free) {
            throw std::runtime_error(
                "expert upload failure did not safely drain before destruction");
        }
    }
#endif // !_MSC_VER
}
} // namespace ninfer::test
