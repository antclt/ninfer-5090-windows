target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/argmax.cu"
  "${CMAKE_CURRENT_LIST_DIR}/target_logprobs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/topk_logprobs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_logprobs_launch.cu"
)
