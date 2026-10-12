target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/prepare.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_dynamic_grouped_conv_prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/add/add.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/add/launch.cu")
