find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
if(NINFER_ENABLE_DISTRIBUTED)
  find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(NCCL_LIBRARY NAMES nccl REQUIRED)
  add_library(ninfer::nccl UNKNOWN IMPORTED GLOBAL)
  set_target_properties(ninfer::nccl PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
endif()

if(MSVC)
  # Windows: FFmpeg dev binaries are staged into ffmpeg/ (see build_windows.bat).
  # Expose the same target name upstream consumers link, so nothing else changes.
  add_library(PkgConfig::FFMPEG INTERFACE IMPORTED GLOBAL)
  target_include_directories(PkgConfig::FFMPEG INTERFACE
    "${PROJECT_SOURCE_DIR}/ffmpeg/include")
  target_link_directories(PkgConfig::FFMPEG INTERFACE
    "${PROJECT_SOURCE_DIR}/ffmpeg/lib")
  target_link_libraries(PkgConfig::FFMPEG INTERFACE
    avcodec avformat avutil swscale swresample)

  # hwloc is deliberately absent here: upstream's CPU execution (src/core/cpu) is gated to Linux
  # x86-64 by src/core/CMakeLists.txt, and this port has no Windows hwloc. The Windows CPU topology
  # and affinity path is this port's own work, not a build-dependency substitution.
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(HWLOC REQUIRED IMPORTED_TARGET hwloc>=2.5)
  pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing>=2.5)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

if(NINFER_BUILD_ENGINE)
  if(MSVC)
    # Windows: libjpeg-turbo dev binaries are staged into jpeg/ (see build_windows.bat). Expose the
    # same JPEG::JPEG target name upstream consumers link, so nothing else changes.
    add_library(JPEG::JPEG INTERFACE IMPORTED GLOBAL)
    target_include_directories(JPEG::JPEG INTERFACE "${PROJECT_SOURCE_DIR}/jpeg/include")
    target_link_directories(JPEG::JPEG INTERFACE "${PROJECT_SOURCE_DIR}/jpeg/lib")
    target_link_libraries(JPEG::JPEG INTERFACE jpeg)
  else()
    find_package(JPEG REQUIRED)
  endif()
  if(NOT MSVC)
    # On Windows the staged-binaries PkgConfig::FFMPEG above is the whole story: there is no
    # pkg-config here, so the pkg_check_modules call upstream keeps cannot run. Consumers link
    # the same target name either way.
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
      libavformat libavcodec libavutil libswscale)
  endif()
  add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)
  add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)
endif()

if(NINFER_BUILD_PRODUCT_SUPPORT)
  add_library(ninfer::toml INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::toml INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/tomlplusplus)
  target_compile_features(ninfer::toml INTERFACE cxx_std_17)

  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  # Use CMake's FindCURL module, which works on both Linux and Windows and creates
  # the CURL::libcurl imported target. CURL is optional: when absent, remote HTTP media
  # is unsupported at runtime (the existing NINFER_HAVE_LIBCURL guard handles this).
  find_package(CURL 7.85)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
