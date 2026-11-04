cmake_minimum_required(VERSION 3.16)
project(vizion_core C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(VIZION_ARCH x86)
else()
    set(VIZION_ARCH x64)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/${VIZION_ARCH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/${VIZION_ARCH})

add_library(vizion_core SHARED
    src/cpu.c
    src/threshold.c
    src/threshold_scalar.c
    src/threshold_sse2.c
    src/threshold_avx2.c
    src/region_runs.c
    src/gray_morph.c
    src/gray_rect_sse2.c
    src/gray_rect_avx2.c
    src/region_morph.c
    src/filter.c
    src/filter_mean_sse2.c
    src/filter_mean_avx2.c
    src/score.c
    src/score_sse2.c
    src/score_avx2.c
    src/measure.c
    src/measure_sample_sse2.c
    src/measure_sample_avx2.c
    src/edge.c
    src/edge_sep_sse2.c
    src/edge_sep_avx2.c
    src/remap_sse2.c
    src/remap_avx2.c
    src/sobel.c
    src/sobel_sse2.c
    src/sobel_avx2.c
    src/pyrdown.c
    src/pyrdown_sse2.c
    src/pyrdown_avx2.c
    src/bgr.c
    src/bgr_sse2.c
    src/bgr_avx2.c
    src/arith.c
    src/arith_sse2.c
    src/tone.c
    src/map.c
    src/map_sse2.c
    src/local.c
    src/intensity.c
    src/intensity_sse2.c
    src/intensity_avx2.c
    src/rotate.c
    src/rotate_sse2.c
    src/rotate_avx2.c
    src/edge_link.c
    src/fit.c
    src/pose.c)
target_include_directories(vizion_core PUBLIC include PRIVATE src)
target_compile_definitions(vizion_core PRIVATE VIZION_BUILD)

find_package(Threads REQUIRED)
target_link_libraries(vizion_core PUBLIC Threads::Threads m)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vizion_core PRIVATE -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter)
    set_source_files_properties(src/threshold_scalar.c src/region_runs.c src/gray_morph.c src/region_morph.c src/filter.c src/score.c src/measure.c PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
    set_source_files_properties(src/edge.c src/sobel.c src/pyrdown.c src/bgr.c src/intensity.c src/rotate.c src/edge_link.c src/fit.c src/pose.c src/arith.c src/tone.c src/map.c src/local.c PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-ffp-contract=off")
    set_source_files_properties(src/threshold_sse2.c src/gray_rect_sse2.c src/filter_mean_sse2.c src/score_sse2.c src/measure_sample_sse2.c PROPERTIES COMPILE_OPTIONS "-msse2")
    set_source_files_properties(src/edge_sep_sse2.c src/remap_sse2.c src/sobel_sse2.c src/pyrdown_sse2.c src/bgr_sse2.c src/intensity_sse2.c src/rotate_sse2.c src/arith_sse2.c src/map_sse2.c PROPERTIES COMPILE_OPTIONS "-msse2;-ffp-contract=off")
    set_source_files_properties(src/threshold_avx2.c src/gray_rect_avx2.c src/filter_mean_avx2.c src/score_avx2.c src/measure_sample_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(src/edge_sep_avx2.c src/remap_avx2.c src/sobel_avx2.c src/pyrdown_avx2.c src/bgr_avx2.c src/intensity_avx2.c src/rotate_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
endif()

add_executable(vizion_core_check src/check_threshold.c)
target_link_libraries(vizion_core_check PRIVATE vizion_core)
if(UNIX)
    set_target_properties(vizion_core_check PROPERTIES BUILD_RPATH "\$ORIGIN")
endif()
