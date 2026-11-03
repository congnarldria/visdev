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
    src/region_morph.c)
target_include_directories(vizion_core PUBLIC include PRIVATE src)
target_compile_definitions(vizion_core PRIVATE VIZION_BUILD)

find_package(Threads REQUIRED)
target_link_libraries(vizion_core PUBLIC Threads::Threads m)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vizion_core PRIVATE -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter)
    set_source_files_properties(src/threshold_scalar.c src/region_runs.c src/gray_morph.c src/region_morph.c PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
    set_source_files_properties(src/threshold_sse2.c src/gray_rect_sse2.c PROPERTIES COMPILE_OPTIONS "-msse2")
    set_source_files_properties(src/threshold_avx2.c src/gray_rect_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

add_executable(vizion_core_check src/check_threshold.c)
target_link_libraries(vizion_core_check PRIVATE vizion_core)
if(UNIX)
    set_target_properties(vizion_core_check PROPERTIES BUILD_RPATH "\$ORIGIN")
endif()
