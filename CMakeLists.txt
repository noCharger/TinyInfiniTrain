cmake_minimum_required(VERSION 3.28)

project(infini_train VERSION 0.3.0 LANGUAGES CXX)

include(CMakeDependentOption)
option(USE_CUDA "Support NVIDIA CUDA" OFF)
cmake_dependent_option(USE_NCCL "Support NCCL collective communication" OFF "USE_CUDA" OFF)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_TESTING "third party tests" OFF)
cmake_dependent_option(BUILD_TEST_CORE "Build tests for core components" ON BUILD_TEST OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add gflags
add_subdirectory(third_party/gflags)
include_directories(${gflags_SOURCE_DIR}/include)

set(WITH_GFLAGS OFF CACHE BOOL "Disable glog finding system gflags" FORCE)
set(WITH_GTEST OFF CACHE BOOL "Disable glog finding system gtest" FORCE)

# Add glog
add_subdirectory(third_party/glog)
include_directories(${glog_SOURCE_DIR}/src)

# Add eigen
if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    include_directories(/opt/homebrew/opt/libomp/include)
endif()
find_package(OpenMP REQUIRED)
if(EXISTS ${PROJECT_SOURCE_DIR}/third_party/eigen/CMakeLists.txt)
    add_subdirectory(third_party/eigen)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
else()
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)
list(FILTER SRC EXCLUDE REGEX ".*kernels/cpu/.*")
# The concrete pipeline schedule calls GPT2 methods, so compile it into the
# GPT2 example library below instead of introducing a base-library cycle.
list(FILTER SRC EXCLUDE REGEX ".*distributed/pipeline_schedule\\.cc$")

file (GLOB_RECURSE CPU_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cpu/*.cc)
add_library(infini_train_cpu_kernels STATIC ${CPU_KERNELS})
target_link_libraries(infini_train_cpu_kernels glog Eigen3::Eigen OpenMP::OpenMP_CXX)

if(USE_CUDA)
    add_compile_definitions(USE_CUDA=1)
    enable_language(CUDA)
    include(FindCUDAToolkit)

    # enable CUDA-related compilation options
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
    endif()
    file(GLOB_RECURSE CUDA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)
    add_library(infini_train_cuda_kernels STATIC ${CUDA_KERNELS})
    set_target_properties(infini_train_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(infini_train_cuda_kernels glog CUDA::cudart CUDA::cublas)

    if(USE_NCCL)
        find_path(NCCL_INCLUDE_DIR nccl.h
            HINTS /usr/local/cuda/include $ENV{NCCL_INCLUDE_DIR})
        find_library(NCCL_LIBRARY nccl
            HINTS /usr/local/lib /usr/lib $ENV{NCCL_LIB_DIR})
        if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
            message(FATAL_ERROR "NCCL not found. Set NCCL_INCLUDE_DIR and NCCL_LIB_DIR.")
        endif()
        message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
        add_library(NCCL::NCCL IMPORTED INTERFACE)
        set_target_properties(NCCL::NCCL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${NCCL_LIBRARY}")
        add_compile_definitions(USE_NCCL=1)

    endif()

    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,infini_train_cpu_kernels>"
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,infini_train_cuda_kernels>")
    if(USE_NCCL)
        target_link_libraries(infini_train NCCL::NCCL CUDA::cudart)
    endif()
else()
    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags OpenMP::OpenMP_CXX
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,infini_train_cpu_kernels>")
endif()

if(BUILD_TEST)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "Do not build gmock" FORCE)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "Do not install gtest" FORCE)
  add_subdirectory(third_party/googletest)
  include_directories(third_party/googletest/googletest/include)
endif()

set(EXAMPLE_GPT2_SOURCES
    example/common/tiny_shakespeare_dataset.cc
    example/common/tokenizer.cc
    example/gpt2/net.cc
)
if(USE_NCCL)
  list(APPEND EXAMPLE_GPT2_SOURCES infini_train/src/distributed/pipeline_schedule.cc)
endif()
add_library(example_gpt2 STATIC ${EXAMPLE_GPT2_SOURCES})
target_link_libraries(example_gpt2 infini_train) 

add_executable(parallel_analysis tools/parallel_analysis.cc)

if(USE_CUDA AND USE_NCCL)
  add_executable(pipeline_benchmark tools/distributed_pipeline_benchmark.cc)
  target_link_libraries(pipeline_benchmark infini_train example_gpt2)
endif()

function(build_test files)
  # Non-recursive glob for skip failed tests
  file(GLOB TEST_SOURCES ${files})
  foreach(testsourcefile ${TEST_SOURCES})
    get_filename_component(testname ${testsourcefile} NAME_WE)
    add_executable(${testname} ${testsourcefile})
    target_link_libraries(${testname} infini_train example_gpt2 GTest::gtest_main)
    add_test(NAME ${testname} COMMAND ${testname})
  endforeach(testsourcefile ${TEST_SOURCES})
endfunction()

if(BUILD_TEST)
  add_compile_definitions(BUILD_TEST=1)
  enable_testing()
  add_test(NAME parallel_analysis COMMAND parallel_analysis)
  if(BUILD_TEST_CORE)
    build_test(test/autograd/test_checkpoint.cc)
    build_test(test/autograd/test_elementwise.cc)
    build_test(test/kernels/test_matmul.cc)
    build_test(test/kernels/test_mixed_precision.cc)
    build_test(test/kernels/test_transformer.cc)
    build_test(test/kernels/test_dispatcher.cc)
    build_test(test/tensor/test_tensor.cc)
    build_test(test/optimizer/test_adam.cc)
    build_test(test/example/test_gpt2.cc)
    build_test(test/example/test_gpt2_tiny.cc)
    build_test(test/example/test_gpt2_parity.cc)
    if(USE_CUDA)
      build_test(test/kernels/test_matmul_cuda.cc)
      build_test(test/kernels/test_transformer_cuda.cc)
      build_test(test/optimizer/test_adam_cuda.cc)
      if(USE_NCCL)
        build_test(test/distributed/test_distributed_nccl.cc)
        # This binary needs rank/world-size/rendezvous variables and multiple
        # concurrent processes. Build it here, but run it only through
        # tools/launch_distributed_tests.py rather than a single-process CTest.
        set_tests_properties(test_distributed_nccl PROPERTIES DISABLED TRUE)
      endif()
    endif()
  endif()
endif()
