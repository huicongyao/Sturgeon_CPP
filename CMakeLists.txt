cmake_minimum_required(VERSION 3.22)

project(Sturgeon)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    message(STATUS "CMAKE_BUILD_TYPE = Debug")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
    message(STATUS "CMAKE_BUILD_TYPE = Release")
endif ()

# 手动指定 Boost 库路径（如果默认路径无效）
set(Boost_USE_STATIC_LIBS OFF)  # 禁用静态库
set(Boost_USE_STATIC_RUNTIME OFF)  # 禁用静态运行时库
set(Boost_INCLUDE_DIR /usr/include)
set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu)
find_package(Boost REQUIRED COMPONENTS system filesystem thread serialization)

if (Boost_FOUND)
    message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "Boost not found!")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_IGNORE_PATH /home/zyserver/miniconda3/lib)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
set(CUDNN_INCLUDE_DIR /usr/local/cuda-12.4/include)
set(CUDNN_LIBRARY /usr/local/cuda-12.4/lib64)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

#set(CMAKE_PREFIX_PATH /home/zyserver/miniconda3/envs/Torch2.0/lib/python3.10/site-packages/torch/share/cmake)
set(CMAKE_PREFIX_PATH /home/zyserver/Downloads/libtorch/share/cmake)
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)
find_package(spdlog REQUIRED)
find_package(CURL REQUIRED)


message(STATUS "Torch path: ${TORCH_LIBRARIES}")

include_directories(
        /usr/include/hdf5/serial
        ${Boost_INCLUDE_DIR}
)

link_directories(
        /usr/lib/x86_64-linux-gnu/hdf5/serial
#        /usr/local/HDF_Group/HDF5/1.14.5/lib
#        /usr/lib/x86_64-linux-gnu/hdf5/openmpi/
)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(${PROJECT_NAME} main.cpp
    DataLoader/DataLoader.cpp
    DataLoader/DataLoader.h
    test2.hpp
        utils.h
    utils.cpp
    3rd_party/thread_pool.hpp
    Model/basecall.cpp
    Model/basecall.h
    CTC_Decoder/CTC_Decoder.cpp
    CTC_Decoder/CTC_Decoder.h
    Model/model.cpp
    Model/model.h
    Stitch_Result/stitch_result.cpp
    Stitch_Result/stitch_result.h
    Fastq_Writer/writer.cpp
    Fastq_Writer/writer.h
        3rd_party/argparse.hpp
)

target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC
        ${TORCH_LIBRARIES}
        spdlog::spdlog_header_only
        ${Boost_LIBRARIES}
        CURL::libcurl
        "libhdf5_cpp.so"
        "libhdf5_hl_cpp.so"
        "libhdf5.so"
        "libhdf5_hl.so"
)

add_executable(
        Sturgeon_sub_loader
        DataLoader/DataLoader_sub_proc.cpp
        DataLoader/DataLoader.cpp
        DataLoader/DataLoader.h
        utils.h
        utils.cpp
        3rd_party/argparse.hpp
)

target_link_libraries(
        Sturgeon_sub_loader
        PUBLIC
        ${TORCH_LIBRARIES}
        spdlog::spdlog_header_only
        CURL::libcurl
        ${Boost_LIBRARIES}
        "libhdf5_cpp.so"
        "libhdf5_hl_cpp.so"
        "libhdf5.so"
        "libhdf5_hl.so"
)

add_executable(
        data_loader_recv
        DataLoader/DataLoader_recv.cpp
        DataLoader/DataLoader.cpp
        DataLoader/DataLoader.h
        utils.h
        utils.cpp
        3rd_party/argparse.hpp
)

target_link_libraries(
        data_loader_recv
        PUBLIC
        ${TORCH_LIBRARIES}
        spdlog::spdlog_header_only
        CURL::libcurl
        ${Boost_LIBRARIES}
        "libhdf5_cpp.so"
        "libhdf5_hl_cpp.so"
        "libhdf5.so"
        "libhdf5_hl.so"
)