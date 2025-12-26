cmake_minimum_required(VERSION 3.16)
project(SelfDrivingInference VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Options
option(USE_CUDA "Enable CUDA support" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_JNI "Build JNI bridge" ON)
option(BUILD_TESTS "Build tests" ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Find ONNX Runtime
# =============================================================================
# Set ONNX Runtime path (download from https://github.com/microsoft/onnxruntime/releases)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

if(ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    
    if(WIN32)
        set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    else()
        set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()
else()
    # Try pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ONNXRUNTIME QUIET libonnxruntime)
    endif()
    
    if(NOT ONNXRUNTIME_FOUND)
        message(WARNING "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or install via package manager.")
        message(STATUS "Download from: https://github.com/microsoft/onnxruntime/releases")
    endif()
endif()

# =============================================================================
# CUDA Support
# =============================================================================
if(USE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA support enabled")
        message(STATUS "CUDA version: ${CUDA_VERSION}")
    else()
        message(WARNING "CUDA not found, falling back to CPU")
        set(USE_CUDA OFF)
    endif()
endif()

# =============================================================================
# JNI Support
# =============================================================================
if(BUILD_JNI)
    find_package(JNI REQUIRED)
    if(JNI_FOUND)
        message(STATUS "JNI found")
        message(STATUS "JNI include dirs: ${JNI_INCLUDE_DIRS}")
    else()
        message(WARNING "JNI not found, JNI bridge will not be built")
        set(BUILD_JNI OFF)
    endif()
endif()

# =============================================================================
# Include directories
# =============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

if(BUILD_JNI AND JNI_FOUND)
    include_directories(${JNI_INCLUDE_DIRS})
endif()

# =============================================================================
# Inference Engine Library
# =============================================================================
add_library(selfdriving_engine
    src/inference_engine.cpp
)

target_link_libraries(selfdriving_engine
    ${ONNXRUNTIME_LIBRARIES}
)

if(USE_CUDA)
    target_link_libraries(selfdriving_engine
        ${CUDA_LIBRARIES}
    )
endif()

# =============================================================================
# JNI Bridge Library
# =============================================================================
if(BUILD_JNI AND JNI_FOUND)
    add_library(selfdriving_jni SHARED
        src/jni_bridge.cpp
    )
    
    target_link_libraries(selfdriving_jni
        selfdriving_engine
        ${JNI_LIBRARIES}
    )
    
    # Set output name without lib prefix on Linux
    set_target_properties(selfdriving_jni PROPERTIES
        PREFIX ""
        OUTPUT_NAME "selfdriving_jni"
    )
endif()

# =============================================================================
# Test Executable
# =============================================================================
if(BUILD_TESTS)
    add_executable(test_engine
        src/test_engine.cpp
    )
    
    target_link_libraries(test_engine
        selfdriving_engine
    )
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS selfdriving_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

if(BUILD_JNI AND JNI_FOUND)
    install(TARGETS selfdriving_jni
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== Self-Driving Inference Engine Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA support: ${USE_CUDA}")
message(STATUS "JNI support: ${BUILD_JNI}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARIES}")
message(STATUS "")
