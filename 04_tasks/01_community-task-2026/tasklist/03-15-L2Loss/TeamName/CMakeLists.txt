cmake_minimum_required(VERSION 3.16)
project(L2Loss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 路径设置
if(NOT DEFINED ENV{ASCEND_HOME_DIR})
    message(FATAL_ERROR "Please set ASCEND_HOME_DIR environment variable")
endif()
set(ASCEND_HOME_DIR $ENV{ASCEND_HOME_DIR})

# 包含目录
include_directories(${ASCEND_HOME_DIR}/ascend-toolkit/latest/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/op_host)

# 源文件
set(OP_API_SOURCES
    op_api/aclnn_l2_loss.cpp
)

set(OP_HOST_SOURCES
    op_host/l2_loss.cpp
)

# 编译选项
add_compile_options(-O2 -Wall -Wextra)

# 生成算子库
add_library(l2_loss SHARED
    ${OP_API_SOURCES}
    ${OP_HOST_SOURCES}
)

target_link_libraries(l2_loss
    ${ASCEND_HOME_DIR}/ascend-toolkit/latest/lib/libascendcl.so
)

# 安装
install(TARGETS l2_loss
    LIBRARY DESTINATION lib
)
