cmake_minimum_required(VERSION 3.1)
project(micro_graph_optimizer)
set (CMAKE_CXX_STANDARD 11)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

option(ENABLE_LOGGING "Enable logging." ON )
if(ENABLE_LOGGING)
add_definitions(-DMGO_ENABLE_LOG)
endif()

option(ENABLE_ASSERTIONS "Enable assertions." ON )
if(ENABLE_ASSERTIONS)
add_definitions(-DMGO_ENABLE_ASSERT)
endif()

find_package(Eigen3)

include_directories( 
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}
    )
    
add_library(${PROJECT_NAME}
    "micro_graph_optimizer.cpp" "micro_graph_optimizer.h"
    )

add_executable(hello_slam
    "examples/hello_slam.cpp"
    )

target_link_libraries(hello_slam
    ${PROJECT_NAME}
    )

add_executable(slam2d
    examples/slam2d.cpp
    )

target_link_libraries(slam2d
    ${PROJECT_NAME}
    )