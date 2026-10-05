cmake_minimum_required(VERSION 3.16)
project(mpc_gp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(usv_dynamic_model src/usv_dynamic_model.cpp)
target_include_directories(usv_dynamic_model PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(usv_dynamic_model PUBLIC Eigen3::Eigen)
target_compile_options(usv_dynamic_model PRIVATE -Wall -Wextra)

add_executable(usv_demo src/main.cpp)
target_link_libraries(usv_demo PRIVATE usv_dynamic_model)
