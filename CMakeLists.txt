cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(ahc003_solver)

find_package(Boost COMPONENTS program_options filesystem)

add_compile_definitions(LOCAL)
add_executable(ahc003_solver main.cpp)

target_include_directories(ahc003_solver PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(ahc003_solver ${Boost_LIBRARIES})
message("Boost_FOUND: ${Boost_FOUND}")
message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message("Boost_LIBRARIES: ${Boost_LIBRARIES}")
