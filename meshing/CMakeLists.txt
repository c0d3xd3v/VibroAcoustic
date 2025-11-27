cmake_minimum_required(VERSION 3.8)
project(FloatTetwildWrapper)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(WIN32)
    #find_package(GMP REQUIRED)
    include_directories(${GMP_INCLUDE_DIRS} ${GMP_INCLUDE_DIR})
else()
    set(GMP_LIBRARIES gmp)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0174 NEW)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0169 OLD)

include(FetchContent)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
	add_compile_options(-Wno-error=changes-meaning)
endif()

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
    INSTALL_COMMAND "echo \"eigen c++\""
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    INSTALL_COMMAND "echo \"pybind11\""
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
    ftetwild
    GIT_REPOSITORY https://github.com/c0d3xd3v/fTetWild.git
    GIT_TAG master
    INSTALL_COMMAND "echo \"fTetWild\""
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ftetwild)

message(STATUS "fTetWild source directory: ${ftetwild_SOURCE_DIR}")
include_directories(${ftetwild_SOURCE_DIR}/src)

add_library(${PROJECT_NAME} MODULE
        src/Utils.cpp
        src/Utils.hpp
        src/FloatTetwildWrapper.h
        src/FloatTetwildWrapper.cpp)

target_link_libraries(${PROJECT_NAME}
            PRIVATE
            Eigen3::Eigen
            ${GMP_LIBRARY}
            ${GMPXX_LIBRARY}
            FloatTetwild)

target_include_directories(${PROJECT_NAME} PRIVATE ${ftetwild_BINARY_DIR}/include)

file(GLOB_RECURSE   TETWILD_LIB ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}.so
                                ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}.dll)
file(GLOB_RECURSE   TETWILD_INC ${CMAKE_CURRENT_SOURCE_DIR}/src/${PROJECT_NAME}.h)


add_library(py${PROJECT_NAME} MODULE
        src/Utils.cpp
        src/Utils.hpp
        src/FloatTetwildWrapper.h
        src/FloatTetwildWrapper.cpp
        src/bindings.cpp)

target_link_libraries(py${PROJECT_NAME}
            PRIVATE
            Eigen3::Eigen
            pybind11::module
            Python::Python
            ${GMP_LIBRARY}
            ${GMPXX_LIBRARY}
            FloatTetwild)

target_include_directories(py${PROJECT_NAME} PRIVATE ${ftetwild_BINARY_DIR}/include)
SET_TARGET_PROPERTIES(py${PROJECT_NAME} PROPERTIES PREFIX "")


file(GLOB_RECURSE PYTETWILD_LIB ${CMAKE_CURRENT_BINARY_DIR}/py${PROJECT_NAME}.so
                                ${CMAKE_CURRENT_BINARY_DIR}/py${PROJECT_NAME}.dll
                                ${CMAKE_CURRENT_BINARY_DIR}/py${PROJECT_NAME}.pyd)


install(FILES ${PYTETWILD_LIB} DESTINATION python/py${PROJECT_NAME})
install(FILES ${TETWILD_LIB} DESTINATION lib)
install(FILES ${TETWILD_INC} DESTINATION include)

message(STATUS ${PYTETWILD_LIB})
