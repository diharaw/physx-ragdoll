cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_program(CLANG_FORMAT_EXE NAMES "clang-format" DOC "Path to clang-format executable")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

file(GLOB_RECURSE PHYSX_DEBUG_LIBS ${PROJECT_SOURCE_DIR}/lib/Debug/external/*.lib)
file(GLOB_RECURSE PHYSX_RELEASE_LIBS ${PROJECT_SOURCE_DIR}/lib/Release/external/*.lib)

file(GLOB_RECURSE PHYSX_RAGDOLL_SOURCES ${PROJECT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE PHYSX_RAGDOLL_HEADERS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SHADER_SOURCES ${PROJECT_SOURCE_DIR}/src/*.glsl)
set(PHYSX_RAGDOLL_ALL_SOURCES ${PHYSX_RAGDOLL_HEADERS} ${PHYSX_RAGDOLL_SOURCES})

if (APPLE)
    add_executable(PhysXRagdoll MACOSX_BUNDLE ${PHYSX_RAGDOLL_ALL_SOURCES} ${SHADER_SOURCES} ${ASSET_SOURCES})
    set(MACOSX_BUNDLE_BUNDLE_NAME "PhysXRagdoll") 
    set_source_files_properties(${SHADER_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/shader)
    set_source_files_properties(${ASSET_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
    add_executable(PhysXRagdoll ${PHYSX_RAGDOLL_ALL_SOURCES}) 
endif()

target_link_libraries(PhysXRagdoll dwSampleFramework)
target_link_libraries(PhysXRagdoll debug "${PHYSX_DEBUG_LIBS}" )
target_link_libraries(PhysXRagdoll optimized "${PHYSX_RELEASE_LIBS}") 

if (NOT APPLE)
    add_custom_command(TARGET PhysXRagdoll POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:PhysXRagdoll>/shader)
endif()

if(CLANG_FORMAT_EXE)
    add_custom_target(PhysXRagdoll-clang-format COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${PHYSX_RAGDOLL_ALL_SOURCES} ${SHADER_SOURCES})
endif()

set_property(TARGET PhysXRagdoll PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$(Configuration)")