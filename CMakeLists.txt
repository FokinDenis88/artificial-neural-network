# CMakeList.txt : CMake project for artificial-neural-networks, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8)

# Can be different with target
project("artificial-neural-networks" VERSION 1.0)

set(SOURCES
    src/artificial-neural-network.cpp
    src/multilayer-perceptron.cpp
    src/node-mlp.cpp
    #src/networkss/main_network.cpp
    #src/networkss/zero_classifier.cpp
    )

set(INCLUDES
    include
    external/eigen-3.4
    external/file-stream-wrapper/include
    external/regex/include
    )

set(HEADER_FILES_FILTER
    include/activation-functions.hpp
    include/derivatives-activation-functions.hpp
    include/derivatives-loss-function.hpp
    include/errors-mlp.hpp
    include/loss-function.hpp
    include/multilayer-perceptron.hpp
    include/node-mlp.hpp
    )
set(DOC_FILTER
    doc/to-do.txt
    doc/to-do-old.txt
    doc/copy-past-helper.txt
    doc/!wiki.odt
    )
set(NETWORKS_FILTER
    include/networks/binary-or.hpp
    include/networks/main_network.hpp
    include/networks/sample_network.hpp
    include/networks/students_mark.hpp
    include/networks/zero_classifier.hpp
    )

find_package(OpenMP REQUIRED)

# Add source to this project's executable.
add_executable(artificial-neural-networks ${SOURCES}
                ${HEADER_FILES_FILTER} ${DOC_FILTER} ${NETWORKS_FILTER})
# Include Directories
target_include_directories(artificial-neural-networks PUBLIC ${INCLUDES})
target_link_libraries(artificial-neural-networks PRIVATE OpenMP::OpenMP_CXX)

# Filters for Grouping files in project
source_group("Header Files" FILES ${HEADER_FILES_FILTER})
source_group("Doc" FILES ${DOC_FILTER})
source_group("Networks" FILES ${NETWORKS_FILTER})


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET artificial-neural-networks PROPERTY CXX_STANDARD 20)
  set_property(TARGET artificial-neural-networks PROPERTY C_STANDARD 17)
endif()
set_property(TARGET artificial-neural-networks PROPERTY SUBSYSTEM CONSOLE)





# Help Information
# TODO: Add tests and install targets if needed.


# Добавляет директории для дальнейшей обработки. В них нужно добавлять еще одни CMakeLists.txt файлы
# add_subdirectory (Hello)

#add_library (Hello hello.cxx)
#target_include_directories (Hello PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#add_executable (helloDemo demo.cxx demo_b.cxx)
#target_link_libraries (helloDemo LINK_PUBLIC Hello)

#set(SOURCE_EXE main.cpp)   set(SOURCE_LIB foo.cpp)
#add_library(foo STATIC ${SOURCE_LIB})	# Создание статической библиотеки с именем foo
#add_executable(main ${SOURCE_EXE})	# Создает исполняемый файл с именем main
#target_link_libraries(main foo)
#include_directories(foo)			# Расположение заголовочных файлов
#file(GLOB <variable> [LIST_DIRECTORIES true|false] [RELATIVE <path>] [CONFIGURE_DEPENDS] [<globbing-expressions>...])
#target_compile_options(first-test PRIVATE -fexceptions)

#find_package(OpenMP)
#if (OPENMP_FOUND)
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

#set_property(TARGET artificial-neural-networks PROPERTY SUBSYSTEM WINDOWS)
#CMAKE_CXX_FLAGS    /SUBSYSTEM:CONSOLE

#source_group("Header Files" FILES ${INCLUDES} REGULAR_EXPRESSION *)

#OpenMP
#find_package(OpenMP REQUIRED)
#target_link_libraries(artificial-neural-networks PRIVATE OpenMP::OpenMP_CXX)