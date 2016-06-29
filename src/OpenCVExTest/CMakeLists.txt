project(OpenCVExTest)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#file(GLOB SOURCES "*.cpp")
#file(GLOB HEADERS "*.h")
set (HEADERS 
		${INCLUDE_DIR}/OpenCVEx_Export.h
		${INCLUDE_DIR}/OpenCVEx.h 
	)
set (SOURCES 
		OpenCVExTest.cpp
	)

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(OpenCVExTest ${SOURCES} ${HEADERS})
target_link_libraries(OpenCVExTest OpenCVEx)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

MESSAGE("OpenCV_DIR: ${OpenCV_DIR}")
MESSAGE("OpenCV_LIBS: ${OpenCV_LIBS}")
MESSAGE("OpenCV_LIB_COMPONENTS: ${OpenCV_LIB_COMPONENTS}")

# add_custom_command(TARGET OpenCVExTest POST_BUILD
    # COMMAND ${CMAKE_COMMAND} -E copy_if_different
	# ${OpenCV_DIR}/../bin/opencv_world310d.dll
	# "${CMAKE_BINARY_DIR}/bin/$<CONFIGURATION>/")

install(TARGETS OpenCVExTest DESTINATION ${CMAKE_BINARY_DIR}/deploy)