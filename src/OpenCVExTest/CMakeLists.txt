project(OpenCVExTest)

#file(GLOB SOURCES "*.cpp")
#file(GLOB HEADERS "*.h")
set (HEADERS 
		${INCLUDE_DIR}/OpenCVEx_Export.h
		${INCLUDE_DIR}/Student.h 
	)
set (SOURCES 
		mainapp.cpp
	)

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
add_executable(OpenCVExTest ${SOURCES} ${HEADERS})
target_link_libraries(OpenCVExTest OpenCVEx)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

#find_package(OpenCV REQUIRED )
#include_directories(${OpenCV_INCLUDE_DIR})
#target_link_libraries(OpenCVExTest, ${OpenCV_LIBS})


install(TARGETS OpenCVExTest DESTINATION ${CMAKE_BINARY_DIR}/deploy)