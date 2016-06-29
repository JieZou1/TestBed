project(OpenCVEx)

#include (GenerateExportHeader)

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake-modules)

#find_package(OpenCV REQUIRED )

#file(GLOB SOURCES "*.cpp")
#file(GLOB HEADERS "*.h")
set (HEADERS 
		${INCLUDE_DIR}/OpenCVEx_Export.h
		${INCLUDE_DIR}/Student.h 
	)
set (SOURCES 
		student.cpp
	)
set (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
add_library(OpenCVEx SHARED ${SOURCES} ${HEADERS})

#add_library(OpenCVEx SHARED ${SOURCES})
# GENERATE_EXPORT_HEADER( OpenCVEx
             # BASE_NAME OpenCVEx
             # EXPORT_MACRO_NAME OpenCVEx_EXPORT
             # EXPORT_FILE_NAME OpenCVEx_Export.h
             # STATIC_DEFINE OpenCVEx_BUILT_AS_STATIC
# )

# add_custom_command(TARGET OpenCVEx POST_BUILD
    # COMMAND ${CMAKE_COMMAND} -E copy_if_different
	# "$<TARGET_FILE:OpenCVEx>"
	#"${CMAKE_BINARY_DIR}/$<CONFIGURATION>/")

#install(TARGETS OpenCVEx DESTINATION ${CMAKE_BINARY_DIR}/src/OpenCVEx/)

install(TARGETS OpenCVEx DESTINATION ${CMAKE_BINARY_DIR}/deploy)
install(FILES ${HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/deploy)
