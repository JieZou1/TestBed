project(OpenCVEx)

#include (GenerateExportHeader)

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake-modules)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#file(GLOB SOURCES "*.cpp")
#file(GLOB HEADERS "*.h")
set (HEADERS 
		${INCLUDE_DIR}/OpenCVEx_Export.h
		${INCLUDE_DIR}/OpenCVEx.h 
	)
set (SOURCES 
		OpenCVEx.cpp
	)
set (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_library(OpenCVEx SHARED ${SOURCES} ${HEADERS})
target_link_libraries(OpenCVEx ${OpenCV_LIBS})

# GENERATE_EXPORT_HEADER( OpenCVEx
             # BASE_NAME OpenCVEx
             # EXPORT_MACRO_NAME OpenCVEx_EXPORT
             # EXPORT_FILE_NAME OpenCVEx_Export.h
             # STATIC_DEFINE OpenCVEx_BUILT_AS_STATIC
# )

install(TARGETS OpenCVEx DESTINATION ${CMAKE_BINARY_DIR}/deploy)
install(FILES ${HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/deploy)
