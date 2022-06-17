cmake_minimum_required(VERSION 2.8)
project( metascope )
find_package( OpenCV 3.5 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)

add_executable( vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS} )
