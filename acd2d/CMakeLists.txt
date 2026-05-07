cmake_minimum_required(VERSION 3.10)
project(acd2d)

# -----------------------------------------------------------------------------
# 1. Cấu hình Compiler & Options (Tương ứng với CXX, OPTS trong Makefile)
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Các cờ biên dịch từ Makefile: -W -O3 -Wno-deprecated -Wno-unused-parameter
add_compile_options(-W -O3 -Wno-deprecated -Wno-unused-parameter)

# -----------------------------------------------------------------------------
# 2. Định nghĩa các file nguồn (Tương ứng với SUBDIRS, GUIS, SRCS)
# -----------------------------------------------------------------------------
# Core sources (Thuật toán - Dùng cho cả GUI, CLI và Python)
file(GLOB_RECURSE CORE_SRCS
    "src/*.cpp"
    "src/mathtool/*.cpp"
    "src/hull/*.cpp"
    "src/edge_visibility/*.cpp"
)

# GUI sources (Chỉ dùng cho acd2d_gui)
set(GUI_SRCS
    "gui/acd2d_drawPS.cpp"
    "gui/acd2d_main_gui.cpp"
    "gui/acd2d_stat.cpp"
    "gui/GL/gl2ps.cpp"
    "gui/GL/gli.cpp"
    "gui/GL/gliDump.cpp"
    "gui/GL/gliFont.cpp"
    "gui/GL/gliTexture.cpp"
)

# CLI source (Chỉ dùng cho acd2d_cli)
set(CLI_SRCS
    "cli/acd2d_main_cli.cpp"
)

# Include directories (Tương ứng với biến INCLUDE)
set(ACD2D_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mathtool
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hull
    ${CMAKE_CURRENT_SOURCE_DIR}/src/edge_visibility
    ${CMAKE_CURRENT_SOURCE_DIR}/gui
    ${CMAKE_CURRENT_SOURCE_DIR}/gui/GL
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/FIST
)

# -----------------------------------------------------------------------------
# 3. Cấu hình thư viện FIST (Tương ứng với đoạn PLATFORM check)
# -----------------------------------------------------------------------------
if(APPLE)
    set(FIST_LIB_NAME "libFIST.osx.a")
elseif(UNIX AND NOT APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(FIST_LIB_NAME "libFIST.linux64.a")
    else()
        set(FIST_LIB_NAME "libFIST.linux.a")
    endif()
endif()

set(FIST_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/libs/FIST/${FIST_LIB_NAME}")

# Kiểm tra xem file thư viện có tồn tại không
if(NOT EXISTS "${FIST_LIBRARY}")
    message(WARNING "Khong tim thay thu vien FIST tai: ${FIST_LIBRARY}")
    message(WARNING "CLI se duoc build ma khong co FIST")
    set(FIST_AVAILABLE FALSE)
else()
    set(FIST_AVAILABLE TRUE)
endif()

# -----------------------------------------------------------------------------
# 4. TARGET 1: Ứng dụng GUI (acd2d_gui) - Giống hệt Makefile cũ
# -----------------------------------------------------------------------------
# Tìm các thư viện đồ họa (OpenGL, GLUT, X11) - chỉ cần cho GUI
find_package(OpenGL QUIET)
find_package(GLUT QUIET)

if(UNIX AND NOT APPLE)
    find_package(X11 QUIET)
endif()

# GUI target (nếu tìm thấy OpenGL/GLUT)
if(OPENGL_FOUND AND GLUT_FOUND)
    add_executable(acd2d_gui ${CORE_SRCS} ${GUI_SRCS})

# Include paths
target_include_directories(acd2d_gui PRIVATE ${ACD2D_INCLUDE_DIRS})

# Link libraries
if(APPLE)
    target_link_libraries(acd2d_gui PRIVATE ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${FIST_LIBRARY} "-framework Foundation")
else()
    # Linux: Link X11, OpenGL, GLUT, Math(m), và FIST
    target_link_libraries(acd2d_gui PRIVATE
        ${X11_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES} 
        glut               # <--- Thêm dòng này để ép buộc link thư viện glut
        GLU                
        GL                 
        ${FIST_LIBRARY}
        m 
    )
    
    # QUAN TRỌNG: Thêm cờ -no-pie để sửa lỗi linker với libFIST cũ
    target_link_options(acd2d_gui PRIVATE -no-pie)
endif()
else()
    message(STATUS "OpenGL/GLUT not found; skipping acd2d_gui target")
endif()

# -----------------------------------------------------------------------------
# 5. TARGET 2: Command-Line Interface (acd2d_cli) - Không cần GUI libs
# -----------------------------------------------------------------------------
add_executable(acd2d_cli ${CORE_SRCS} ${CLI_SRCS})

# Include paths
target_include_directories(acd2d_cli PRIVATE ${ACD2D_INCLUDE_DIRS})

# Link libraries (chỉ cần FIST và math, không cần OpenGL/GLUT)
if(FIST_AVAILABLE)
    if(APPLE)
        target_link_libraries(acd2d_cli PRIVATE ${FIST_LIBRARY} "-framework Foundation")
    else()
        # Linux: Chỉ link FIST và math
        target_link_libraries(acd2d_cli PRIVATE ${FIST_LIBRARY} m)
        
        # QUAN TRỌNG: Thêm cờ -no-pie để sửa lỗi linker với libFIST cũ
        target_link_options(acd2d_cli PRIVATE -no-pie)
    endif()
else()
    # Build mà không có FIST (nếu không tìm thấy)
    if(APPLE)
        target_link_libraries(acd2d_cli PRIVATE "-framework Foundation")
    else()
        target_link_libraries(acd2d_cli PRIVATE m)
    endif()
    message(STATUS "Building acd2d_cli without FIST library")
endif()

# Set output name
set_target_properties(acd2d_cli PROPERTIES OUTPUT_NAME acd2d)

# -----------------------------------------------------------------------------
# 6. Installation (Optional)
# -----------------------------------------------------------------------------
if(OPENGL_FOUND AND GLUT_FOUND)
    install(TARGETS acd2d_gui acd2d_cli
        RUNTIME DESTINATION bin
    )
else()
    install(TARGETS acd2d_cli
        RUNTIME DESTINATION bin
    )
endif()

# -----------------------------------------------------------------------------
# 7. Build information
# -----------------------------------------------------------------------------
message(STATUS "=== ACD2D Build Configuration ===")
message(STATUS "Build targets:")
if(OPENGL_FOUND AND GLUT_FOUND)
    message(STATUS "  - acd2d_gui (GUI application)")
endif()
message(STATUS "  - acd2d_cli (Command-line tool, output: acd2d)")
message(STATUS "FIST library: ${FIST_AVAILABLE}")
if(FIST_AVAILABLE)
    message(STATUS "FIST path: ${FIST_LIBRARY}")
endif()
message(STATUS "=================================")