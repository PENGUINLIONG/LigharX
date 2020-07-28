add_subdirectory("assets")

add_executable(Demo "app.cpp")
target_link_libraries(
    Demo
    LigharX
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)
