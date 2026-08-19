add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE samcore)

add_executable(gen_data gen_data.cpp)
target_link_libraries(gen_data PRIVATE samcore)
