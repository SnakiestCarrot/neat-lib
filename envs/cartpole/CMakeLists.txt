add_executable(neat_cartpole main.cpp)
target_include_directories(neat_cartpole PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(neat_cartpole PRIVATE neat_core)
set_target_properties(neat_cartpole PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
