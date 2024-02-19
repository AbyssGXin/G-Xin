cmake_minimum_required (VERSION 3.17.5)

project(BLOOMModel)

include_directories(./bloom ./torch ./utils)

add_compile_options(-std=gnu++17  -lpthread)

add_library(bloom bloom/bloom.cpp)
add_library(torch torch/torch.cpp)
add_library(utils utils/utils.cpp)
add_library(tokenizer ../bloom_tokenizer/tokenization.cpp)

add_executable(test test.cpp)

target_link_libraries(test bloom torch utils tokenizer)
