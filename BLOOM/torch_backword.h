#pragma once
#include "utils.h"
#include<vector>
#include<cmath>
#include<assert.h>
#include<string>
#include<tuple>

#define PI 3.14159265358979323846

//Tanh(x) = [(e^x) - (e^-x)]/[(e^x) + (e^-x)]
vec4d_t Tanh(vec4d_t input);

//Cosh(x) = [(e^x) - (e^-x)] / 2
vec4d_t Cosh(vec4d_t input_vec);

vec2d_t matmul(vec2d_t input_1, vec2d_t input_2);

vec4d_t MatMul(vec4d_t mat_1, vec4d_t mat_2);

vec4d_t NewGELUActivation(vec4d_t input_vec);

vec4d_t Addmm(vec4d_t mat_1, vec4d_t mat_2, vec1d_t bias, uint32_t beta = 1, uint32_t alpha = 1);

vec4d_t Conv1D(vec4d_t x, uint32_t nf , std::string weights_path, std::string bias_path);

vec4d_t softmax(vec4d_t& input);

vec1d_t softmax(vec1d_t& input);

vec4d_t transpose(vec4d_t hidden_states, uint32_t dim_1, uint32_t dim_2);

vec4d_t permute(vec4d_t hidden_states, const uint32_t dim_0, const uint32_t dim_1, const  uint32_t dim_2, const uint32_t dim_3);

vec4d_t view(vec4d_t hidden_states, uint32_t dim_0, uint32_t dim_1, uint32_t dim_2, uint32_t dim_3);

vec2d_t Embedding(vec2d_t Embedding, vec1d_t indices);

vec3d_t Embedding(vec2d_t Embedding, vec2d_t indices);

vec4d_t LayerNorm(vec4d_t&& input_vec, std::string path, float eps = 1e-5);

vec4d_t LayerNorm(vec4d_t& input_vec, std::string path, float eps = 1e-5);

vec4d_t Linear(vec4d_t input, vec2d_t weight);

vec4d_t NewGELUActivationBackward(vec4d_t input_vec, vec4d_t back_ward);

vec4d_t GPT2LayerNormDxBackward(vec4d_t forward_input, vec4d_t backward_input, std::string weights_path, float eps = 0.000001);

vec1d_t GPT2LayerNormDwBackward(vec4d_t forward_input, vec4d_t backward_input, float eps = 0.000001);

vec1d_t GPT2LayerNormDbBackward(vec4d_t backward_input);

vec4d_t MatmulBackDw(vec4d_t forward_input, vec4d_t backward_input);

vec4d_t MatmulBackDx(vec4d_t forward_input, vec4d_t backward_input);

vec4d_t Conv1DBackDx(vec4d_t dx, uint32_t nf, std::string weights_path);

vec4d_t Conv1DBackDw(vec4d_t forward_input, vec4d_t backward_input);

vec1d_t Conv1DBackDb(vec4d_t dx);

vec4d_t Softmax_back(vec4d_t& forward_output, vec4d_t& backward_input);