#pragma once
#include "../utils/utils.h"
#include<vector>
#include<cmath>
#include<assert.h>
#include<string>
#include<tuple>
#include<random>
#include<numeric>
#include <regex>
#include <filesystem>

#define PI 3.14159265358979323846

//Tanh(x) = [(e^x) - (e^-x)]/[(e^x) + (e^-x)]

// inline unsigned short float32_to_bfloat16(float value);
// inline float bfloat16_to_float32(unsigned short value);
void transition(vec3d_t& input);
void transition(vec2d_t& input);
void transition(vec1d_t& input);

vec4d_t Tanh(vec4d_t input);

vec3d_t Tanh(vec3d_t input);

vec4d_t MatMul(vec4d_t mat_1, vec4d_t mat_2);

vec4d_t NewGELUActivation(vec4d_t input_vec);

vec3d_t BACK_NewGELUActivation(vec4d_t input_vec, vec4d_t back_vec);

vec4d_t Addmm(vec4d_t mat_1, vec4d_t mat_2, vec1d_t bias, uint32_t beta = 1, uint32_t alpha = 1);

vec4d_t Conv1D(vec4d_t x, uint32_t nf , std::string weights_path, std::string bias_path);

vec4d_t softmax(vec4d_t& input);

vec1d_t softmax(vec1d_t& input);

vec1d_t logsoftmax(vec1d_t input);

vec4d_t transpose(vec4d_t hidden_states, uint32_t dim_1, uint32_t dim_2);

vec4d_t permute(vec4d_t hidden_states, const uint32_t dim_0, const uint32_t dim_1, const  uint32_t dim_2, const uint32_t dim_3);

vec4d_t view(vec4d_t hidden_states, uint32_t dim_0, uint32_t dim_1, uint32_t dim_2, uint32_t dim_3);

vec2d_t Embedding(vec2d_t Embedding, vec1d_t indices);

vec3d_t Embedding(vec2d_t Embedding, vec2d_t indices);

vec4d_t LayerNorm(vec4d_t&& input_vec, vec1d_t ln_weights, vec1d_t ln_bias, float eps);

vec4d_t LayerNorm(vec4d_t& input_vec, vec1d_t weights, vec1d_t bias, float eps);

vec3d_t reshape(vec4d_t input, int dim1, int dim2, int dim3);


vec4d_t Linear(vec4d_t input, vec2d_t weight, vec1d_t bias);		

vec4d_t Linear_Nobias(vec4d_t input, vec2d_t weight);

vec1d_t CrossEntropyLoss(vec2d_t logits, vec1d_t_i labels);

vec2d_t matmul(vec2d_t input_1, vec2d_t input_2);

vec4d_t MatmulBackDx(vec4d_t forward_input, vec4d_t backward_input);

vec4d_t MatmulBackDw(vec4d_t forward_input, vec4d_t backward_input);
vec4d_t MatmulBackDw_yinxun(vec4d_t forward_input, vec4d_t backward_input);
vec4d_t LinearBackDx(vec4d_t dx, vec4d_t weights);

vec4d_t LinearBackDw(vec4d_t forward_input, vec4d_t backward_input);

vec1d_t LinearBackDb(vec4d_t dx);

vec4d_t BLOOMLayerNormDxBackward(vec4d_t forward_input, vec4d_t backward_input, vec1d_t weight, vec1d_t bias, float eps); 

vec1d_t BLOOMLayerNormDwBackward(vec4d_t forward_input, vec4d_t backward_input, float eps);

vec1d_t BLOOMLayerNormDbBackward(vec4d_t backward_input);

vec3d_t dropout(vec3d_t input, float prob, bool training);

vec3d_t dropout_add(vec3d_t input, vec3d_t residual, float prob, bool training);

vec3d_t build_alibi(vec2d_t_i& input, int num_heads);

vec4d_t reshape(vec3d_t input, int dim1, int dim2, uint32_t dim3, uint32_t dim4);

vec2d_t diag(vec1d_t x);

vec3d_t baddbmm(vec3d_t input, vec3d_t batch1, vec3d_t batch2, float beta, float alpha);

vec3d_t cat(vec3d_t input1, vec3d_t input2, int dim);

vec4d_t masked_fill(vec4d_t scores, vec4d_t_i mask, float value);

vec3d_t bmm(vec3d_t input, vec3d_t mat2);

vec2d_t vec1d_mul(vec1d_t x);

vec4d_t Softmax_back(vec4d_t& forward_output, vec4d_t& backward_input);

vec4d_t LinearDx(vec4d_t backward_input, vec2d_t weights);

vec2d_t LinearDw(vec4d_t forward_input, vec4d_t backward_input);

vec1d_t LinearDb(vec4d_t backward_input);

vec3d_t bmmDx(vec3d_t backward_input, vec3d_t mat2);

vec3d_t bmmDw(vec3d_t forward_input, vec3d_t backward_input);

vec4d_t dropoutBackward(vec4d_t backward_input, float prob, vec4d_t mask, bool is_training);

std::tuple<vec3d_t, vec3d_t> baddbmmBackward(vec3d_t backward_input, vec3d_t batch1, vec3d_t batch2, float beta, float alpha);


std::string get_last_checkpoint(std::string folder);