#ifndef _UTILS_
#define _UTILS_
#include<vector>
#include<cmath>
#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<assert.h>

typedef std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>>  vec6d_t;
typedef std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>  vec5d_t;
typedef std::vector<std::vector<std::vector<std::vector<float>>>>  vec4d_t;
typedef std::vector<std::vector<std::vector<float>>>  vec3d_t;
typedef std::vector<std::vector<float>>  vec2d_t;
typedef std::vector<float>  vec1d_t;

typedef std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>  vec6d_t_i;
typedef std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>  vec5d_t_i;
typedef std::vector<std::vector<std::vector<std::vector<int>>>>  vec4d_t_i;
typedef std::vector<std::vector<std::vector<int>>>  vec3d_t_i;
typedef std::vector<std::vector<int>>  vec2d_t_i;
typedef std::vector<int>  vec1d_t_i;

template<typename T>
struct dimension { static constexpr std::size_t value = 0; };

template<typename T, typename... V>
struct dimension<std::vector<T, V...>>{
    static constexpr std::size_t value = 1 + dimension<T>::value;
};

void PrintVector(vec4d_t vec);

void LoadInput(vec5d_t& input_vec, std::string input_name);
void LoadInput(vec4d_t& input_vec, std::string input_name);
void LoadInput(vec3d_t& input_vec, std::string input_name);
void LoadInput(vec2d_t& input_vec, std::string input_name);
void LoadInput(vec1d_t& input_vec, std::string input_name);
void LoadInputHex(vec5d_t &input_vec, std::string input_name);
void LoadInputHex(vec4d_t &input_vec, std::string input_name);
void LoadInputHex(vec3d_t &input_vec, std::string input_name);
void LoadInputHex(vec2d_t &input_vec, std::string input_name);
void LoadInputHex(vec1d_t &input_vec, std::string input_name);
void LoadInputI(vec5d_t_i &input_vec, std::string input_name);
void LoadInputI(vec4d_t_i &input_vec, std::string input_name);
void LoadInputI(vec3d_t_i &input_vec, std::string input_name);
void LoadInputI(vec2d_t_i &input_vec, std::string input_name);
void LoadInputI(vec1d_t_i &input_vec, std::string input_name);
void LoadInputB(vec5d_t_i &input_vec, std::string input_name);
void LoadInputB(vec4d_t_i &input_vec, std::string input_name);
void LoadInputB(vec3d_t_i &input_vec, std::string input_name);
void LoadInputB(vec2d_t_i &input_vec, std::string input_name);
void LoadInputB(vec1d_t_i &input_vec, std::string input_name);
void WriteOutput(vec3d_t data, std::string name);
void WriteOutput(vec4d_t data, std::string name);
void WriteOutput(vec5d_t data, std::string name);
void WriteOutput(vec6d_t data, std::string name);
void WriteOutput(vec5d_t_i data, std::string name);
vec4d_t BHWC2BCHW(const vec4d_t& data);
vec4d_t BCHW2BHWC(const vec4d_t& data);
vec4d_t Rearrange(const vec4d_t& data, int dim0, int dim1, int dim2, int dim3);
vec4d_t View(const vec4d_t& data, int dim0, int dim1, int dim2, int dim3);
vec4d_t AddVector(vec4d_t vec1, vec4d_t vec2);
vec4d_t SubVector(vec4d_t vec1, vec4d_t vec2);
vec4d_t MulVector(vec4d_t& vec1, vec4d_t& vec2);
vec4d_t AddVector(vec4d_t& vec1, vec4d_t&& vec2);
vec4d_t MulVector(vec4d_t&& vec1, vec4d_t&& vec2);

vec5d_t_i getHeadMask(vec5d_t_i &head_mask, uint32_t num_hidden_layers, bool is_attention_chunked = false);
template <typename T, typename... V>
T operator*(float val, T x);

template <typename T>
vec5d_t_i convertHeadMaskTo5d(T head_mask, int num_hidden_layers);

vec2d_t update_weight(vec2d_t old_weight, vec2d_t grad, float lr);

vec1d_t update_bias(vec1d_t old_bias, vec1d_t grad, float lr);
#endif