#pragma once
#include "GPT2.h"

vec4d_t GPT2AttentionBackward(GPT2Config config, vec4d_t dx, std::string weights_path);

vec3d_t GPT2MLPBackward(GPT2Config config, vec3d_t backward_input, std::string weights_path, std::map<std::string, std::any> forward);