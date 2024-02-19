#pragma once
#ifndef __GPT2_DEFINE_H__
#define __GPT2_DEFINE_H__

#include <string>

#include "FuncHelper.h"
#include "InstHelper.h"

#define __INLINE__

#include "InstHelper2.h"

typedef Inst2 INST_TYPE;
#define INST_TYPE_IS_INST2

#    define kVMemSeg (kNumberOfCores / kBytePerWord)
#    define kVMemSegShift (7 - 2)
#    define kVRegPerVMem (kNumberOfSubcores / kVMemSeg)

struct BLOOMConfig
{
    uint32_t vocab_size = 250880;
    uint32_t hidden_size=4096;
    uint32_t n_layer=30;
    uint32_t n_head=32;
    uint32_t layer_norm_epsilon=1e-5;
    uint32_t initializer_range=0.02;
	int pretraining_tp = 1;
    bool use_cache = true;
    bool training = true;
	bool slow_but_exact = false;
    bool apply_residual_connection_post_layernorm = false;
    float hidden_dropout=0.0;
    float attention_dropout=0.0;
	float update_lr = 0.00002;
	float update_lr_max = 0.00002;

    std::string activation_function = "gelu_new";  
};

struct GPT2Config
{
  uint32_t num_heads = 12;
	bool use_cache = true;
	bool scale_attn_weights = true;
	uint32_t n_embd = 768;
	std::string activation_function = "gelu_new";
	const uint32_t mlp_fc = 3072;
	uint32_t n_layer = 12;
	bool is_training = false;
	float learning_rate = 0.0005;
};

struct GPT2RuntimeConfigStruct
{
	uint32_t GPT2AttentionCattnWeightH = 768;
	uint32_t GPT2AttentionCattnWeightW = 2304;
	uint32_t GPT2AttentionCattnBiasW = 2304;
	uint32_t GPT2AttentionCprojWeightH = 768;
	uint32_t GPT2AttentionCprojWeightW = 768;
	uint32_t GPT2AttentionCprojBiasW = 768;
	uint32_t GPT2ModelWpeVocabSize = 1024;
	uint32_t GPT2LayerNormWeight = 768;
};

inline GPT2RuntimeConfigStruct &
GPT2RuntimeConfig()
{
    static GPT2RuntimeConfigStruct config;
    return config;
};
#endif