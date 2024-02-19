#pragma once
#ifndef __BLOOM_H__
  #define __BLOOM_H__
#include "../torch/torch.h"
#include "../utils/utils.h"

data<4> BLOOMMLP(INST_TYPE &inst2, 
				BLOOMConfig config, 
				data<4> hidden_states, 
				data<4> residual,
				uint32_t output_addr, 
				uint32_t test_addr,
				std::string weight_name, 
				std::map<std::string, uint32_t> weights_addr,
        std::map<std::string, uint32_t> forwardMap); 

// std::tuple<data<3>, std::tuple<data<3>, data<3>>, data<4>> 
data<3>
BLOOMAttention(INST_TYPE &inst2,
            	BLOOMConfig config,
	            data<3> hidden_states,
              data<3> residual, 
              data<3> alibi, 
              data<4> attention_mask, 
	            std::string weightPath,
	            std::map<std::string, uint32_t> weightMap,
              std::map<std::string, uint32_t> forwardMap,
	            uint32_t attn_output_addr,
	            uint32_t present_addr);

// std::tuple<data<3>, std::tuple<data<3>, data<3>>, data<4>>
data<3>
BLOOMBlock(INST_TYPE &inst2,
					BLOOMConfig config,
					data<4> hidden_states,
					data<4> alibi,
					data<4> attention_mask,
					std::string weight_name,
					std::map<std::string, uint32_t> weights_addr,
          std::map<std::string, uint32_t> forwardMap,
					uint32_t outputs_addr);


data<3> BLOOMModel(INST_TYPE &inst2,
                    BLOOMConfig config,
                    data<2> &input_ids,
                    data<2> attention_mask,
                    std::string weight_name,
                    std::map<std::string, uint32_t> weights_addr,
                    std::map<std::string, uint32_t> forwardMap,
                    uint32_t output_addr);


// ========================= Backward ========================

data<3> BLOOMAttentionBackward(INST_TYPE &inst2,
                              BLOOMConfig config,
                              const data<3> &backward_input,
                              std::string path,
                              std::map<std::string, uint32_t> weights_addr,
                              std::map<std::string, uint32_t> forward_addr,
                              uint32_t output_addr);

data<4> BLOOMMLPBackward(INST_TYPE &inst2,
                            BLOOMConfig config,
                            data<4> hidden_states,
                            uint32_t output_addr,
                            uint32_t test_addr,
                            std::string weights_name,
                            std::map<std::string, uint32_t> weights_addr,
                            std::map<std::string, uint32_t> forward_addr);

data<4> BLOOMBlockBackward(INST_TYPE &inst2,
                            BLOOMConfig config,
                            data<4> hidden_states,
                            uint32_t output_addr,
                            uint32_t test_addr,
                            std::string weights_name,
                            std::map<std::string, uint32_t> weights_addr,
                            std::map<std::string, uint32_t> forward_addr);

data<4> BLOOMModelBackward(INST_TYPE &inst2,
                            BLOOMConfig config,
                            data<4> hidden_states,
                            uint32_t output_addr,
                            uint32_t test_addr,
                            std::string weights_name,
                            std::map<std::string, uint32_t> weights_addr,
                            std::map<std::string, uint32_t> forward_addr);


data<4> BLOOMForCausalLMBackward(INST_TYPE &inst2, 
                                    BLOOMConfig config,
                                    data<4> hidden_states,
                                    uint32_t output_addr,
                                    uint32_t test_addr,
                                    std::string weights_name,
                                    std::map<std::string, uint32_t> weights_addr);
#endif