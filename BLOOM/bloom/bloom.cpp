#include "bloom.h"
// using namespace std;

std::string Training_7B_Back = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_backward_f32/";
std::string Training_7B_For = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/";

std::string Training_560M_Back = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/";
std::string Training_560M_For = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/";

std::string Detect_Back = Training_7B_Back;
std::string Detect_For = Training_7B_For;


std::string str = "f32";


data<4> BLOOMMLP(INST_TYPE &inst2, 
                BLOOMConfig config, 
                data<4> hidden_states, 
                data<4> residual,
                uint32_t output_addr, 
                uint32_t test_addr,
                std::string weights_name, 
                std::map<std::string, uint32_t> weights_addr,
                std::map<std::string, uint32_t> forwardMap)
{
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    Instruction inst;
    data<3> output;
    data<3> savedata;

    data<2> mlp_fc_weight;
    mlp_fc_weight.hbmaddr = weights_addr.at(weights_name + ".dense_h_to_4h.weight");
    mlp_fc_weight.dims = {4 * config.hidden_size, config.hidden_size};
    data<2> mlp_proj_weight;
    mlp_proj_weight.hbmaddr = weights_addr.at(weights_name + ".dense_4h_to_h.weight");
    mlp_proj_weight.dims = {config.hidden_size, 4 * config.hidden_size}; 

    data<1> mlp_fc_bias;
    mlp_fc_bias.hbmaddr = weights_addr.at(weights_name + ".dense_h_to_4h.bias");
    mlp_fc_bias.dims = {4 * config.hidden_size};
    data<1> mlp_proj_bias;
    mlp_proj_bias.hbmaddr = weights_addr.at(weights_name + ".dense_4h_to_h.bias");
    mlp_proj_bias.dims = {config.hidden_size};

    if (config.training)
    {
      std::string name = weights_name + ".dense_h_to_4h_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {hidden_states.dims[1], hidden_states.dims[2], hidden_states.dims[3]};
      VMEM_TO_HBM(instruction_list, hidden_states.addr, savedata.hbmaddr, hidden_states.size());
    }


    inst2.Spy(weights_name + ".dense_h_to_4h_in", hidden_states.asVMem(inst2), Detect_For + weights_name + ".dense_h_to_4h_in.txt");
    output = linear(inst2, hidden_states[0], mlp_fc_weight, mlp_fc_bias, test_addr);
    inst2.Spy(weights_name + ".dense_h_to_4h_out", output.asVMem(inst2), Detect_For + weights_name + ".dense_h_to_4h_out.txt");


    if (config.training)
    {
      std::string name = weights_name + ".gelu_impl_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {output.dims[0], output.dims[1], output.dims[2]};
      VMEM_TO_HBM(instruction_list, output.addr, savedata.hbmaddr, output.size());
    }

    inst2.Spy(weights_name + ".gelu_impl_in", output.asVMem(inst2), Detect_For + weights_name + ".gelu_impl_in.txt");
    output = NewGELUActivation(inst2, output.as<4>(), output.addr)[0];
    inst2.Spy(weights_name + ".gelu_impl_out", output.asVMem(inst2), Detect_For + weights_name + ".gelu_impl_out.txt");


    if (config.training)
    {
      std::string name = weights_name + ".dense_4h_to_h_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {output.dims[0], output.dims[1], output.dims[2]};
      VMEM_TO_HBM(instruction_list, output.addr, savedata.hbmaddr, output.size());
    }


    inst2.Spy(weights_name + ".dense_4h_to_h_in", output.asVMem(inst2), Detect_For + weights_name + ".dense_4h_to_h_in.txt");
    output = linear(inst2, output, mlp_proj_weight, mlp_proj_bias, output.addr + output.size());
    inst2.Spy(weights_name + ".dense_4h_to_h_out", output.asVMem(inst2), Detect_For + weights_name + ".dense_4h_to_h_out.txt");


    output = dropout_add(inst2, config, output, residual[0], output.addr + output.size());
    if(output.addr != output_addr)
    {
        data<3> dest(output_addr, output.dims);
        INST_TYPE inst2;
        Memcopy(output.asVMem(inst2), dest.asVMem(inst2));
        instruction_list.insert(instruction_list.end(), inst2.inst.insts.begin(), inst2.inst.insts.end());
    }
    output.addr = output_addr;
    return output.as<4>();
}


data<3> BLOOMAttention(INST_TYPE &inst2,
              BLOOMConfig config,
              data<3> hidden_states,
              data<3> residual, 
              data<3> alibi, 
              data<4> attention_mask, 
              std::string weightPath,
              std::map<std::string, uint32_t> weightMap,
              std::map<std::string, uint32_t> forwardMap,
              uint32_t attn_output_addr,
              uint32_t present_addr) 
{
  std::cout << "attention name: " << weightPath << std::endl;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  data<3> savedata;

  data<2> query_key_value_weights;
  query_key_value_weights.hbmaddr = weightMap.at(weightPath + ".query_key_value.weight");
  query_key_value_weights.dims = {uint32_t(3) * config.hidden_size, hidden_states.dims[2]};
  data<1> query_key_value_bias;
  query_key_value_bias.hbmaddr = weightMap.at(weightPath + ".query_key_value.bias");
  query_key_value_bias.dims = {uint32_t(3) * config.hidden_size};

  if (config.training)
  {
    std::string name = weightPath + ".query_key_value_in";
    if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
    savedata.hbmaddr = forwardMap.at(name);
    savedata.dims = {hidden_states.dims[0], hidden_states.dims[1], hidden_states.dims[2]};
    VMEM_TO_HBM(instruction_list, hidden_states.addr, savedata.hbmaddr, hidden_states.size());
  }

  inst2.Spy(weightPath + ".query_key_value_in", hidden_states.asVMem(inst2), Detect_For + weightPath + ".query_key_value_in.txt");
  data<3> fused_qkv = linear(inst2, hidden_states, query_key_value_weights, query_key_value_bias, attn_output_addr);
  inst2.Spy(weightPath + ".query_key_value_out", fused_qkv.asVMem(inst2), Detect_For + weightPath + ".query_key_value_out.txt");


  data<4> query_layer(fused_qkv.addr + fused_qkv.size(), {fused_qkv.dims[0], fused_qkv.dims[1], config.n_head, config.hidden_size / config.n_head});
  data<4> key_layer(fused_qkv.addr + fused_qkv.size() + query_layer.size(), {fused_qkv.dims[0], fused_qkv.dims[1], config.n_head, config.hidden_size / config.n_head});
  data<4> value_layer(fused_qkv.addr + fused_qkv.size() + 2 * query_layer.size(), {fused_qkv.dims[0], fused_qkv.dims[1], config.n_head, config.hidden_size / config.n_head});


  int batch_size = fused_qkv.dims[0];
  int q_length = fused_qkv.dims[1];

  for (int b = 0; b < query_layer.dims[0]; b++) {
    for (int s = 0; s < query_layer.dims[1]; s += kNumberOfSubcoresPerCore) {
      uint32_t use_row = std::min((uint32_t)kNumberOfSubcoresPerCore, query_layer.dims[1] - s);
      for (int h = 0; h < fused_qkv.dims[2]; h += (config.hidden_size / config.n_head * 3)) {
        uint32_t use_col = std::min((uint32_t)(config.hidden_size / config.n_head), (fused_qkv.dims[2] - h) / 3);
        auto q_reg = inst2.AllocVReg("");
        auto k_reg = inst2.AllocVReg("");
        auto v_reg = inst2.AllocVReg("");
        uint32_t load_offset = b * query_layer.dims[1] * fused_qkv.dims[2] + s * fused_qkv.dims[2] + h;
        uint32_t store_offset = b * query_layer.dims[1] * config.hidden_size + s * config.hidden_size + h / 3;
        Load8_128(inst2, q_reg, use_row, use_col, fused_qkv.addr + load_offset, fused_qkv.dims[2]);
        Load8_128(inst2, k_reg, use_row, use_col, fused_qkv.addr + load_offset + use_col, fused_qkv.dims[2]);
        Load8_128(inst2, v_reg, use_row, use_col, fused_qkv.addr + load_offset + use_col * 2, fused_qkv.dims[2]);
        Store8_128(inst2, q_reg, use_row, use_col, query_layer.addr + store_offset, fused_qkv.dims[2] / 3);
        Store8_128(inst2, k_reg, use_row, use_col, query_layer.addr + store_offset + query_layer.size(), fused_qkv.dims[2] / 3);
        Store8_128(inst2, v_reg, use_row, use_col, query_layer.addr + store_offset + query_layer.size() * 2, fused_qkv.dims[2] / 3);
      }
    }
  }

  std::vector<uint32_t> dimSize = {query_layer.dims[0], query_layer.dims[1], query_layer.dims[2], query_layer.dims[3]};
  std::vector<uint32_t> permute_dims = {0, 2, 1, 3};
  Permute(instruction_list,
          {}, {}, {}, {},
          query_layer.addr,
          dimSize,
          permute_dims,
          fused_qkv.addr);
  query_layer.addr = fused_qkv.addr;
  query_layer.dims = {query_layer.dims[0], query_layer.dims[2], query_layer.dims[1], query_layer.dims[3]};
  

  Permute(instruction_list,
          {}, {}, {}, {},
          key_layer.addr,
          dimSize,
          {0, 2, 3, 1},
          fused_qkv.addr + query_layer.size());
  key_layer.addr = fused_qkv.addr + query_layer.size();
  key_layer.dims = {key_layer.dims[0], key_layer.dims[2], key_layer.dims[3], key_layer.dims[1]};

  Permute(instruction_list,
          {}, {}, {}, {},
          value_layer.addr,
          dimSize,
          permute_dims,
          fused_qkv.addr + query_layer.size() + key_layer.size());
  value_layer.addr = fused_qkv.addr + query_layer.size() + key_layer.size();
  value_layer.dims = {value_layer.dims[0], value_layer.dims[2], value_layer.dims[1], value_layer.dims[3]};

  data<3> _query_layer(query_layer.addr, {batch_size * config.n_head, q_length, config.hidden_size / config.n_head});
  data<3> _key_layer(key_layer.addr, {batch_size * config.n_head, config.hidden_size / config.n_head , q_length});
  data<3> _value_layer(value_layer.addr, {batch_size * config.n_head, q_length, config.hidden_size / config.n_head});

  int kv_length = _key_layer.dims[2];
  std::tuple<data<3>, data<3>> present;

  if (config.training)
  {
    std::string name = weightPath + ".query_layer";
    if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
    savedata.hbmaddr = forwardMap.at(name);
    savedata.dims = {_query_layer.dims[0], _query_layer.dims[1], _query_layer.dims[2]};
    VMEM_TO_HBM(instruction_list, _query_layer.addr, savedata.hbmaddr, _query_layer.size());

    name = weightPath + ".key_layer";
    if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
    savedata.hbmaddr = forwardMap.at(name);
    savedata.dims = {_key_layer.dims[0], _key_layer.dims[1], _key_layer.dims[2]};
    VMEM_TO_HBM(instruction_list, _key_layer.addr, savedata.hbmaddr, _key_layer.size());

    name = weightPath + ".value_layer";
    if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
    savedata.hbmaddr = forwardMap.at(name);
    savedata.dims = {_value_layer.dims[0], _value_layer.dims[1], _value_layer.dims[2]};
    VMEM_TO_HBM(instruction_list, _value_layer.addr, savedata.hbmaddr, _value_layer.size());
  } 

  inst2.Spy(weightPath + ".query_layer", _query_layer.asVMem(inst2), Detect_For + weightPath + ".query_layer.txt");
  if(config.use_cache) {
    present = std::make_tuple(_key_layer, _value_layer);
  }
  inst2.Spy(weightPath + ".key_layer", _key_layer.asVMem(inst2), Detect_For + weightPath + ".key_layer.txt");
  float inv_norm_factor = 1.0 / std::sqrt(config.hidden_size / config.n_head);
  float beta = 1.0;
  inst2.Spy(weightPath + ".value_layer", _value_layer.asVMem(inst2), Detect_For + weightPath + ".value_layer.txt");


  inst2.Spy(weightPath + ".baddbmm_in", alibi.asVMem(inst2), Detect_For + weightPath + ".baddbmm_in.txt");
  data<3> matmul_result = baddbmm(inst2, alibi, _query_layer, _key_layer, beta, inv_norm_factor, _value_layer.addr + AlignTo128Bytes(_value_layer.size()));
  inst2.Spy(weightPath + ".baddbmm_out", matmul_result.asVMem(inst2), Detect_For + weightPath + ".baddbmm_out.txt");

  data<4> attention_scores(matmul_result.addr, {batch_size, config.n_head, q_length, kv_length});
  
  data<4> attention_weights = maskedFill(inst2, attention_scores, attention_mask, -3.4028234663852886e+38, attention_scores.addr + AlignTo128Bytes(attention_scores.size()));

  inst2.Spy(weightPath + ".softmax_in", attention_weights.asVMem(inst2), Detect_For + weightPath + ".softmax_in.txt");
  Softmax(instruction_list, attention_weights.addr, attention_weights.addr + AlignTo128Bytes(attention_weights.size()), attention_weights.dims[0] * attention_weights.dims[1] * attention_weights.dims[2], attention_weights.dims[3]);
  data<3> attention_probs_reshaped(attention_weights.addr + AlignTo128Bytes(attention_weights.size()), {batch_size * config.n_head, q_length, kv_length});
  inst2.Spy(weightPath + ".softmax_out", attention_probs_reshaped.asVMem(inst2), Detect_For + weightPath + ".softmax_out.txt");
  
  if (config.training)
  {
    std::string name = weightPath + ".softmax_out";
    if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
    savedata.hbmaddr = forwardMap.at(name);
    savedata.dims = {attention_probs_reshaped.dims[0], attention_probs_reshaped.dims[1], attention_probs_reshaped.dims[2]};
    VMEM_TO_HBM(instruction_list, attention_probs_reshaped.addr, savedata.hbmaddr, attention_probs_reshaped.size());
  }

  if (config.training)
  {
    std::string name = weightPath + ".attention_probs_reshaped";
    if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
    savedata.hbmaddr = forwardMap.at(name);
    savedata.dims = {attention_probs_reshaped.dims[0], attention_probs_reshaped.dims[1], attention_probs_reshaped.dims[2]};
    VMEM_TO_HBM(instruction_list, attention_probs_reshaped.addr, savedata.hbmaddr, attention_probs_reshaped.size());
  }

  inst2.Spy(weightPath + ".bmm_in", attention_probs_reshaped.asVMem(inst2), Detect_For + weightPath + ".bmm_in.txt");
  data<3> context_layer = matmulIvWv(inst2, attention_probs_reshaped.as<4>(), _value_layer.as<4>(), attention_probs_reshaped.addr + AlignTo128Bytes(attention_probs_reshaped.size()))[0];
  inst2.Spy(weightPath + ".bmm_out", context_layer.asVMem(inst2), Detect_For + weightPath + ".bmm_out.txt");

  Permute(instruction_list,
          {}, {}, {}, {},
          context_layer.addr,
          {context_layer.dims[0] / config.n_head, uint32_t(config.n_head), context_layer.dims[1], context_layer.dims[2]},
          {0, 2, 1, 3},
          context_layer.addr + context_layer.size());

  context_layer.addr = context_layer.addr + context_layer.size();
  context_layer.dims = {(context_layer.dims[0] / config.n_head), context_layer.dims[1], (uint32_t)config.hidden_size};

  data<2> dense_weights;
  dense_weights.hbmaddr = weightMap.at(weightPath + ".dense.weight");
  dense_weights.dims = {uint32_t(config.hidden_size), uint32_t(config.hidden_size)};
  data<1> dense_bias;
  dense_bias.hbmaddr = weightMap.at(weightPath + ".dense.bias");
  dense_bias.dims = {uint32_t(config.hidden_size)};

  data<3> output_tensor;
  if (config.pretraining_tp > 1 && config.slow_but_exact)
  {
    int slices = config.hidden_size / config.pretraining_tp;
    for (int i = 0; i < config.pretraining_tp; i++) {

    }
  }
  else{
    if (config.training)
    {
      std::string name = weightPath + ".dense_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {context_layer.dims[0], context_layer.dims[1], context_layer.dims[2]};
      VMEM_TO_HBM(instruction_list, context_layer.addr, savedata.hbmaddr, context_layer.size());
    }
    inst2.Spy(weightPath + ".dense_in", context_layer.asVMem(inst2), Detect_For + weightPath + ".dense_in.txt");
    output_tensor = linear(inst2, context_layer, dense_weights, dense_bias, context_layer.addr + AlignTo128Bytes(context_layer.size()));
    inst2.Spy(weightPath + ".dense_out", output_tensor.asVMem(inst2), Detect_For + weightPath + ".dense_out.txt");
  }

  AddVector(instruction_list, output_tensor.as<4>(), residual.as<4>(), output_tensor.addr);
  
  if(output_tensor.addr != attn_output_addr)
  {
      data<3> dest(attn_output_addr, output_tensor.dims);
      INST_TYPE inst2;
      Memcopy(output_tensor.asVMem(inst2), dest.asVMem(inst2));
      instruction_list.insert(instruction_list.end(), inst2.inst.insts.begin(), inst2.inst.insts.end());
  }
  output_tensor.addr = attn_output_addr;
  return output_tensor;
}


data<3>
BLOOMBlock(INST_TYPE &inst2, 
                    BLOOMConfig config, 
                    data<4> hidden_states, 
                    data<4> build_alibi,
                    data<4> attention_mask,
                    std::string weights_name,
                    std::map<std::string, uint32_t> weights_addr,
                    std::map<std::string, uint32_t> forwardMap,
                    uint32_t outputs_addr)
{
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    Instruction inst;
    data<3> layernorm_output;
    data<3> output(outputs_addr, {1, hidden_states.dims[2], config.hidden_size});
    data<1> input_layernorm_weight(output.addr + output.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".input_layernorm.weight"), input_layernorm_weight.addr, input_layernorm_weight.size());
    data<1> input_layernorm_bias(input_layernorm_weight.addr + input_layernorm_weight.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list,weights_addr.at(weights_name + ".input_layernorm.bias"), input_layernorm_bias.addr, input_layernorm_bias.size());
    
    data<3> savedata;
    if (config.training)
    {
      std::string name = weights_name + ".input_layernorm_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {hidden_states.dims[1], hidden_states.dims[2], hidden_states.dims[3]};
      VMEM_TO_HBM(instruction_list, hidden_states.addr, savedata.hbmaddr, hidden_states.size());
    }
    inst2.Spy(weights_name + ".input_layernorm_in", hidden_states.asVMem(inst2), Detect_For + weights_name + ".input_layernorm_in.txt");
    layernorm_output = LayerNorm(inst2, hidden_states[0], input_layernorm_weight, input_layernorm_bias, input_layernorm_bias.addr + input_layernorm_bias.size(), config.layer_norm_epsilon);
    inst2.Spy(weights_name + ".input_layernorm_out", layernorm_output.asVMem(inst2), Detect_For + weights_name + ".input_layernorm_out.txt");


    data<3> residual(0, {hidden_states.dims[1], hidden_states.dims[2], hidden_states.dims[3]});

    if (config.apply_residual_connection_post_layernorm)
    {
        residual.addr = layernorm_output.addr;
    }
    else
    {
        residual.addr = hidden_states.addr;
    }

    uint32_t attn_addr = std::max(layernorm_output.addr + layernorm_output.size(), residual.addr + residual.size());
    std::cout << "attn_addr: " << attn_addr << std::endl;

    inst2.Spy(weights_name + ".self_attention_in", layernorm_output.asVMem(inst2), Detect_For + weights_name + ".self_attention_in.txt");
    data<3> attn_output = BLOOMAttention(inst2, config, layernorm_output, residual, build_alibi[0], attention_mask, weights_name + ".self_attention", weights_addr, forwardMap, attn_addr, attn_addr);
    inst2.Spy(weights_name + ".self_attention_out", attn_output.asVMem(inst2), Detect_For + weights_name + ".self_attention_out.txt");

    data<1> post_layernorm_weight(attn_output.addr + attn_output.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".post_attention_layernorm.weight"), post_layernorm_weight.addr, post_layernorm_weight.size());
    data<1> post_layernorm_bias(post_layernorm_weight.addr + post_layernorm_weight.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".post_attention_layernorm.bias"), post_layernorm_bias.addr, post_layernorm_bias.size());

    if (config.training)
    {
      std::string name = weights_name + ".post_attention_layernorm_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {attn_output.dims[0], attn_output.dims[1], attn_output.dims[2]};
      VMEM_TO_HBM(instruction_list, attn_output.addr, savedata.hbmaddr, attn_output.size());
    }
    inst2.Spy(weights_name + ".post_attention_layernorm_in", attn_output.asVMem(inst2), Detect_For + weights_name + ".post_attention_layernorm_in.txt");
    layernorm_output = LayerNorm(inst2, attn_output, post_layernorm_weight, post_layernorm_bias, post_layernorm_bias.addr + post_layernorm_bias.size(), config.layer_norm_epsilon);
    inst2.Spy(weights_name + ".post_attention_layernorm_out", layernorm_output.asVMem(inst2), Detect_For + weights_name + ".post_attention_layernorm_out.txt");
    if(config.apply_residual_connection_post_layernorm)
    {
          residual.addr = layernorm_output.addr;
    }
    else
    {
        residual.addr = attn_output.addr;
    }
    uint32_t base_addr = std::max(layernorm_output.addr + layernorm_output.size(), residual.addr + residual.size());


    inst2.Spy(weights_name + ".mlp_in", layernorm_output.asVMem(inst2), Detect_For + weights_name + ".mlp_in.txt");
    output = BLOOMMLP(inst2, config, layernorm_output.as<4>(), residual.as<4>(), output.addr, base_addr, weights_name + ".mlp", weights_addr, forwardMap)[0];
    inst2.Spy(weights_name + ".mlp_out", output.asVMem(inst2), Detect_For + weights_name + ".mlp_out.txt");
    if(output.addr != outputs_addr)
    {
        data<3> dest(outputs_addr, output.dims);
        INST_TYPE inst2;
        Memcopy(output.asVMem(inst2), dest.asVMem(inst2));
        instruction_list.insert(instruction_list.end(), inst2.inst.insts.begin(), inst2.inst.insts.end());
    }
    output.addr = outputs_addr;
    return output;
}

data<3> BLOOMModel(INST_TYPE &inst2,
                    BLOOMConfig config,
                    data<2> &input_ids,
                    data<2> attention_mask,
                    std::string weight_name,
                    std::map<std::string, uint32_t> weights_addr,
                    std::map<std::string, uint32_t> forwardMap,
                    uint32_t output_addr)
{
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    Instruction inst;

    bool gradient_checkpointing = false;
    uint32_t batch_size = input_ids.dims[0];
    uint32_t seq_length = input_ids.dims[1];

    data<3> inputs_embeds(output_addr, {1, input_ids.dims[1], config.hidden_size});
    data<2> word_embedding_weight;
    word_embedding_weight.hbmaddr = weights_addr.at(weight_name + ".word_embeddings.weight");
    word_embedding_weight.dims = {config.vocab_size, config.hidden_size};

    // word_embeddings(Embeddings): [1, 115] -> [1, 115, 1024]
    inst2.Spy(weight_name + ".word_embeddings_in", input_ids.asVMem(inst2), Detect_For + weight_name + ".word_embeddings_in.txt");
    InstVer::Embeddings(inst2, word_embedding_weight.hbmaddr, word_embedding_weight.dims[0], word_embedding_weight.dims[1], input_ids.addr, input_ids.size(), inputs_embeds.addr, inputs_embeds.addr + inputs_embeds.size());
    inst2.Spy(weight_name + ".word_embeddings_out", inputs_embeds.asVMem(inst2), Detect_For + weight_name + ".word_embeddings_out.txt");

    data<1> word_embeddings_layernorm_weight(inputs_embeds.addr + inputs_embeds.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weight_name + ".word_embeddings_layernorm.weight"), word_embeddings_layernorm_weight.addr, word_embeddings_layernorm_weight.size());
    data<1> word_embeddings_layernorm_bias(word_embeddings_layernorm_weight.addr + word_embeddings_layernorm_weight.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weight_name + ".word_embeddings_layernorm.bias"), word_embeddings_layernorm_bias.addr, word_embeddings_layernorm_bias.size());

    data<3> savedata;
    if (config.training)
    {
      std::string name = weight_name + ".word_embeddings_layernorm_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {inputs_embeds.dims[0], inputs_embeds.dims[1], inputs_embeds.dims[2]};
      VMEM_TO_HBM(instruction_list, inputs_embeds.addr, savedata.hbmaddr, inputs_embeds.size());
    }

    // word_embeddings_layernorm(LayerNorm): [1, 115, 1024] -> [1, 115, 1024] 
    /*
    --LayerNorm:
        把神经元在经过非线性函数映射后向取值区间极限饱和区靠拢的输入分布强行拉回到均值为0方差为1的比较标准的正态分布的区间，
      使得非线性变换函数的输入值落入激活函数比较敏感的区域，这样会让让梯度变大，由此避免了梯度消失的问题。而梯度变大也意味
      着学习收敛速度快，能大大加快训练速度。
    */
    inst2.Spy(weight_name + ".word_embeddings_layernorm_in", inputs_embeds.asVMem(inst2), Detect_For + weight_name + ".word_embeddings_layernorm_in.txt");
    data<3> hidden_states = LayerNorm(inst2, inputs_embeds, word_embeddings_layernorm_weight, word_embeddings_layernorm_bias, word_embeddings_layernorm_bias.addr + word_embeddings_layernorm_bias.size(), config.layer_norm_epsilon);
    inst2.Spy(weight_name + ".word_embeddings_layernorm_out", hidden_states.asVMem(inst2), Detect_For + weight_name + ".word_embeddings_layernorm_out.txt");

    std::vector<std::tuple<data<3>, data<3>>> presents;
    std::vector<data<4>> all_self_attentions;
    data<4> all_hidden_stats;

    if (gradient_checkpointing && config.training)
    {
        if (config.use_cache)
        {
            std::clog << "`use_cache = Trus` is incompatible with gradient checkpointing. Setting `use_cache = False`...";
        }
        config.use_cache = false;
    }

    uint32_t seq_length_with_past = seq_length;
    uint32_t past_key_values_length = 0;

    std::tuple<uint32_t, uint32_t> input_shape(batch_size, seq_length);
    if(attention_mask.size() == 0) {
      attention_mask.addr = AlignTo128Bytes(hidden_states.addr + hidden_states.size());
      attention_mask.dims = {batch_size, seq_length};
      auto temp_reg = inst2.AllocVReg("");
      auto core_id_reg = inst2.AllocVReg("");
      inst2(VMov, 48, temp_reg.id);
      inst2(VCoreId, core_id_reg.id);
      for (int i = 0; i < batch_size * seq_length; i += kNumberOfSubcores) {
        int set_mask = std::min(kNumberOfSubcores, int(batch_size * seq_length - i));
        inst2(VLsS, core_id_reg.id, inst2.inst.ImmeS(set_mask), 0);
        inst2(VStM0, temp_reg.id, (attention_mask.addr + i) / (kVMemSeg * 4));
      }
    }

    /*
    --Alibi:        [1, config.n_head, 1, 115]
        与传统方法不同，ALiBi不向单词embedding中添加位置embedding，而是根据token之间的
      距离给 attention score 加上一个预设好的偏置矩阵.
    */
    data<3> alibi(std::max(hidden_states.addr + AlignTo128Bytes(hidden_states.size()), attention_mask.addr + AlignTo128Bytes(attention_mask.size())), {config.n_head * attention_mask.dims[0], 1, seq_length});
    build_alibi(inst2, attention_mask, config.n_head, alibi.addr);

    // inst2.Spy("alibi", alibi.asVMem(inst2), "alibi");
    data<4> causal_mask = _prepare_attn_mask(inst2, attention_mask, input_shape, past_key_values_length, alibi.addr + AlignTo128Bytes(alibi.size()));
    uint32_t Block_addr = causal_mask.addr + AlignTo128Bytes(causal_mask.size());
    // inst2.Spy("causal_mask", causal_mask.asVMem(inst2), "causal_mask");

    std::vector<data<3>> block_output;
    for (int i = 0; i <= 23; i++)
    {   
        std::cout << "hidden_states_1: " << hidden_states.addr << std::endl;
        inst2.Spy(weight_name + ".h." + std::to_string(i) + "_in", hidden_states.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/" + weight_name + ".h." + std::to_string(i) + "_in.txt");
        data<3> block = BLOOMBlock(inst2, config, hidden_states.as<4>(), alibi.as<4>(), causal_mask, weight_name + ".h." + std::to_string(i), weights_addr, forwardMap, Block_addr);
        inst2.Spy(weight_name + ".h." + std::to_string(i) + "_out", block.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/" + weight_name + ".h." + std::to_string(i) + "_out.txt");
        hidden_states.addr = block.addr;
    }
    
    data<1> ln_f_weight(hidden_states.addr + hidden_states.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weight_name + ".ln_f.weight"), ln_f_weight.addr, ln_f_weight.size());
    data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weight_name + ".ln_f.bias"), ln_f_bias.addr, ln_f_bias.size());

    if (config.training)
    {
      std::string name = weight_name + ".ln_f_in";
      if (forwardMap.find(name) == forwardMap.end()) std::cout << "Not forwardMap: " << name << std::endl;
      savedata.hbmaddr = forwardMap.at(name);
      savedata.dims = {hidden_states.dims[0], hidden_states.dims[1], hidden_states.dims[2]};
      VMEM_TO_HBM(instruction_list, hidden_states.addr, savedata.hbmaddr, hidden_states.size());
    }
    inst2.Spy(weight_name + ".ln_f_in", hidden_states.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/" + weight_name + ".ln_f_in.txt");
    hidden_states = LayerNorm(inst2, hidden_states, ln_f_weight, ln_f_bias, ln_f_bias.addr + ln_f_bias.size(), config.layer_norm_epsilon);
    inst2.Spy(weight_name + ".ln_f_out", hidden_states.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/" + weight_name + ".ln_f_out.txt");
    if(hidden_states.addr != output_addr) {
      data<3> src(output_addr, hidden_states.dims);
      INST_TYPE inst2;
      Memcopy(hidden_states.asVMem(inst2), src.asVMem(inst2));
      instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
    }
    hidden_states.addr = output_addr;
    return hidden_states;
}


data<3> BLOOMAttentionBackward(INST_TYPE &inst2,
                              BLOOMConfig config,
                              const data<3> &backward_input,
                              std::string path,
                              std::map<std::string, uint32_t> weights_addr,
                              std::map<std::string, uint32_t> forward_addr,
                              uint32_t output_addr) 
{
  
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  Instruction *inst;

  data<3> dense_dx;
  data<2> dense_dw;
  data<1> dense_db;

  if (config.pretraining_tp > 1 && config.slow_but_exact)
  {
  }
  else{
    data<2> dense_weights;
    dense_weights.hbmaddr = weights_addr.at(path + ".dense.weight");
    dense_weights.dims = {config.hidden_size, config.hidden_size};
    data<1> dense_bias;
    dense_bias.hbmaddr = weights_addr.at(path + ".dense.bias");
    dense_bias.dims = {config.hidden_size};
    data<2> dense_forward_in;
    dense_forward_in.hbmaddr = forward_addr.at(path + ".dense_in");
    dense_forward_in.dims = {backward_input.dims[1], config.hidden_size};

    inst2.Spy(path + ".dense_in", backward_input.asVMem(inst2), Detect_Back + path + ".dense_in.txt");
    dense_dx = matmul(inst2, backward_input.as<4>(), dense_weights, output_addr /* + backward_input.size()*/)[0];
    inst2.Spy(path + ".dense_out", dense_dx.asVMem(inst2), Detect_Back + path + ".dense_out.txt");

    if (config.training)
    {
      HBM_TO_VMEM(instruction_list, dense_forward_in.hbmaddr, dense_dx.addr + dense_dx.size(), dense_forward_in.size());
      dense_forward_in.addr = dense_dx.addr + dense_dx.size();
      
      for (int i = 0; i < backward_input.dims[0]; i++) {
        Transpose(instruction_list,
                  {}, {}, {}, {},
                  backward_input.addr + i * backward_input.dims[1] * backward_input.dims[2],
                  backward_input.dims[1],
                  backward_input.dims[2],
                  dense_forward_in.addr + dense_forward_in.size() + i * backward_input.dims[1] * backward_input.dims[2]);
      }

      data<3> backward_input_T(dense_forward_in.addr + dense_forward_in.size(), {backward_input.dims[0], backward_input.dims[2], backward_input.dims[1]});
      MatMulUpdateWeight(inst2, backward_input_T, dense_forward_in.as<3>(), dense_weights, config.update_lr, backward_input_T.addr + backward_input_T.size());

      dense_db = Conv1DBackDb(inst2, backward_input.as<4>(), dense_dx.addr + dense_dx.size());
      UpdateBias(inst2, config.update_lr, dense_bias, dense_db, dense_db.addr + dense_db.size());
    }
  }

  Permute(instruction_list,
          {}, {}, {}, {},
          dense_dx.addr,
          {dense_dx.dims[0], dense_dx.dims[1], config.n_head, config.hidden_size / config.n_head},
          {0, 2, 1, 3},
          dense_dx.addr + dense_dx.size());
 
  dense_dx.addr = dense_dx.addr + dense_dx.size();
  dense_dx.dims = {dense_dx.dims[0] * config.n_head, dense_dx.dims[1], config.hidden_size / config.n_head};

  data<3> bmm_forward_in1;
  bmm_forward_in1.hbmaddr = forward_addr.at(path + ".attention_probs_reshaped");
  bmm_forward_in1.dims = {backward_input.dims[0] * config.n_head, backward_input.dims[1], backward_input.dims[1]};
  data<3> bmm_forward_in2;
  bmm_forward_in2.hbmaddr = forward_addr.at(path + ".value_layer");
  bmm_forward_in2.dims = {backward_input.dims[0] * config.n_head, backward_input.dims[1], config.hidden_size / config.n_head};

  HBM_TO_VMEM(inst2, bmm_forward_in2.hbmaddr, dense_dx.addr + dense_dx.size(), bmm_forward_in2.size());
  bmm_forward_in2.addr = dense_dx.addr + dense_dx.size();

  inst2.Spy(path + ".bloombmm_in", dense_dx.asVMem(inst2), Detect_Back + path + ".bloombmm_in.txt");
  data<3> bmm_dx = MatMulDxBackward(inst2,
                                    bmm_forward_in2.as<4>(),
                                    dense_dx.as<4>(),
                                    bmm_forward_in2.addr + bmm_forward_in2.size())[0];
  inst2.Spy(path + ".bloombmm_out", bmm_dx.asVMem(inst2), Detect_Back + path + ".bloombmm_out.txt");

  int length1 = ((bmm_forward_in1.size() + 127) / 128) * 128;
  HBM_TO_VMEM(inst2, bmm_forward_in1.hbmaddr, AlignTo128Bytes(bmm_dx.addr + bmm_dx.size()), length1);
  bmm_forward_in1.addr = AlignTo128Bytes(bmm_dx.addr + bmm_dx.size());

  inst2.Spy(path + ".attention_dropout_in", bmm_dx.asVMem(inst2), Detect_Back + path + ".attention_dropout_in.txt");
  data<3> _value_layer_dx = MatMulDwBackward(inst2,
                                             bmm_forward_in1.as<4>(),
                                             dense_dx.as<4>(),
                                             AlignTo128Bytes(bmm_forward_in1.addr + bmm_forward_in1.size()))[0];
  inst2.Spy(path + ".attention_dropout_out", bmm_dx.asVMem(inst2), Detect_Back + path + ".attention_dropout_out.txt");

  data<4> softmax_forward_out;
  softmax_forward_out.hbmaddr = forward_addr.at(path + ".softmax_out");
  softmax_forward_out.dims = {backward_input.dims[0], config.n_head, backward_input.dims[1], backward_input.dims[1]};

  int length2 = ((softmax_forward_out.size() + 127) / 128) * 128;
  HBM_TO_VMEM(inst2, softmax_forward_out.hbmaddr, _value_layer_dx.addr + _value_layer_dx.size(), length2);
  softmax_forward_out.addr = _value_layer_dx.addr + _value_layer_dx.size();

  data<4> attention_probs_reshaped_dx(bmm_dx.addr, {bmm_dx.dims[0] / config.n_head, config.n_head, bmm_dx.dims[1], bmm_dx.dims[2]});

  inst2.Spy(path + ".softmax_in", attention_probs_reshaped_dx.asVMem(inst2), Detect_Back + path + ".softmax_in.txt");
  data<4> softmax_dx = SoftmaxBack(inst2, softmax_forward_out, attention_probs_reshaped_dx, AlignTo128Bytes(softmax_forward_out.addr + softmax_forward_out.size()));
  inst2.Spy(path + ".softmax_out", softmax_dx.asVMem(inst2), Detect_Back + path + ".softmax_out.txt");
 

  data<3> attention_scores_dx(softmax_dx.addr, {softmax_dx.dims[0] * softmax_dx.dims[1], softmax_dx.dims[2], softmax_dx.dims[3]});
  float inv_norm_factor = 1.0 / std::sqrt(config.hidden_size / config.n_head);
  auto inv_norm_factor_reg = inst2.AllocVReg("");
  if(1) {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(inv_norm_factor).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(inv_norm_factor).first);
    VectorOperationState move(V_U32_MOVE, 0, 0, 44, inv_norm_factor_reg.id);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  for (int i = 0; i < attention_scores_dx.size(); i += kNumberOfSubcores) {
    uint32_t data_addr = attention_scores_dx.addr + i;
    auto load_reg = inst2.AllocVReg("");
    inst2(VLoad, data_addr / (kVMemSeg * 4), load_reg.id);
    inst2(VMulF, inv_norm_factor_reg.id, load_reg.id, load_reg.id);
    inst2(VStore, load_reg.id, data_addr / (kVMemSeg * 4));
  }
  
  data<3> baddbmm_forward_1;
  baddbmm_forward_1.hbmaddr = forward_addr.at(path + ".query_layer");
  baddbmm_forward_1.dims = {config.n_head, backward_input.dims[1],config.hidden_size / config.n_head};
  std::cout << "query_layer" << std::endl;
  data<3> baddbmm_forward_2;
  baddbmm_forward_2.hbmaddr = forward_addr.at(path + ".key_layer");
  baddbmm_forward_2.dims = {config.n_head, config.hidden_size / config.n_head, backward_input.dims[1]};
  std::cout << "key_layer" << std::endl;
  HBM_TO_VMEM(inst2, baddbmm_forward_2.hbmaddr, AlignTo128Bytes(attention_scores_dx.addr + attention_scores_dx.size()), baddbmm_forward_2.size());
  baddbmm_forward_2.addr = AlignTo128Bytes(attention_scores_dx.addr + attention_scores_dx.size());
  
  data<3> _query_layer_dx = MatMulDxBackward(inst2, baddbmm_forward_2.as<4>(), attention_scores_dx.as<4>(), AlignTo128Bytes(baddbmm_forward_2.addr + baddbmm_forward_2.size()))[0];


  HBM_TO_VMEM(inst2, baddbmm_forward_1.hbmaddr, AlignTo128Bytes(_query_layer_dx.addr + _query_layer_dx.size()), baddbmm_forward_1.size());
  baddbmm_forward_1.addr = AlignTo128Bytes(_query_layer_dx.addr + _query_layer_dx.size());
  data<3> _key_layer_dx = MatMulDwBackward(inst2, baddbmm_forward_1.as<4>(), attention_scores_dx.as<4>(), AlignTo128Bytes(baddbmm_forward_1.addr + baddbmm_forward_1.size()))[0];

  data<4> query_layer_dx(_key_layer_dx.addr + _key_layer_dx.size(), {backward_input.dims[0], config.n_head, backward_input.dims[1], config.hidden_size / config.n_head});
  data<4> key_layer_dx(query_layer_dx.addr + query_layer_dx.size(), {backward_input.dims[0], config.n_head, config.hidden_size / config.n_head, backward_input.dims[1]});
  data<4> value_layer_dx(key_layer_dx.addr + key_layer_dx.size(), {backward_input.dims[0], config.n_head, backward_input.dims[1], config.hidden_size / config.n_head});
  Permute(instruction_list,
          {}, {}, {}, {},
          _query_layer_dx.addr,
          {backward_input.dims[0], config.n_head, backward_input.dims[1], config.hidden_size / config.n_head},
          {0, 2, 1, 3},
          query_layer_dx.addr);
  query_layer_dx.dims = {backward_input.dims[0], backward_input.dims[1], config.n_head, config.hidden_size / config.n_head};

  Permute(instruction_list,
          {}, {}, {}, {},
          _key_layer_dx.addr,
          {backward_input.dims[0], config.n_head, config.hidden_size / config.n_head, backward_input.dims[1]},
          {0, 3, 1, 2},
          key_layer_dx.addr);
  key_layer_dx.dims = {backward_input.dims[0], backward_input.dims[1], config.n_head, config.hidden_size / config.n_head};

  Permute(instruction_list,
          {}, {}, {}, {},
          _value_layer_dx.addr,
          {backward_input.dims[0], config.n_head, backward_input.dims[1], config.hidden_size / config.n_head},
          {0, 2, 1, 3},
          value_layer_dx.addr);
  value_layer_dx.dims = {backward_input.dims[0], backward_input.dims[1], config.n_head, config.hidden_size / config.n_head};

  for (int i = 0; i < query_layer_dx.dims[0]; i++) {
    for(int j = 0; j < query_layer_dx.dims[1]; j += kNumberOfSubcoresPerCore) {
      uint32_t use_row = std::min(kNumberOfSubcoresPerCore, int(query_layer_dx.dims[1] - j));
      for (int r = 0; r < query_layer_dx.dims[2] * query_layer_dx.dims[3]; r += query_layer_dx.dims[3]) {
        uint32_t use_col = std::min(query_layer_dx.dims[3], uint32_t(query_layer_dx.dims[2] * query_layer_dx.dims[3] - r));
        uint32_t load_offset_addr = i * query_layer_dx.dims[1] * query_layer_dx.dims[2] * query_layer_dx.dims[3] + j * query_layer_dx.dims[2] * query_layer_dx.dims[3] + r;
        uint32_t store_offset_addr = i * query_layer_dx.dims[1] * query_layer_dx.dims[2] * query_layer_dx.dims[3] * 3 + j * query_layer_dx.dims[2] * query_layer_dx.dims[3] * 3;
        auto q_reg = inst2.AllocVReg("");
        auto k_reg = inst2.AllocVReg("");
        auto v_reg = inst2.AllocVReg("");
        Load8_128(inst2, q_reg, use_row, use_col, query_layer_dx.addr + load_offset_addr, query_layer_dx.dims[2] * query_layer_dx.dims[3]);
        Store8_128(inst2, q_reg, use_row, use_col, output_addr + backward_input.size() + store_offset_addr + r * 3, query_layer_dx.dims[2] * query_layer_dx.dims[3] * 3);
        
        Load8_128(inst2, k_reg, use_row, use_col, key_layer_dx.addr + load_offset_addr, query_layer_dx.dims[2] * query_layer_dx.dims[3]);
        Store8_128(inst2, k_reg, use_row, use_col, output_addr + backward_input.size() + store_offset_addr + r * 3 + use_col, query_layer_dx.dims[2] * query_layer_dx.dims[3] * 3);
        
        Load8_128(inst2, v_reg, use_row, use_col, value_layer_dx.addr + load_offset_addr, query_layer_dx.dims[2] * query_layer_dx.dims[3]);
        Store8_128(inst2, v_reg, use_row, use_col, output_addr + backward_input.size() + store_offset_addr + r * 3 + use_col * 2, query_layer_dx.dims[2] * query_layer_dx.dims[3] * 3);
      }
    }
  }

  data<3> fuse_qkv(output_addr + backward_input.size(), {backward_input.dims[0], backward_input.dims[1], backward_input.dims[2] * 3});

  data<2> qkv_weight;
  qkv_weight.hbmaddr = weights_addr.at(path + ".query_key_value.weight");
  qkv_weight.dims = {3 * config.hidden_size, config.hidden_size};

  data<1> qkv_bias;
  qkv_bias.hbmaddr = weights_addr.at(path + ".query_key_value.bias");
  qkv_bias.dims = {3 * config.hidden_size};

  data<3> attention_forward;
  attention_forward.hbmaddr = forward_addr.at(path + ".query_key_value_in");
  attention_forward.dims = {backward_input.dims[0], backward_input.dims[1], backward_input.dims[2]};

  inst2.Spy(path + ".query_key_value_in", fuse_qkv.asVMem(inst2), Detect_Back + path + ".query_key_value_in.txt");
  data<3>  qkv_dx = matmul(inst2, fuse_qkv.as<4>(), qkv_weight, fuse_qkv.addr + fuse_qkv.size())[0];
  inst2.Spy(path + ".query_key_value_out", qkv_dx.asVMem(inst2), Detect_Back + path + ".query_key_value_out.txt");
  
  if (config.training)
  {
    HBM_TO_VMEM(instruction_list, attention_forward.hbmaddr, qkv_dx.addr + qkv_dx.size(), attention_forward.size());
    attention_forward.addr = qkv_dx.addr + qkv_dx.size();
    for (int i = 0; i < fuse_qkv.dims[0]; i++) {
      Transpose(instruction_list,
                {}, {}, {}, {},
                fuse_qkv.addr + i * fuse_qkv.dims[1] * fuse_qkv.dims[2],
                fuse_qkv.dims[1],
                fuse_qkv.dims[2],
                attention_forward.addr + attention_forward.size() + i * fuse_qkv.dims[1] * fuse_qkv.dims[2]);
    }

    data<3> fuse_qkv_T(attention_forward.addr + attention_forward.size(), {fuse_qkv.dims[0], fuse_qkv.dims[2], fuse_qkv.dims[1]});
    MatMulUpdateWeight(inst2, fuse_qkv_T, attention_forward, qkv_weight, config.update_lr, fuse_qkv_T.addr + fuse_qkv_T.size());

    data<1> qkv_db = Conv1DBackDb(inst2, fuse_qkv.as<4>(), attention_forward.addr + attention_forward.size());
    UpdateBias(inst2, config.update_lr, qkv_bias, qkv_db, qkv_db.addr + qkv_db.size());
  }

  if(qkv_dx.addr != output_addr) {
    data<3> src(output_addr, qkv_dx.dims);
    INST_TYPE inst2;
    Memcopy(qkv_dx.asVMem(inst2), src.asVMem(inst2));
    instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
  }
  qkv_dx.addr = output_addr;
  return qkv_dx;
}

data<4> BLOOMMLPBackward(INST_TYPE &inst2,
                            BLOOMConfig config,
                            data<4> hidden_states,
                            uint32_t output_addr,
                            uint32_t test_addr,
                            std::string weights_name,
                            std::map<std::string, uint32_t> weights_addr,
                            std::map<std::string, uint32_t> forward_addr)
{
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    Instruction inst;
    data<3> proj_dx;
    data<3> fc_dx;
    data<3> fc_dw;
    data<3> _output;
    data<1> _proj_bias;

    data<2> _proj_weight;
    _proj_weight.hbmaddr = weights_addr.at(weights_name + ".dense_4h_to_h.weight");
    _proj_weight.dims = {hidden_states.dims[3],  4 * hidden_states.dims[3]};

    inst2.Spy(weights_name + ".dense_4h_to_h_in", hidden_states.asVMem(inst2), Detect_Back + weights_name + ".dense_4h_to_h_in.txt");
    proj_dx = linearBackDx(inst2, hidden_states[0], _proj_weight, hidden_states.addr + hidden_states.size());
    inst2.Spy(weights_name + ".dense_4h_to_h_out", proj_dx.asVMem(inst2), Detect_Back + weights_name + ".dense_4h_to_h_out.txt");

    if (config.training)
    {
      data<3> forward_proj(proj_dx.addr + proj_dx.size(), {hidden_states.dims[1], hidden_states.dims[2], 4 * config.hidden_size});
      forward_proj.hbmaddr = forward_addr.at(weights_name + ".dense_4h_to_h_in");
      HBM_TO_VMEM(instruction_list, forward_proj.hbmaddr, forward_proj.addr, forward_proj.size());
      for (int i = 0; i < hidden_states.dims[1]; i++) {
        Transpose(instruction_list,
                  {}, {}, {}, {},
                  hidden_states.addr + i * hidden_states.dims[2] * hidden_states.dims[3],
                  hidden_states.dims[2],
                  hidden_states.dims[3],
                  forward_proj.addr + forward_proj.size() + i * hidden_states.dims[2] * hidden_states.dims[3]);
      }
      data<3> hidden_states_T(forward_proj.addr + forward_proj.size(), {hidden_states.dims[1], hidden_states.dims[3], hidden_states.dims[2]});
      MatMulUpdateWeight(inst2, hidden_states_T, forward_proj, _proj_weight, config.update_lr, hidden_states_T.addr + hidden_states_T.size());

      data<1> proj_db = Conv1DBackDb(inst2, hidden_states, proj_dx.addr + proj_dx.size());
      _proj_bias.hbmaddr = weights_addr.at(weights_name + ".dense_4h_to_h.bias");
      _proj_bias.dims = {hidden_states.dims[3]};
      UpdateBias(inst2, config.update_lr, _proj_bias, proj_db, proj_db.size() + proj_db.addr);
    }

    data<4> gelu_for(proj_dx.addr + proj_dx.size(), {1, proj_dx.dims[0], proj_dx.dims[1], proj_dx.dims[2]});
    HBM_TO_VMEM(instruction_list, forward_addr.at(weights_name + ".gelu_impl_in"), gelu_for.addr, gelu_for.size());

    inst2.Spy(weights_name + ".gelu_impl_in", proj_dx.asVMem(inst2), Detect_Back + weights_name + ".gelu_impl_in.txt");
    _output = NewGELUActivationBackward(inst2, gelu_for, proj_dx.as<4>(), gelu_for.size() + gelu_for.addr)[0];
    inst2.Spy(weights_name + ".gelu_impl_out", _output.asVMem(inst2), Detect_Back + weights_name + ".gelu_impl_out.txt");

    data<2> _fc_weight;
    _fc_weight.hbmaddr = weights_addr.at(weights_name + ".dense_h_to_4h.weight");
    _fc_weight.dims = {4 * hidden_states.dims[3], hidden_states.dims[3]};
    
    inst2.Spy(weights_name + ".dense_h_to_4h_in", _output.asVMem(inst2), Detect_Back + weights_name + ".dense_h_to_4h_in.txt");
    fc_dx = linearBackDx(inst2, _output, _fc_weight, _output.addr + _output.size());
    inst2.Spy(weights_name + ".dense_h_to_4h_out", fc_dx.asVMem(inst2), Detect_Back + weights_name + ".dense_h_to_4h_out.txt");

    if (config.training)
    {
      data<3> forward_fc(fc_dx.addr + fc_dx.size(), {hidden_states.dims[1], hidden_states.dims[2], config.hidden_size});
      forward_fc.hbmaddr = forward_addr.at(weights_name + ".dense_h_to_4h_in");
      HBM_TO_VMEM(instruction_list, forward_fc.hbmaddr, forward_fc.addr, forward_fc.size());
      for (int i = 0; i < hidden_states.dims[1]; i++) {
        Transpose(instruction_list,
                  {}, {}, {}, {},
                  _output.addr + i * _output.dims[1] * _output.dims[2],
                  _output.dims[1],
                  _output.dims[2],
                  forward_fc.addr + forward_fc.size() + i * _output.dims[1] * _output.dims[2]);
      }
      data<3> _output_T(forward_fc.addr + forward_fc.size(), {_output.dims[0], _output.dims[2], _output.dims[1]});
      MatMulUpdateWeight(inst2, _output_T, forward_fc, _fc_weight, config.update_lr, _output_T.addr + _output_T.size());


      data<1> _fc_bias;
      _fc_bias.hbmaddr = weights_addr.at(weights_name + ".dense_h_to_4h.bias");
      _fc_bias.dims = {4 * hidden_states.dims[3]};
      data<1> fc_db = Conv1DBackDb(inst2, _output.as<4>(), fc_dx.addr + fc_dx.size());
      UpdateBias(inst2, config.update_lr, _fc_bias, fc_db, fc_db.size() + fc_db.addr);
    }

    if(fc_dx.addr != output_addr) {
        data<3> src(output_addr, fc_dx.dims);
        INST_TYPE inst2;
        Memcopy(fc_dx.asVMem(inst2), src.asVMem(inst2));
        instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
    }
    fc_dx.addr = output_addr;
    return fc_dx.as<4>();
}

data<4> BLOOMBlockBackward(INST_TYPE &inst2,
                            BLOOMConfig config,
                            data<4> hidden_states,
                            uint32_t output_addr,
                            uint32_t test_addr,
                            std::string weights_name,
                            std::map<std::string, uint32_t> weights_addr,
                            std::map<std::string, uint32_t> forward_addr)
{
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    Instruction inst;

    data<4> mlp_dx(test_addr, {hidden_states.dims[0], hidden_states.dims[1], hidden_states.dims[2], hidden_states.dims[3]});
    std::cout << "block addr: " << test_addr << std::endl;
    
    inst2.Spy(weights_name + ".mlp_in", hidden_states.asVMem(inst2), Detect_Back + weights_name + ".mlp_in.txt");
    mlp_dx = BLOOMMLPBackward(inst2, config, hidden_states, mlp_dx.addr, mlp_dx.addr, weights_name + ".mlp", weights_addr, forward_addr);
    inst2.Spy(weights_name + ".mlp_out", mlp_dx.asVMem(inst2), Detect_Back + weights_name + ".mlp_out.txt");
    
    data<4> forward_in(mlp_dx.addr + mlp_dx.size(), {1, 1, hidden_states.dims[2], config.hidden_size});
    HBM_TO_VMEM(instruction_list, forward_addr.at(weights_name + ".post_attention_layernorm_in"), forward_in.addr, forward_in.size());
    data<1> forward_weight(forward_in.addr + forward_in.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".post_attention_layernorm.weight"), forward_weight.addr, forward_weight.size());

    inst2.Spy(weights_name + ".post_attention_layernorm_in", mlp_dx.asVMem(inst2), Detect_Back + weights_name + ".post_attention_layernorm_in.txt");
    data<4> pal_dx = LayerNormDxBackward(inst2, forward_in, mlp_dx, forward_weight, forward_weight.addr + forward_weight.size(), config.layer_norm_epsilon);
    inst2.Spy(weights_name + ".post_attention_layernorm_out", pal_dx.asVMem(inst2), Detect_Back + weights_name + ".post_attention_layernorm_out.txt");

    data<1> pal_dw = LayerNormDwBackward(inst2, forward_in, mlp_dx, pal_dx.addr + pal_dx.size(), config.layer_norm_epsilon);
    
    if (config.training)
    {
      data<2> forward_weight_new;
      forward_weight_new.hbmaddr = weights_addr.at(weights_name + ".post_attention_layernorm.weight");
      forward_weight_new.dims = {1, config.hidden_size};
      std::cout << "block post_attention wieght: " << forward_weight_new.hbmaddr << std::endl;
      UpdateWeight(inst2, config.update_lr, forward_weight_new, pal_dw.as<2>(), pal_dw.addr + pal_dw.size());
    }
    data<1> pal_db = LayerNormDbBackward(inst2, mlp_dx, pal_dw.addr + pal_dw.size());
    if (config.training)
    {
      data<1> forward_bias;
      forward_bias.hbmaddr = weights_addr.at(weights_name + ".post_attention_layernorm.bias");
      forward_bias.dims = {config.hidden_size};
      UpdateBias(inst2, config.update_lr, forward_bias, pal_db, pal_db.addr + pal_db.size());
    }

    pal_dx = AddVector(instruction_list, pal_dx, hidden_states, pal_dx.addr);

    inst2.Spy(weights_name + ".self_attention_in", pal_dx.asVMem(inst2), Detect_Back + weights_name + ".self_attention_in.txt");
    data<3> attn = BLOOMAttentionBackward(inst2, config, pal_dx[0], weights_name + ".self_attention", weights_addr, forward_addr, pal_db.addr + pal_db.size());
    inst2.Spy(weights_name + ".self_attention_out", attn.asVMem(inst2), Detect_Back + weights_name + ".self_attention_out.txt");


    data<4> forward_sec(attn.addr + attn.size(), {1, 1, hidden_states.dims[2], config.hidden_size});
    HBM_TO_VMEM(instruction_list, forward_addr.at(weights_name + ".input_layernorm_in"), forward_sec.addr, forward_sec.size());
    data<1> forward_sec_weight(forward_sec.addr + forward_sec.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".input_layernorm.weight"), forward_sec_weight.addr, forward_sec_weight.size());

    inst2.Spy(weights_name + ".input_layernorm_in", attn.asVMem(inst2), Detect_Back + weights_name + ".input_layernorm_in.txt");
    data<4> inl_dx = LayerNormDxBackward(inst2, forward_sec, attn.as<4>(), forward_sec_weight, output_addr, config.layer_norm_epsilon);
    inst2.Spy(weights_name + ".input_layernorm_out", inl_dx.asVMem(inst2), Detect_Back + weights_name + ".input_layernorm_out.txt");

    if (config.training)
    {
      data<2> forward_sec_weight_new;
      forward_sec_weight_new.hbmaddr = weights_addr.at(weights_name + ".input_layernorm.weight");
      forward_sec_weight_new.dims = {1, config.hidden_size};
      data<1> inl_dw = LayerNormDwBackward(inst2, forward_sec, attn.as<4>(), inl_dx.addr + inl_dx.size(), config.layer_norm_epsilon);
      UpdateWeight(inst2, config.update_lr, forward_sec_weight_new, inl_dw.as<2>(), inl_dw.addr + inl_dw.size());
      
      data<1> forward_sec_bias;
      forward_sec_bias.hbmaddr = weights_addr.at(weights_name + ".input_layernorm.bias");
      forward_sec_bias.dims = {config.hidden_size};
      data<1> inl_db = LayerNormDbBackward(inst2, attn.as<4>(), inl_dw.addr + inl_dw.size());
      UpdateBias(inst2, config.update_lr, forward_sec_bias, inl_db, inl_db.addr + inl_db.size());
    }
    inl_dx = AddVector(instruction_list, inl_dx, pal_dx, inl_dx.addr);
    if(inl_dx.addr != output_addr) {
        data<3> src(output_addr, inl_dx[0].dims);
        INST_TYPE inst2;
        Memcopy(inl_dx.asVMem(inst2), src.asVMem(inst2));
        instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
    }
    inl_dx.addr = output_addr;
    return inl_dx;
}

data<4> BLOOMModelBackward(INST_TYPE &inst2,
                            BLOOMConfig config,
                            data<4> hidden_states,
                            uint32_t output_addr,
                            uint32_t test_addr,
                            std::string weights_name,
                            std::map<std::string, uint32_t> weights_addr,
                            std::map<std::string, uint32_t> forward_addr)
{
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    Instruction inst;

    data<4> ln_f_in(test_addr, {hidden_states.dims[0], hidden_states.dims[1], hidden_states.dims[2], hidden_states.dims[3]});
    HBM_TO_VMEM(instruction_list, forward_addr.at(weights_name + ".ln_f_in"), ln_f_in.addr, ln_f_in.size());
    data<1> ln_f_weight(ln_f_in.addr + ln_f_in.size(), {1024});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".ln_f.weight"), ln_f_weight.addr, ln_f_weight.size());
    data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {1024});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".ln_f.bias"), ln_f_bias.addr, ln_f_bias.size());

    inst2.Spy(weights_name + ".ln_f_in", hidden_states.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/" + weights_name + ".ln_f_in.txt");
    data<4> ln_f_dx = LayerNormDxBackward(inst2, ln_f_in, hidden_states, ln_f_weight, ln_f_bias.addr + ln_f_bias.size(), config.layer_norm_epsilon);
    inst2.Spy(weights_name + ".ln_f_out", ln_f_dx.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/" + weights_name + ".ln_f_out.txt");
    

    data<1> ln_f_dw = LayerNormDwBackward(inst2, ln_f_in, hidden_states, ln_f_dx.addr + ln_f_dx.size(), config.layer_norm_epsilon);
    if (config.training)
    {
      data<2> ln_f_weight_new;
      ln_f_weight_new.hbmaddr = weights_addr.at(weights_name + ".ln_f.weight");
      ln_f_weight_new.dims = {1, ln_f_weight.dims[0]};
      UpdateWeight(inst2, config.update_lr, ln_f_weight_new, ln_f_dw.as<2>(), ln_f_dw.addr + ln_f_dw.size());
    }
    
    data<1> ln_f_db = LayerNormDbBackward(inst2, hidden_states, ln_f_dw.addr + ln_f_dw.size());
    if (config.training)
    {
      data<1> ln_f_bias_new;
      ln_f_bias_new.hbmaddr = weights_addr.at(weights_name + ".ln_f.bias");
      ln_f_bias_new.dims = {ln_f_bias.dims[0]};
      UpdateBias(inst2, config.update_lr, ln_f_bias_new, ln_f_db, ln_f_db.addr + ln_f_db.size());
    }
    

    int addr = ln_f_db.addr + ln_f_db.size();
    data<4> block(addr, {ln_f_dx.dims[0], ln_f_dx.dims[1], ln_f_dx.dims[2], ln_f_dx.dims[3]});
    for (int i = 23; i >= 23; i--)
    {
      if (i == 23)
        inst2.Spy(weights_name + ".h." + std::to_string(i) + "_in", ln_f_dx.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/" + weights_name + ".h." + std::to_string(i) + "_in.txt");
      else
        inst2.Spy(weights_name + ".h." + std::to_string(i) + "_in", block.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/" + weights_name + ".h." + std::to_string(i) + "_in.txt");
      block = BLOOMBlockBackward(inst2, config, ln_f_dx, addr, addr, weights_name + ".h." + std::to_string(i), weights_addr, forward_addr);
      inst2.Spy(weights_name + ".h." + std::to_string(i) + "_out", block.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/" + weights_name + ".h." + std::to_string(i) + "_out.txt");
      ln_f_dx.addr = block.addr;
      addr = block.addr + block.size();
    }



    data<4> embedding_in(block.addr + block.size(), {block.dims[0], block.dims[1], block.dims[2], block.dims[3]});
    HBM_TO_VMEM(instruction_list, forward_addr.at(weights_name + ".word_embeddings_layernorm_in"), embedding_in.addr, embedding_in.size());

    data<1> embedding_in_weight(embedding_in.addr + embedding_in.size(), {1024});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".word_embeddings_layernorm.weight"), embedding_in_weight.addr, embedding_in_weight.size());

    data<1> embedding_in_bias(embedding_in_weight.addr + embedding_in_weight.size(), {1024});
    HBM_TO_VMEM(instruction_list, weights_addr.at(weights_name + ".word_embeddings_layernorm.bias"), embedding_in_bias.addr, embedding_in_bias.size());

    data<4> embedding_dx = LayerNormDxBackward(inst2, embedding_in, block, embedding_in_weight, embedding_in_bias.addr, config.layer_norm_epsilon);
    data<1> embedding_dw = LayerNormDwBackward(inst2, embedding_in, block, embedding_dx.addr + embedding_dx.size(), config.layer_norm_epsilon);
    if (config.training)
    {
      data<2> embedding_weight_new;
      embedding_weight_new.hbmaddr = weights_addr.at(weights_name + ".word_embeddings_layernorm.weight");
      embedding_weight_new.dims = {1, embedding_in_weight.dims[0]};
      UpdateWeight(inst2, config.update_lr, embedding_weight_new, embedding_dw.as<2>(), embedding_dw.addr + embedding_dw.size());
    }

    data<1> embedding_db = LayerNormDbBackward(inst2, block, embedding_dw.addr + embedding_dx.size());
    if (config.training)
    {
      data<1> embedding_bias_new;
      embedding_bias_new.hbmaddr = weights_addr.at(weights_name + ".word_embeddings_layernorm.bias");
      embedding_bias_new.dims = {embedding_in_bias.dims[0]};
      UpdateBias(inst2, config.update_lr, embedding_bias_new, embedding_db, embedding_db.addr + embedding_db.size());
    }
    return block;
}

data<4> BLOOMForCausalLMBackward(INST_TYPE &inst2, 
                                    BLOOMConfig config,
                                    data<4> hidden_states,
                                    uint32_t output_addr,
                                    uint32_t test_addr,
                                    std::string weights_name,
                                    std::map<std::string, uint32_t> weights_addr)
{
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    Instruction *inst;
    
    hidden_states.hbmaddr = weights_addr.at("hidden_states");

    data<2> lm_head_weight;
    lm_head_weight.hbmaddr = weights_addr.at(weights_name + ".lm_head.weight");
    lm_head_weight.dims = {config.vocab_size, config.hidden_size};
    // lm_head_weight.dims = {1024, 4096};

    std::cout << "test_addr: " << test_addr << std::endl;
    data<4> lm_head = matmulIbWb(inst2, hidden_states, lm_head_weight, test_addr);

    // if (1)
    // {
    //     inst = new Instruction();
    //     VectorOperationState move(V_U32_MOVE, 0, 0, 49, 1);
    //     inst->SetOperationState(Instruction::VECTORONE, &move);
    //     CompleteInstruction(inst);
    //     instruction_list.push_back(inst);
    // }
    // uint32_t j = 1;
    // // j += 1 << 1;
    // if (1)
    // {
    //     inst = new Instruction();
    //     inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
    //     ScalarOperationState set_base(S_U32_MOVE, 0, 0, 46, 0);
    //     inst->SetOperationState(Instruction::SCALARONE, &set_base);
    //     inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
    //     inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
    //     inst->SetImmediateValue(Instruction::IMMEDIATE5, j);
    //     VectorStoreOperationState store(V_STORE_WITH_VMASK1, 0, 1, 1, 2, 4, 0, 7);
    //     inst->SetOperationState(Instruction::VECTORSTORE, &store);
    //     CompleteInstruction(inst);
    //     instruction_list.push_back(inst);
    // }

    return lm_head;
}