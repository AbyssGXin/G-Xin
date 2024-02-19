#include "torch.h"


data<4> Tanh(INST_TYPE &inst2, data<4> input, uint32_t output_addr) {
  Instruction* inst;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, input.dims);
  output.addr = output_addr;
  uint32_t input_total_size = input.size();

  for(uint32_t index = 0; index < input_total_size; index+=kNumberOfSubcores) {
    uint32_t load_addr = input.addr + index;
    uint32_t store_addr = output.addr + index;

    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState tanh(V_F32_SIGMOID, 0, 1, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &tanh);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      MTROperationState pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 3, 0);
      inst->SetOperationState(Instruction::MTR, &pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
        Instruction *instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
        instr->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
        instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        instr->SetOperationState(Instruction::SCALARONE, &set_base);
        instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 3, 1, 2, 4, 0, 0);
        instr->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
    }
  }

  return output;
}

data<4> NewGELUActivation(INST_TYPE &inst2, data<4> input, uint32_t output_addr) {
  Instruction* inst;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  data<4> temp_1 = input;
  temp_1.addr = input.addr + input.size();
  data<4> output(output_addr ,input.dims);
  const uint32_t total_size = input.size();
  //temp_value = sqrt(2.0/PI)
  const uint32_t temp_value_reg = 10; 

  //sqrt(2.0/PI)
  if(1) {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(sqrt(2.0/M_PI)).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(sqrt(2.0/M_PI)).first);
    VectorOperationState move(V_U32_MOVE, 0, 0, 44, temp_value_reg);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  //x = temp_value * (x + 0.044715 * pow(x, 3))
  for(uint32_t index = 0; index < total_size; index+= kNumberOfSubcores) {
    uint32_t load_addr = input.addr + index;
    uint32_t store_addr = temp_1.addr + index;

    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mut(V_F32_MULTIPLICATION, 0, 1, 1, 2);
      inst->SetOperationState(Instruction::VECTORONE, &mut);
      CompleteInstruction(inst);
      instruction_list.push_back(inst); 
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mut(V_F32_MULTIPLICATION, 0, 1, 2, 3);
      inst->SetOperationState(Instruction::VECTORONE, &mut);
      CompleteInstruction(inst);
      instruction_list.push_back(inst); 
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(0.044715).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(0.044715).first);
      VectorOperationState mut(V_F32_MULTIPLICATION, 0, 3, 44, 4);
      inst->SetOperationState(Instruction::VECTORONE, &mut);
      CompleteInstruction(inst);
      instruction_list.push_back(inst); 
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState add(V_F32_ADDITION, 0, 1, 4, 5);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst); 
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mut(V_F32_MULTIPLICATION, 0, temp_value_reg, 5, 6);
      inst->SetOperationState(Instruction::VECTORONE, &mut);
      CompleteInstruction(inst);
      instruction_list.push_back(inst); 
    }

    if (1) {
        Instruction *instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
        instr->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
        instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        instr->SetOperationState(Instruction::SCALARONE, &set_base);
        instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 6, 1, 2, 4, 0, 0);
        instr->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
    }
  }

  Tanh(inst2, temp_1, temp_1.addr);

  for(uint32_t index = 0; index< total_size; index+= kNumberOfSubcores) {
    uint32_t temp_addr = temp_1.addr + index;
    uint32_t input_addr = input.addr + index;
    uint32_t store_addr = output.addr + index;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(temp_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(temp_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(input_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(input_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, 50, 3);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      VectorOperationState add(V_F32_ADDITION, 0, 1, 49, 4);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, 3, 4, 5);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);     
    }

    if (1) {
      Instruction *instr = new Instruction();
      instr->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
      instr->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
      instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      instr->SetOperationState(Instruction::SCALARONE, &set_base);
      instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 5, 1, 2, 4, 0, 0);
      instr->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(instr);
      instruction_list.push_back(instr);
    }
  }

  return output;
}

data<4> matmul(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  auto core_id_reg = inst2.AllocVReg("");
  auto subcore_id_reg = inst2.AllocVReg("");
  auto zero_val_reg = inst2.AllocVReg("");

  uint32_t weights_use_vmem_size = kVectorDataMemorySize - (output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[1]);
  // std::cout << "Int: " << (int)weights_use_vmem_size << std::endl;
  // std::cout << "addr: " << kVectorDataMemorySize - (output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[1]) << std::endl;
  // std::cout << "kVectorDataMemorySize: " << kVectorDataMemorySize << std::endl;
  // std::cout << "weights_use_vmem_size: " << weights_use_vmem_size << std::endl;
  uint32_t weights_one_use_row = weights_use_vmem_size / weights.dims[1];
  uint32_t weights_dma_num = (weights.dims[0] + weights_one_use_row - 1) / weights_one_use_row;
  uint32_t weights_one_use_vmem_size = weights_one_use_row * weights.dims[1];
  uint32_t weights_VMEM_addr = AlignTo128Bytes(output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[1]);

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col;
  uint32_t min_use_weight_row;

  //get core id
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  if(1) {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(kNumberOfCores).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(kNumberOfCores).first);
    VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 1);
    inst->SetOperationState(Instruction::VECTORONE, &less);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  if(1) {
    inst = new Instruction();
    VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_val_reg.id);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  } 

  //set core value 
  //0~127
  //  .
  //  .   (8 row)
  //  .
  //0~127
  if(1) {
    std::vector<VReg> temp;

    if(1) {
      inst = new Instruction();
      VectorOperationState select_vmask1(V_SELECT_VMASK1, 0, zero_val_reg.id, core_id_reg.id, subcore_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &select_vmask1);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(uint32_t i = 0; i < 8; i++) {
      temp.push_back(inst2.AllocVReg(""));
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, subcore_id_reg.id, temp[0].id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(int i = 0; i < 7; i++) {
      if(1) {
        inst = new Instruction();
        VectorOperationState rotate(V_SUBCORE_ROTATE, 0, temp[i].id, 1, temp[i+1].id);
        inst->SetOperationState(Instruction::VECTORONE, &rotate);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_S32_ADDITION, 0, subcore_id_reg.id, temp[i+1].id, subcore_id_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }
    }
  }
  std::cout << "matmul" << std::endl;
  for(uint32_t dma_num = 0; dma_num < weights_dma_num; dma_num++) {
    uint32_t weights_HBM_addr = weights.hbmaddr + dma_num * weights_one_use_vmem_size;
    uint32_t weights_use_row = std::min(weights.dims[0] - dma_num * weights_one_use_row, weights_one_use_row);
    uint32_t weights_use_vmem_size = weights_use_row * weights.dims[1];
    // std::cout << "weights_one_use_row: " << weights_one_use_row << std::endl;
    // std::cout << "weights_use_row: " << weights_use_row << std::endl;
    // std::cout << "matmul HTV:  input_addr: " << weights_HBM_addr << ", dest_addr: " << weights_VMEM_addr << ", length: " << weights_use_vmem_size << std::endl; 
    HBM_TO_VMEM(instruction_list, weights_HBM_addr, weights_VMEM_addr, weights_use_vmem_size);
    
    // weights_use_row = 128;
    for(uint32_t input_col = 0; input_col < weights_use_row; input_col+=kNumberOfCores){
      min_use_input_col = std::min(weights_use_row - input_col, (uint32_t)kNumberOfCores);
      min_use_weight_row = min_use_input_col;
      auto left_mask = inst2.AllocVMask();

      for(uint32_t weights_col = 0; weights_col < weights.dims[1]; weights_col+=kNumberOfCores) {
        min_use_weight_col = std::min(weights.dims[1] - weights_col, (uint32_t)kNumberOfCores);
        auto right_reg = inst2.AllocVReg("");
        auto right_mask = inst2.AllocVMask();

        //load right mat and push gain
        for(uint32_t i = min_use_weight_row; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (input_col + i) * weights.dims[1] + weights_col;

            Load8_128(inst2, right_reg, one_use_row, min_use_weight_col, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
          min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
          auto left_reg = inst2.AllocVReg("");
          auto result_reg = inst2.AllocVReg("");
          auto old_result = inst2.AllocVReg("");

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load left mat
          if(1) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + dma_num * weights_one_use_row + input_col;
            Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
          }

          //Matrix multiplication
          if(1) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg.id, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          if(input_col != 0 || dma_num != 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            uint32_t store_addr = output_addr + input_row* weights.dims[1] + weights_col;
            Load8_128(inst2, old_result, min_use_input_row, min_use_weight_col, store_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result.id, result_reg.id, result_reg.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
          Store8_128(inst2, result_reg, min_use_input_row, min_use_weight_col, store_addr, weights.dims[1]);
        }
      }
    }
  
  }
  std::cout << "matmul end" << std::endl;
  return output;
}

data<4> matmulT(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[0]});

  auto subcore_id_reg = inst2.AllocVReg("");
  auto zero_val_reg = inst2.AllocVReg("");

  uint32_t weights_use_vmem_size = kVectorDataMemorySize - (output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[0]);
  uint32_t weights_one_use_col = weights_use_vmem_size / weights.dims[1];
  uint32_t weights_dma_num = (weights.dims[0] + weights_one_use_col - 1) / weights_one_use_col;
  uint32_t weights_one_use_vmem_size = weights_one_use_col * weights.dims[1];
  uint32_t weights_VMEM_addr = AlignTo128Bytes(output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[0]);

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col;
  uint32_t min_use_weight_row;

  //set core value 
  //0~127
  //  .
  //  .   (8 row)
  //  .
  //0~127
  if(1) {
    auto core_id_reg = inst2.AllocVReg("");
    std::vector<VReg> temp;

    //get core id
    if(1) {
      inst = new Instruction();
      VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(kNumberOfCores).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(kNumberOfCores).first);
      VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 1);
      inst->SetOperationState(Instruction::VECTORONE, &less);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_val_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    } 

    if(1) {
      inst = new Instruction();
      VectorOperationState select_vmask1(V_SELECT_VMASK1, 0, zero_val_reg.id, core_id_reg.id, subcore_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &select_vmask1);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(uint32_t i = 0; i < 8; i++) {
      temp.push_back(inst2.AllocVReg(""));
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, subcore_id_reg.id, temp[0].id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(int i = 0; i < 7; i++) {
      if(1) {
        inst = new Instruction();
        VectorOperationState rotate(V_SUBCORE_ROTATE, 0, temp[i].id, 1, temp[i+1].id);
        inst->SetOperationState(Instruction::VECTORONE, &rotate);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_S32_ADDITION, 0, subcore_id_reg.id, temp[i+1].id, subcore_id_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }
    }
  }

  for(uint32_t dma_num = 0; dma_num < weights_dma_num; dma_num++) {
    // if(dma_num != 1) 
    //   continue;
    uint32_t weights_HBM_addr = weights.hbmaddr + dma_num * weights_one_use_vmem_size;
    uint32_t weights_use_col = std::min(weights.dims[0] - dma_num * weights_one_use_col, weights_one_use_col);
    uint32_t weights_use_vmem_size = weights_use_col * weights.dims[1];
    // std::cout << "HBM: " << weights_HBM_addr << "  VMEM: " << weights_VMEM_addr << std::endl;
    HBM_TO_VMEM(instruction_list, weights_HBM_addr, weights_VMEM_addr, weights_use_vmem_size);

    // weights_use_row = 128;
    for(uint32_t input_col = 0; input_col < weights.dims[1]; input_col+=kNumberOfCores){
      min_use_input_col = std::min(weights.dims[1] - input_col, (uint32_t)kNumberOfCores);
      min_use_weight_row = min_use_input_col;

      for(uint32_t weights_col = 0; weights_col < weights_use_col; weights_col+=kNumberOfCores) {
        min_use_weight_col = std::min(weights_use_col - weights_col, (uint32_t)kNumberOfCores);
        auto right_reg = inst2.AllocVReg("");

        //load right mat and push gain
        for(uint32_t i = min_use_weight_col; i > 0; ) {
          uint32_t one_use_col = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_col;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (weights_col + i) * weights.dims[1] + input_col;
            Load8_128(inst2, right_reg, one_use_col, min_use_weight_row, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, right_reg.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSTF_ROUNDED, 0, 0, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
          min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
          auto left_reg = inst2.AllocVReg("");
          auto result_reg = inst2.AllocVReg("");
          auto old_result = inst2.AllocVReg("");

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load left mat
          if(1) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col;
            Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
          }

          //Matrix multiplication
          if(1) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg.id, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(input_col != 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result 
            if(1) {
              uint32_t base_addr = output_addr + input_row * weights.dims[0] + dma_num * weights_one_use_col + weights_col;
              Load8_128(inst2, old_result, min_use_input_row, min_use_weight_col, base_addr, weights.dims[0]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result.id, result_reg.id, result_reg.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          uint32_t store_addr = output_addr + input_row * weights.dims[0] + dma_num * weights_one_use_col + weights_col;
          Store8_128(inst2, result_reg, min_use_input_row, min_use_weight_col, store_addr, weights.dims[0]);
        }
      }
    }
  
  }

  return output;
}

data<4> matmulIvWv(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  auto subcore_id_reg = inst2.AllocVReg("");

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col;
  uint32_t min_use_weight_row;
 
  //set core value 
  //0~127
  //  .
  //  .   (8 row)
  //  .
  //0~127
  if(1) {
    auto core_id_reg = inst2.AllocVReg("");
    auto zero_val_reg = inst2.AllocVReg("");
    std::vector<VReg> temp;

    //get core id
    if(1) {
      inst = new Instruction();
      VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(kNumberOfCores).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(kNumberOfCores).first);
      VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 1);
      inst->SetOperationState(Instruction::VECTORONE, &less);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_val_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState select_vmask1(V_SELECT_VMASK1, 0, zero_val_reg.id, core_id_reg.id, subcore_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &select_vmask1);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(uint32_t i = 0; i < 8; i++) {
      temp.push_back(inst2.AllocVReg(""));
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, subcore_id_reg.id, temp[0].id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(int i = 0; i < 7; i++) {
      if(1) {
        inst = new Instruction();
        VectorOperationState rotate(V_SUBCORE_ROTATE, 0, temp[i].id, 1, temp[i+1].id);
        inst->SetOperationState(Instruction::VECTORONE, &rotate);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_S32_ADDITION, 0, subcore_id_reg.id, temp[i+1].id, subcore_id_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }
    }
  }

  for(uint32_t input_col = 0; input_col < weights.dims[0]; input_col+=kNumberOfCores){
    min_use_input_col = std::min( weights.dims[0] - input_col, (uint32_t)kNumberOfCores);
    min_use_weight_row = min_use_input_col;
    auto left_vmask = inst2.AllocVMask();

    for(uint32_t weights_col = 0; weights_col < weights.dims[1]; weights_col+=kNumberOfCores) {
      min_use_weight_col = std::min(weights.dims[1] - weights_col, (uint32_t)kNumberOfCores);
      auto right_reg = inst2.AllocVReg("");
      auto right_vmask = inst2.AllocVMask();

      //load right mat and push gain
      for(uint32_t i = min_use_weight_row; i > 0; ) {
        uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
        i -= one_use_row;
        uint32_t base_addr  = weights.addr + (input_col + i) * weights.dims[1] + weights_col;

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        Load8_128(inst2, right_reg, one_use_row, min_use_weight_col, base_addr, weights.dims[1]);

        if(1) {
          inst = new Instruction();
          MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &push);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      //fake mul
      if(1) {
        if(1) {
          inst = new Instruction();
          MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
          inst->SetOperationState(Instruction::MTI, &fake_mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
          inst->SetOperationState(Instruction::MTR, &pop);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      
      for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
        min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
        auto left_reg = inst2.AllocVReg("");
        auto result_reg = inst2.AllocVReg("");
        auto old_result = inst2.AllocVReg("");

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
          inst->SetOperationState(Instruction::VECTORTWO, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load left mat
        if(1) {
          uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col; 
          Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
        }

        //Matrix multiplication
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg.id, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
      
        if(input_col != 0) {
          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load old result
          if(1) {
             uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
             Load8_128(inst2, old_result, min_use_input_row, min_use_weight_col, store_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            VectorOperationState add(V_F32_ADDITION, 0, old_result.id, result_reg.id, result_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
        Store8_128(inst2, result_reg, min_use_input_row, min_use_weight_col, store_addr, weights.dims[1]);
      }
    }
  }

  return output;
}

data<4> matmulIvWv(Inst2& inst2, data<4> input, data<4> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[3]});

  auto subcore_id_reg = inst2.AllocVReg("");

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col;
  uint32_t min_use_weight_row;
 
  //set core value 
  //0~127
  //  .
  //  .   (8 row)
  //  .
  //0~127
  if(1) {
    auto core_id_reg = inst2.AllocVReg("");
    auto zero_val_reg = inst2.AllocVReg("");
    std::vector<VReg> temp;

    //get core id
    if(1) {
      inst = new Instruction();
      VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(kNumberOfCores).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(kNumberOfCores).first);
      VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 1);
      inst->SetOperationState(Instruction::VECTORONE, &less);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_val_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState select_vmask1(V_SELECT_VMASK1, 0, zero_val_reg.id, core_id_reg.id, subcore_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &select_vmask1);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(uint32_t i = 0; i < 8; i++) {
      temp.push_back(inst2.AllocVReg(""));
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, subcore_id_reg.id, temp[0].id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for(int i = 0; i < 7; i++) {
      if(1) {
        inst = new Instruction();
        VectorOperationState rotate(V_SUBCORE_ROTATE, 0, temp[i].id, 1, temp[i+1].id);
        inst->SetOperationState(Instruction::VECTORONE, &rotate);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_S32_ADDITION, 0, subcore_id_reg.id, temp[i+1].id, subcore_id_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }
    }
  }

  // std::clog << COLOR::SETSUNA << "input_addr: " << input.addr << "    weights_addr: "  << weights.addr << COLOR::WHITE << std::endl;
  for(int batch = 0; batch < input.dims[0]; batch++) {
    // std::cout << "batch: " << batch << std::endl;
    for (int seq = 0; seq < input.dims[1]; seq++)
    {
      // std::cout << "seq: " << seq << std::endl;
      for (uint32_t input_col = 0; input_col < weights.dims[2]; input_col += kNumberOfCores)
      {
        // std::cout << "input_col: " << input_col << std::endl;
        min_use_input_col = std::min(weights.dims[2] - input_col, (uint32_t)kNumberOfCores);
        min_use_weight_row = min_use_input_col;
        auto left_vmask = inst2.AllocVMask();

        for(uint32_t weights_col = 0; weights_col < weights.dims[3]; weights_col+=kNumberOfCores) {
          // std::cout << "weights_col: " << weights_col << std::endl;
          min_use_weight_col = std::min(weights.dims[3] - weights_col, (uint32_t)kNumberOfCores);
          auto right_reg = inst2.AllocVReg("");
          auto right_vmask = inst2.AllocVMask();

          //load right mat and push gain
          for(uint32_t i = min_use_weight_row; i > 0; ) {
            uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
            i -= one_use_row;
            uint32_t base_addr = weights.addr + batch * weights.dims[1] * weights.dims[2] * weights.dims[3] + seq * weights.dims[2] * weights.dims[3] + (input_col + i) * weights.dims[3] + weights_col;
            // std::clog << COLOR::YELLOW << "weights_addr: " << base_addr << "  [" << one_use_row << ", " << min_use_weight_col << "]" << COLOR::WHITE << std::endl;
            if (1)
            {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            Load8_128(inst2, right_reg, one_use_row, min_use_weight_col, base_addr, weights.dims[3]);

            if(1) {
              inst = new Instruction();
              MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &push);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          //fake mul
          if(1) {
            if(1) {
              inst = new Instruction();
              MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
              inst->SetOperationState(Instruction::MTI, &fake_mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
          
          for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
            // std::cout << "input_row: " << input_row << std::endl;
            min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
            auto left_reg = inst2.AllocVReg("");
            auto result_reg = inst2.AllocVReg("");
            auto old_result = inst2.AllocVReg("");

            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
              inst->SetOperationState(Instruction::VECTORTWO, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load left mat
            if(1) {
              uint32_t base_addr = input.addr + batch * input.dims[1] * input.dims[2] * input.dims[3] + seq * input.dims[2] * input.dims[3] + input_row * input.dims[3] + input_col; 
              // std::clog << COLOR::MIKU << "input_addr: " << base_addr << "  [" << min_use_input_row << ", " << min_use_input_col << "]" << COLOR::WHITE << std::endl;
              Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
            }

            //Matrix multiplication
            if(1) {
              if(1) {
                inst = new Instruction();
                MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
                inst->SetOperationState(Instruction::MTI, &mul);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }

              if(1) {
                inst = new Instruction();
                MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg.id, 0);
                inst->SetOperationState(Instruction::MTR, &pop);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }
            }
          
            if(input_col != 0) {
              if(1) {
                inst = new Instruction();
                VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result.id);
                inst->SetOperationState(Instruction::VECTORONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }

              //load old result
              if(1) {
                uint32_t store_addr = output_addr + batch * input.dims[1] * input.dims[2] * weights.dims[3] + seq * input.dims[2] * weights.dims[3] + input_row * weights.dims[3] + weights_col;
                // std::clog << COLOR::BLUE << "old_addr: " << store_addr << "  [" << min_use_input_row << ", " << min_use_weight_col << "]" << COLOR::WHITE << std::endl;                
                Load8_128(inst2, old_result, min_use_input_row, min_use_weight_col, store_addr, weights.dims[3]);
              }

              if(1) {
                inst = new Instruction();
                VectorOperationState add(V_F32_ADDITION, 0, old_result.id, result_reg.id, result_reg.id);
                inst->SetOperationState(Instruction::VECTORTWO, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }
            }

            uint32_t store_addr = output_addr + batch * input.dims[1] * input.dims[2] * weights.dims[3] + seq * input.dims[2] * weights.dims[3] + input_row * weights.dims[3] + weights_col;
            // std::clog << COLOR::BLUE << "output_addr: " << store_addr << "  [" << min_use_input_row << ", " << min_use_weight_col << "]" << COLOR::WHITE << std::endl;
            Store8_128(inst2, result_reg, min_use_input_row, min_use_weight_col, store_addr, weights.dims[3]);
          }
        }
      }
    }
  }
  return output;
}

data<4> matmul_v2(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  uint32_t weights_use_vmem_size = kVectorDataMemorySize - (output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[1]);
  uint32_t weights_one_use_row = weights_use_vmem_size / weights.dims[1];
  uint32_t weights_dma_num = (weights.dims[0] + weights_one_use_row - 1) / weights_one_use_row;
  uint32_t weights_one_use_vmem_size = weights_one_use_row * weights.dims[1];
  uint32_t weights_VMEM_addr = AlignTo128Bytes(output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[1]);

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col_1;
  uint32_t min_use_weight_row;
  uint32_t min_use_weight_col_2;

  for(uint32_t dma_num = 0; dma_num < weights_dma_num; dma_num++) {
    uint32_t weights_HBM_addr = weights.hbmaddr + dma_num * weights_one_use_vmem_size;
    uint32_t weights_use_row = std::min(weights.dims[0] - dma_num * weights_one_use_row, weights_one_use_row);
    uint32_t weights_use_vmem_size = weights_use_row * weights.dims[1];

    HBM_TO_VMEM(instruction_list, weights_HBM_addr, weights_VMEM_addr, weights_use_vmem_size);

    // weights_use_row = 128;
    for(uint32_t input_col = 0; input_col < weights_use_row; input_col+=kNumberOfCores){
      min_use_input_col = std::min(weights_use_row - input_col, (uint32_t)kNumberOfCores);
      min_use_weight_row = min_use_input_col;

      for(uint32_t weights_col = 0; weights_col < weights.dims[1]; weights_col+=(kNumberOfCores*2)) {
        min_use_weight_col_1 = std::min(weights.dims[1] - weights_col, (uint32_t)kNumberOfCores);
        min_use_weight_col_2 = std::min(weights.dims[1] - weights_col - min_use_weight_col_1, (uint32_t)kNumberOfCores);
        auto right_reg_1 = inst2.AllocVReg("");
        auto right_reg_2 = inst2.AllocVReg("");

        //load right mat and push gain
        for(uint32_t i = min_use_weight_row; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (input_col + i) * weights.dims[1] + weights_col;
            Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //load right mat and push gain
        for(uint32_t i = min_use_weight_row; i > 0 && min_use_weight_col_2 > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (input_col + i) * weights.dims[1] + weights_col + min_use_weight_col_1;

            Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        //fake mul
        if(min_use_weight_col_2 > 0) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
                
        for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
          min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
          auto left_reg = inst2.AllocVReg("");
          auto result_reg_1 = inst2.AllocVReg("");
          auto old_result_1 = inst2.AllocVReg("");
          auto result_reg_2 = inst2.AllocVReg("");
          auto old_result_2 = inst2.AllocVReg("");

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load left mat
          if(1) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + dma_num * weights_one_use_row + input_col;
            Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
          }

          //Matrix multiplication
          if(1) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          //Matrix multiplication
          if(min_use_weight_col_2 > 0) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 1);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
                
          if(input_col != 0 || dma_num != 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            uint32_t store_addr = output_addr + input_row* weights.dims[1] + weights_col;
            Load8_128(inst2, old_result_1, min_use_input_row, min_use_weight_col_1, store_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if((input_col != 0 || dma_num != 0) && min_use_weight_col_2 > 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            uint32_t store_addr = output_addr + input_row* weights.dims[1] + weights_col + min_use_weight_col_1;
            Load8_128(inst2, old_result_2, min_use_input_row, min_use_weight_col_2, store_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(1) {
            uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
            Store8_128(inst2, result_reg_1, min_use_input_row, min_use_weight_col_1, store_addr, weights.dims[1]);
          }

          if(min_use_weight_col_2 > 0) {
            uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col + min_use_weight_col_1;
            Store8_128(inst2, result_reg_2, min_use_input_row, min_use_weight_col_2, store_addr, weights.dims[1]);
          }
        }
      }
    }
  
  }
  return output;
}

data<4> matmulT_v2(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[0]});

  uint32_t weights_use_vmem_size = kVectorDataMemorySize - (output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[0]);
  uint32_t weights_one_use_col = weights_use_vmem_size / weights.dims[1];
  uint32_t weights_dma_num = (weights.dims[0] + weights_one_use_col - 1) / weights_one_use_col;
  uint32_t weights_one_use_vmem_size = weights_one_use_col * weights.dims[1];
  uint32_t weights_VMEM_addr = AlignTo128Bytes(output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[0]);

  uint32_t min_use_input_col_1;
  uint32_t min_use_input_row;
  uint32_t min_use_input_col_2;
  uint32_t min_use_weight_col;
  uint32_t min_use_weight_row_1;
  uint32_t min_use_weight_row_2;

  for(uint32_t dma_num = 0; dma_num < weights_dma_num; dma_num++) {
    // if(dma_num != 1) 
    //   continue;
    uint32_t weights_HBM_addr = weights.hbmaddr + dma_num * weights_one_use_vmem_size;
    uint32_t weights_use_col = std::min(weights.dims[0] - dma_num * weights_one_use_col, weights_one_use_col);
    uint32_t weights_use_vmem_size = weights_use_col * weights.dims[1];

    HBM_TO_VMEM(instruction_list, weights_HBM_addr, weights_VMEM_addr, weights_use_vmem_size);

    for(uint32_t weights_col = 0; weights_col < weights_use_col; weights_col+=kNumberOfCores) {
      min_use_weight_col = std::min(weights_use_col - weights_col, (uint32_t)kNumberOfCores);

      for(uint32_t input_col = 0; input_col < weights.dims[1]; input_col+=(kNumberOfCores*2)){
        min_use_input_col_1 = std::min(weights.dims[1] - input_col, (uint32_t)kNumberOfCores);
        min_use_weight_row_1 = min_use_input_col_1;
        min_use_input_col_2 = std::min(weights.dims[1] - input_col - min_use_input_col_1, (uint32_t)kNumberOfCores);
        min_use_weight_row_2 = min_use_input_col_2;

        auto right_reg_1 = inst2.AllocVReg("");
        auto right_reg_2 = inst2.AllocVReg("");

        //load right mat and push gain
        for(uint32_t i = min_use_weight_col; i > 0; ) {
          uint32_t one_use_col = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_col;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (weights_col + i) * weights.dims[1] + input_col;
            Load8_128(inst2, right_reg_1, one_use_col, min_use_weight_row_1, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, right_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //load right mat and push gain
        for(uint32_t i = min_use_weight_col; i > 0 && min_use_input_col_2 > 0; ) {
          uint32_t one_use_col = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_col;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (weights_col + i) * weights.dims[1] + input_col + min_use_input_col_1;
            Load8_128(inst2, right_reg_2, one_use_col, min_use_weight_row_2, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSTF_ROUNDED, 0, 0, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        //fake mul
        if(min_use_input_col_2 > 0) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSTF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
          min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
          auto left_reg_1 = inst2.AllocVReg("");
          auto left_reg_2 = inst2.AllocVReg("");
          auto result_reg_1 = inst2.AllocVReg("");
          auto old_result = inst2.AllocVReg("");
          auto result_reg_2 = inst2.AllocVReg("");

          if(1) {
            inst = new Instruction();
            VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, left_reg_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move_1);
            VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, left_reg_2.id);
            inst->SetOperationState(Instruction::VECTORTWO, &move_2);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load left mat
          if(1) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col;
            Load8_128(inst2, left_reg_1, min_use_input_row, min_use_input_col_1, base_addr, input.dims[3]);
          }

          //load left mat
          if(min_use_input_col_2 > 0) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col + min_use_input_col_1;
            Load8_128(inst2, left_reg_2, min_use_input_row, min_use_input_col_2, base_addr, input.dims[3]);
          }

          //Matrix multiplication
          if(1) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_1.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          //Matrix multiplication
          if(min_use_input_col_2 > 0) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_2.id, 0, 1);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(min_use_input_col_2 > 0) {
            inst = new Instruction();
            VectorOperationState add(V_F32_ADDITION, 0, result_reg_1.id, result_reg_2.id, result_reg_1.id);
            inst->SetOperationState(Instruction::VECTORTWO, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(input_col != 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result 
            if(1) {
              uint32_t base_addr = output_addr + input_row * weights.dims[0] + dma_num * weights_one_use_col + weights_col;
              Load8_128(inst2, old_result, min_use_input_row, min_use_weight_col, base_addr, weights.dims[0]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result.id, result_reg_1.id, result_reg_1.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          uint32_t store_addr = output_addr + input_row * weights.dims[0] + dma_num * weights_one_use_col + weights_col;
          Store8_128(inst2, result_reg_1, min_use_input_row, min_use_weight_col, store_addr, weights.dims[0]);
        }
      }
    }
  
  }

  return output;
}

data<4> matmulIvWv_v2(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col_1;
  uint32_t min_use_weight_row;
  uint32_t min_use_weight_col_2;
 
  for(uint32_t input_col = 0; input_col < weights.dims[0]; input_col+=kNumberOfCores){
    min_use_input_col = std::min( weights.dims[0] - input_col, (uint32_t)kNumberOfCores);
    min_use_weight_row = min_use_input_col;

    for(uint32_t weights_col = 0; weights_col < weights.dims[1]; weights_col+=(kNumberOfCores*2)) {
      min_use_weight_col_1 = std::min(weights.dims[1] - weights_col, (uint32_t)kNumberOfCores);
      min_use_weight_col_2 = std::min(weights.dims[1] - weights_col - min_use_weight_col_1, (uint32_t)kNumberOfCores);
      auto right_reg_1 = inst2.AllocVReg("");
      auto right_reg_2 = inst2.AllocVReg("");


      //load right mat and push gain
      for(uint32_t i = min_use_weight_row; i > 0; ) {
        uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
        i -= one_use_row;
        uint32_t base_addr  = weights.addr + (input_col + i) * weights.dims[1] + weights_col;

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

        if(1) {
          inst = new Instruction();
          MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &push);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      //load right mat and push gain
      for(uint32_t i = min_use_weight_row; i > 0 && min_use_weight_col_2 > 0; ) {
        uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
        i -= one_use_row;
        uint32_t base_addr  = weights.addr + (input_col + i) * weights.dims[1] + weights_col + min_use_weight_col_1;

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

        if(1) {
          inst = new Instruction();
          MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
          inst->SetOperationState(Instruction::MTI, &push);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }


      //fake mul
      if(1) {
        if(1) {
          inst = new Instruction();
          MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
          inst->SetOperationState(Instruction::MTI, &fake_mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
          inst->SetOperationState(Instruction::MTR, &pop);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      
      //fake mul
      if(min_use_weight_col_2 > 0) {
        if(1) {
          inst = new Instruction();
          MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
          inst->SetOperationState(Instruction::MTI, &fake_mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
          inst->SetOperationState(Instruction::MTR, &pop);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
         
      for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
        min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
        auto left_reg = inst2.AllocVReg("");
        auto result_reg_1 = inst2.AllocVReg("");
        auto old_result_1 = inst2.AllocVReg("");
        auto result_reg_2 = inst2.AllocVReg("");
        auto old_result_2 = inst2.AllocVReg("");

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
          inst->SetOperationState(Instruction::VECTORTWO, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load left mat
        if(1) {
          uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col; 
          Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
        }

        //Matrix multiplication
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
      
        //Matrix multiplication
        if(min_use_weight_col_2 > 0) {
          if(1) {
            inst = new Instruction();
            MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        if(input_col != 0) {
          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load old result
          if(1) {
             uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
             Load8_128(inst2, old_result_1, min_use_input_row, min_use_weight_col_1, store_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
            inst->SetOperationState(Instruction::VECTORTWO, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        if(input_col != 0 && min_use_weight_col_2 > 0) {
          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load old result
          if(1) {
             uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col + min_use_weight_col_1;
             Load8_128(inst2, old_result_2, min_use_input_row, min_use_weight_col_2, store_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
            inst->SetOperationState(Instruction::VECTORTWO, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        if(1) {
          uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
          Store8_128(inst2, result_reg_1, min_use_input_row, min_use_weight_col_1, store_addr, weights.dims[1]);
        }

        if(min_use_weight_col_2 > 0) {
          uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col + min_use_weight_col_1;
          Store8_128(inst2, result_reg_2, min_use_input_row, min_use_weight_col_2, store_addr, weights.dims[1]);
        }
      }
    }
  }

  return output;
}

data<4> matmul_v3(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  uint32_t weights_use_vmem_size = kVectorDataMemorySize - (output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[1]);
  uint32_t weights_one_use_row = weights_use_vmem_size / weights.dims[1];
  uint32_t weights_dma_num = (weights.dims[0] + weights_one_use_row - 1) / weights_one_use_row;
  uint32_t weights_one_use_vmem_size = weights_one_use_row * weights.dims[1];
  uint32_t weights_VMEM_addr = AlignTo128Bytes(output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[1]);

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col_1;
  uint32_t min_use_weight_row;
  uint32_t min_use_weight_col_2;
  uint32_t matmul_num_1 = 0;
  uint32_t matmul_num_2 = 0;

  for(uint32_t dma_num = 0; dma_num < weights_dma_num; dma_num++) {
    uint32_t weights_HBM_addr = weights.hbmaddr + dma_num * weights_one_use_vmem_size;
    uint32_t weights_use_row = std::min(weights.dims[0] - dma_num * weights_one_use_row, weights_one_use_row);
    uint32_t weights_use_vmem_size = weights_use_row * weights.dims[1];

    HBM_TO_VMEM(instruction_list, weights_HBM_addr, weights_VMEM_addr, weights_use_vmem_size);

    // weights_use_row = 128;
    for(uint32_t input_col = 0; input_col < weights_use_row; input_col+=kNumberOfCores){
      min_use_input_col = std::min(weights_use_row - input_col, (uint32_t)kNumberOfCores);
      min_use_weight_row = min_use_input_col;

      for(uint32_t weights_col = 0; weights_col < weights.dims[1]; weights_col+=(kNumberOfCores*2)) {
        min_use_weight_col_1 = std::min(weights.dims[1] - weights_col, (uint32_t)kNumberOfCores);
        min_use_weight_col_2 = std::min(weights.dims[1] - weights_col - min_use_weight_col_1, (uint32_t)kNumberOfCores);
        auto right_reg_1 = inst2.AllocVReg("");
        auto right_reg_2 = inst2.AllocVReg("");

        //load right mat and push gain
        for(uint32_t i = min_use_weight_row; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (input_col + i) * weights.dims[1] + weights_col;

            Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //load right mat and push gain
        for(uint32_t i = min_use_weight_row; i > 0 && min_use_weight_col_2 > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (input_col + i) * weights.dims[1] + weights_col + min_use_weight_col_1;

            Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        //fake mul
        if(min_use_weight_col_2 > 0) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
                
        for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
          min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
          auto left_reg = inst2.AllocVReg("");
          auto result_reg_1 = inst2.AllocVReg("");
          auto old_result_1 = inst2.AllocVReg("");
          auto result_reg_2 = inst2.AllocVReg("");
          auto old_result_2 = inst2.AllocVReg("");

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load left mat
          if(1) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + dma_num * weights_one_use_row + input_col;
            Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
          }

          //Matrix multiplication
          if(1) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          //Matrix multiplication
          if(min_use_weight_col_2 > 0) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 1);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
                
          if(input_col != 0 || dma_num != 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            uint32_t store_addr = output_addr + input_row* weights.dims[1] + weights_col;
            Load8_128(inst2, old_result_1, min_use_input_row, min_use_weight_col_1, store_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if((input_col != 0 || dma_num != 0) && min_use_weight_col_2 > 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            uint32_t store_addr = output_addr + input_row* weights.dims[1] + weights_col + min_use_weight_col_1;
            Load8_128(inst2, old_result_2, min_use_input_row, min_use_weight_col_2, store_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(1) {
            uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
            Store8_128(inst2, result_reg_1, min_use_input_row, min_use_weight_col_1, store_addr, weights.dims[1]);
          }

          if(min_use_weight_col_2 > 0) {
            uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col + min_use_weight_col_1;
            Store8_128(inst2, result_reg_2, min_use_input_row, min_use_weight_col_2, store_addr, weights.dims[1]);
          }
        }
      }
    }
  
  }

  return output;
}

data<4> matmulT_v3(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[0]});

  uint32_t weights_use_vmem_size = kVectorDataMemorySize - (output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[0]);
  uint32_t weights_one_use_col = weights_use_vmem_size / weights.dims[1];
  uint32_t weights_dma_num = (weights.dims[0] + weights_one_use_col - 1) / weights_one_use_col;
  uint32_t weights_one_use_vmem_size = weights_one_use_col * weights.dims[1];
  uint32_t weights_VMEM_addr = AlignTo128Bytes(output_addr + input.dims[0] * input.dims[1] * input.dims[2] * weights.dims[0]);

  uint32_t min_use_input_col_1;
  uint32_t min_use_input_row;
  uint32_t min_use_input_col_2;
  uint32_t min_use_weight_col;
  uint32_t min_use_weight_row_1;
  uint32_t min_use_weight_row_2;

  for(uint32_t dma_num = 0; dma_num < weights_dma_num; dma_num++) {
    // if(dma_num != 1) 
    //   continue;
    uint32_t weights_HBM_addr = weights.hbmaddr + dma_num * weights_one_use_vmem_size;
    uint32_t weights_use_col = std::min(weights.dims[0] - dma_num * weights_one_use_col, weights_one_use_col);
    uint32_t weights_use_vmem_size = weights_use_col * weights.dims[1];

    HBM_TO_VMEM(instruction_list, weights_HBM_addr, weights_VMEM_addr, weights_use_vmem_size);

    for(uint32_t weights_col = 0; weights_col < weights_use_col; weights_col+=kNumberOfCores) {
      min_use_weight_col = std::min(weights_use_col - weights_col, (uint32_t)kNumberOfCores);

      for(uint32_t input_col = 0; input_col < weights.dims[1]; input_col+=(kNumberOfCores*2)){
        min_use_input_col_1 = std::min(weights.dims[1] - input_col, (uint32_t)kNumberOfCores);
        min_use_weight_row_1 = min_use_input_col_1;
        min_use_input_col_2 = std::min(weights.dims[1] - input_col - min_use_input_col_1, (uint32_t)kNumberOfCores);
        min_use_weight_row_2 = min_use_input_col_2;

        auto right_reg_1 = inst2.AllocVReg("");
        auto right_reg_2 = inst2.AllocVReg("");

        //load right mat and push gain
        for(uint32_t i = min_use_weight_col; i > 0; ) {
          uint32_t one_use_col = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_col;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (weights_col + i) * weights.dims[1] + input_col;
            Load8_128(inst2, right_reg_1, one_use_col, min_use_weight_row_1, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, right_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //load right mat and push gain
        for(uint32_t i = min_use_weight_col; i > 0 && min_use_input_col_2 > 0; ) {
          uint32_t one_use_col = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_col;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load right mat
          if(1) {
            uint32_t base_addr = weights_VMEM_addr + (weights_col + i) * weights.dims[1] + input_col + min_use_input_col_1;
            Load8_128(inst2, right_reg_2, one_use_col, min_use_weight_row_2, base_addr, weights.dims[1]);
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSTF_ROUNDED, 0, 0, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        //fake mul
        if(min_use_input_col_2 > 0) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSTF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
          min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
          auto left_reg_1 = inst2.AllocVReg("");
          auto left_reg_2 = inst2.AllocVReg("");
          auto result_reg_1 = inst2.AllocVReg("");
          auto old_result = inst2.AllocVReg("");
          auto result_reg_2 = inst2.AllocVReg("");

          if(1) {
            inst = new Instruction();
            VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, left_reg_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move_1);
            VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, left_reg_2.id);
            inst->SetOperationState(Instruction::VECTORTWO, &move_2);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          //load left mat
          if(1) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col;
            Load8_128(inst2, left_reg_1, min_use_input_row, min_use_input_col_1, base_addr, input.dims[3]);
          }

          //load left mat
          if(min_use_input_col_2 > 0) {
            uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col + min_use_input_col_1;
            Load8_128(inst2, left_reg_2, min_use_input_row, min_use_input_col_2, base_addr, input.dims[3]);
          }

          //Matrix multiplication
          if(1) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_1.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          //Matrix multiplication
          if(min_use_input_col_2 > 0) {
            if(1) {
              inst = new Instruction();
              MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_2.id, 0, 1);
              inst->SetOperationState(Instruction::MTI, &mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(min_use_input_col_2 > 0) {
            inst = new Instruction();
            VectorOperationState add(V_F32_ADDITION, 0, result_reg_1.id, result_reg_2.id, result_reg_1.id);
            inst->SetOperationState(Instruction::VECTORTWO, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(input_col != 0) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result 
            if(1) {
              uint32_t base_addr = output_addr + input_row * weights.dims[0] + dma_num * weights_one_use_col + weights_col;
              Load8_128(inst2, old_result, min_use_input_row, min_use_weight_col, base_addr, weights.dims[0]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result.id, result_reg_1.id, result_reg_1.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          uint32_t store_addr = output_addr + input_row * weights.dims[0] + dma_num * weights_one_use_col + weights_col;
          Store8_128(inst2, result_reg_1, min_use_input_row, min_use_weight_col, store_addr, weights.dims[0]);
        }
      }
    }
  
  }

  return output;
}

data<4> matmulIvWv_v3_(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  uint32_t min_use_input_col;
  uint32_t min_use_input_row;
  uint32_t min_use_weight_col_1;
  uint32_t min_use_weight_row;
  uint32_t matmul_num_1 = 0;
  std::queue<uint32_t> matmul_1_store_addr;
  std::queue<uint32_t> matmul_1_store_row;
  std::queue<uint32_t> matmul_1_store_col;
  std::unordered_map<uint32_t, bool> addr_store;

 
  for(uint32_t input_col = 0; input_col < weights.dims[0]; input_col+=kNumberOfCores){
    min_use_input_col = std::min( weights.dims[0] - input_col, (uint32_t)kNumberOfCores);
    min_use_weight_row = min_use_input_col;

    for(uint32_t weights_col = 0; weights_col < weights.dims[1]; weights_col+=kNumberOfCores) {
      min_use_weight_col_1 = std::min(weights.dims[1] - weights_col, (uint32_t)kNumberOfCores);
      auto right_reg_1 = inst2.AllocVReg("");

      if(input_col == 0 && weights_col == 0) {
        //load right mat and push gain
        for(uint32_t i = min_use_weight_row; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;
          uint32_t base_addr  = weights.addr + (input_col + i) * weights.dims[1] + weights_col;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          inst = new Instruction();
          MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
          inst->SetOperationState(Instruction::MTI, &fake_mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
          inst->SetOperationState(Instruction::MTR, &pop);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      
      for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
        min_use_input_row = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
        auto left_reg = inst2.AllocVReg("");
        auto result_reg_1 = inst2.AllocVReg("");
        auto old_result_1 = inst2.AllocVReg("");

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, left_reg.id);
          inst->SetOperationState(Instruction::VECTORTWO, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col;
          matmul_1_store_addr.push(store_addr);
          matmul_1_store_row.push(min_use_input_row);
          matmul_1_store_col.push(min_use_weight_col_1);
        }

        //load left mat
        if(1) {
          uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col; 
          Load8_128(inst2, left_reg, min_use_input_row, min_use_input_col, base_addr, input.dims[3]);
        }

        //Matrix multiplication
        if(1) {
          if(input_row + min_use_input_row == input.dims[2]) {
            uint32_t _input_col = input_col;
            uint32_t _min_use_weight_col_1 = min_use_weight_col_1;
            uint32_t _weights_col = weights_col;
            uint32_t _min_use_weight_row = min_use_weight_row;

            if(weights_col + min_use_weight_col_1 < weights.dims[1]) {
              _min_use_weight_col_1 = std::min(weights.dims[1] - weights_col - min_use_weight_col_1, (uint32_t)kNumberOfCores);
              _weights_col = weights_col + min_use_weight_col_1;
            }
            else if((weights_col + min_use_weight_col_1 >= weights.dims[1]) && (input_col + min_use_input_col < weights.dims[0])) {
              _min_use_weight_col_1 = std::min(weights.dims[1], (uint32_t)kNumberOfCores);
              _input_col = (min_use_input_col + input_col) % weights.dims[0];
              _weights_col = 0;
              _min_use_weight_row = std::min( weights.dims[0] - input_col - min_use_weight_row, (uint32_t)kNumberOfCores);
            }
            else{
              _min_use_weight_col_1 = 0;
            }

            //load right mat and push gain
            for(uint32_t i = _min_use_weight_row; i > 0; ) {
              uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
              i -= one_use_row;
              uint32_t base_addr  = weights.addr + (_input_col + i) * weights.dims[1] + _weights_col;

              if(1) {
                inst = new Instruction();
                VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
                inst->SetOperationState(Instruction::VECTORONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }

              Load8_128(inst2, right_reg_1, one_use_row, _min_use_weight_col_1, base_addr, weights.dims[1]);

              if(1) {
                inst = new Instruction();
                MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
                inst->SetOperationState(Instruction::MTI, &push);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }
            }

            if(1) {
              inst = new Instruction();
              MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, left_reg.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &fake_mul);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
          else {
            inst = new Instruction();
            MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
          
          matmul_num_1++;

          if(matmul_num_1 == 15) {
            if(1) {
              inst = new Instruction();
              MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
              inst->SetOperationState(Instruction::MTR, &pop);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            if(addr_store.count(matmul_1_store_addr.front())) {
              if(1) {
                inst = new Instruction();
                VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
                inst->SetOperationState(Instruction::VECTORONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }

              //load old result
              if(1) {
                Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
              }

              if(1) {
                inst = new Instruction();
                VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
                inst->SetOperationState(Instruction::VECTORTWO, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
              }
            }
          

            Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), 
                        matmul_1_store_addr.front(), weights.dims[1]);
          
            matmul_1_store_addr.pop();
            matmul_1_store_row.pop();
            matmul_1_store_col.pop();
            matmul_num_1--;
          }
        }
      
      }
    }
  }

  if(matmul_1_store_addr.size() != 0) {
    for(;matmul_1_store_addr.size()>0;) {
      auto result_reg_1 = inst2.AllocVReg("");
      auto old_result_1 = inst2.AllocVReg("");

      if(1) {
        inst = new Instruction();
        MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
        inst->SetOperationState(Instruction::MTR, &pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(addr_store.count(matmul_1_store_addr.front())) {
        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load old result
        if(1) {
          Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
        }

        if(1) {
          inst = new Instruction();
          VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
          inst->SetOperationState(Instruction::VECTORTWO, &add);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);

      matmul_1_store_addr.pop();
      matmul_1_store_row.pop();
      matmul_1_store_col.pop();
    }
  }
        
  return output;
}

data<4> matmulIvWv_v3___(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  uint32_t min_use_input_col_1;
  uint32_t min_use_input_row_1;
  uint32_t min_use_weight_col_1;
  uint32_t min_use_weight_row_1;
  uint32_t min_use_input_col_2;
  uint32_t min_use_input_row_2;
  uint32_t min_use_weight_col_2;
  uint32_t min_use_weight_row_2;
  uint32_t matmul_num_1 = 0;
  uint32_t matmul_num_2 = 0;
  std::queue<uint32_t> matmul_1_store_addr;
  std::queue<uint32_t> matmul_2_store_addr;
  std::queue<uint32_t> matmul_1_store_row;
  std::queue<uint32_t> matmul_2_store_row;
  std::queue<uint32_t> matmul_1_store_col;
  std::queue<uint32_t> matmul_2_store_col;
  std::unordered_map<uint32_t, bool> addr_store;
  uint32_t input_col_1 = 0;
  uint32_t input_col_2 = 0;
  uint32_t weights_col_1 = 0;
  uint32_t weights_col_2 = 0;

 
  for(; input_col_1 < weights.dims[0] || input_col_2 < weights.dims[0]; ){
    min_use_input_col_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);
    min_use_weight_row_1 = min_use_input_col_1;
    min_use_input_col_2 = std::min( weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);
    min_use_weight_row_2 = min_use_input_col_2;

    auto right_reg_1 = inst2.AllocVReg("");
    auto right_reg_2 = inst2.AllocVReg("");


    if(input_col_1 == 0 && input_col_2 == 0 && weights_col_1 == 0 && weights_col_2 == 0){
      min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
      weights_col_2 = min_use_weight_col_1;
      min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
      //load right mat and push gain
      for(uint32_t i = min_use_weight_row_1; i > 0; ) {
        uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
        i -= one_use_row;
        uint32_t base_addr  = weights.addr + (input_col_1 + i) * weights.dims[1] + weights_col_1;

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

        if(1) {
          inst = new Instruction();
          MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &push);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      //fake mul
      if(1) {
        if(1) {
          inst = new Instruction();
          MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
          inst->SetOperationState(Instruction::MTI, &fake_mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
          inst->SetOperationState(Instruction::MTR, &pop);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      
      if(min_use_weight_col_2 > 0) {
        //load right mat and push gain
        for(uint32_t i = min_use_weight_row_2; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;
          uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        } 
      }
      else if(input_col_1 + min_use_input_col_1 < weights.dims[0]) {
        input_col_2 = std::min(weights.dims[0] - input_col_1 - min_use_input_col_1, (uint32_t)kNumberOfCores);
        weights_col_2 = 0;
        min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
        //load right mat and push gain
        for(uint32_t i = min_use_weight_row_2; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;
          uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        } 
      
      }
      else{
        min_use_weight_row_2 = 0;
        min_use_weight_col_2 = 0;
      }
    }

    for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
      min_use_input_row_1 = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
      auto left_reg_1 = inst2.AllocVReg("");
      auto& left_reg_2 = left_reg_1;
      auto result_reg_1 = inst2.AllocVReg("");
      auto old_result_1 = inst2.AllocVReg("");
      auto result_reg_2 = inst2.AllocVReg("");
      auto old_result_2 = inst2.AllocVReg("");

      if(1) {
        uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col_1;
        matmul_1_store_addr.push(store_addr);
        matmul_1_store_row.push(min_use_input_row_1);
        matmul_1_store_col.push(min_use_weight_col_1);
      }

      if(min_use_weight_col_2 > 0) {
        uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col_2;
        matmul_2_store_addr.push(store_addr);
        matmul_2_store_row.push(min_use_input_row_1);
        matmul_2_store_col.push(min_use_weight_col_2);
      }
    
      if(1) {
        if(1) {
          inst = new Instruction();
          VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, left_reg_1.id);
          inst->SetOperationState(Instruction::VECTORTWO, &move_1);
          VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, left_reg_2.id);
          inst->SetOperationState(Instruction::VECTORTWO, &move_2);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load left mat
        if(input_col_1 == input_col_2) {
          uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col_1; 
          Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr, input.dims[3]);
        }
        else{
          left_reg_2 =inst2.AllocVReg("");
          uint32_t base_addr_1 = input.addr + input_row * input.dims[3] + input_col_1; 
          Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr_1, input.dims[3]);
          uint32_t base_addr_2 = input.addr + input_row * input.dims[3] + input_col_2; 
          Load8_128(inst2, left_reg_2, min_use_input_row_1, min_use_input_col_2, base_addr_2, input.dims[3]);
        }
      }

      //Matrix multiplication
      if(min_use_weight_col_1 > 0 && min_use_weight_row_1 > 0) {
        if(input_row + min_use_input_row_1 == input.dims[2]) {
          if(weights_col_1 + min_use_weight_col_1*2 <  weights.dims[1]) {
            weights_col_1 = weights_col_1 + min_use_weight_col_1*2;
            min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
          }
          else if((weights_col_1 + min_use_weight_col_1*2 >= weights.dims[1]) && (input_col_1 + min_use_input_col_1 < weights.dims[0])) {
            input_col_1 = min_use_input_col_1 + input_col_1;
            min_use_weight_row_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);

            if(input_col_1 == input_col_2) {
              if(weights_col_2 + min_use_weight_col_2 < weights.dims[1]){
                weights_col_1 = weights_col_2 + min_use_weight_col_2;
                min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
              }
              else if(input_col_1 + min_use_weight_row_1 < weights.dims[0]) {
                input_col_1 = min_use_input_col_1 + input_col_1;
                min_use_weight_row_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);
                weights_col_1 = 0;
                min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
              }
              else{
                input_col_1 += min_use_weight_row_1;
                min_use_weight_row_1 = 0;
                min_use_weight_col_1 = 0;
              }
            }
            else{
              weights_col_1 = 0;
              min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
            }
          }
          else{
            input_col_1 += min_use_weight_row_1;
            min_use_weight_row_1 = 0;
            min_use_weight_col_1 = 0;
          }

          //load right mat and push gain
          for(uint32_t i = min_use_weight_row_1; i > 0 && min_use_weight_col_1 > 0; ) {
            uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
            i -= one_use_row;
            uint32_t base_addr  = weights.addr + (input_col_1 + i) * weights.dims[1] + weights_col_1;

            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &push);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, left_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        else {
          inst = new Instruction();
          MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_1.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        
        matmul_num_1++;

        if(matmul_num_1 == 2) {
          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(addr_store.count(matmul_1_store_addr.front())) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            if(1) {
              Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          // Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
        
          addr_store[matmul_1_store_addr.front()] = true;
          matmul_1_store_addr.pop();
          matmul_1_store_row.pop();
          matmul_1_store_col.pop();
          matmul_num_1--;
        }
      }
    
      //Matrix multiplication
      if(min_use_weight_col_2 > 0 && min_use_weight_row_2 > 0) {
        if(input_row + min_use_input_row_1 == input.dims[2]) {
          if(weights_col_2 + min_use_weight_col_2*2 <  weights.dims[1]) {
            weights_col_2 = weights_col_2 + min_use_weight_col_2*2;
            min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
          }
          else if((weights_col_2 + min_use_weight_col_2*2 >= weights.dims[1]) && (input_col_2 + min_use_input_col_2 < weights.dims[0])) {
            input_col_2 = min_use_input_col_2 + input_col_2;
            min_use_weight_row_2 = std::min( weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);

            if(input_col_1 == input_col_2) {
              if(weights_col_1 + min_use_weight_col_1 < weights.dims[1]){
                weights_col_2 = weights_col_1 + min_use_weight_col_1;
                min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
              }
              else if(input_col_2 + min_use_weight_row_2 < weights.dims[0]) {
                input_col_2 = min_use_input_col_2 + input_col_2;
                min_use_weight_row_2 = std::min(weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);
                weights_col_2 = 0;
                min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
              }
              else{
                input_col_2 += min_use_weight_row_2;
                min_use_weight_row_2 = 0;
                min_use_weight_col_2 = 0;
              }
            }
            else{
              weights_col_2 = 0;
              min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
            }
          }
          else{
            input_col_2 += min_use_weight_row_2;
            min_use_weight_row_2 = 0;
            min_use_weight_col_2 = 0;
          }

          //load right mat and push gain
          for(uint32_t i = min_use_weight_row_2; i > 0 && min_use_weight_col_2 > 0; ) {
            uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
            i -= one_use_row;
            uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
              inst->SetOperationState(Instruction::MTI, &push);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
          
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, left_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        else {
          inst = new Instruction();
          MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_2.id, 0, 1);
          inst->SetOperationState(Instruction::MTI, &mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        
        matmul_num_2++;

        if(matmul_num_2 == 2) {
          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(addr_store.count(matmul_2_store_addr.front())) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            if(1) {
              Load8_128(inst2, old_result_2, matmul_2_store_row.front(), matmul_2_store_col.front(),matmul_2_store_addr.front(), weights.dims[1]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          // Store8_128(inst2, result_reg_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
        
          addr_store[matmul_2_store_addr.front()] = true;
          matmul_2_store_addr.pop();
          matmul_2_store_row.pop();
          matmul_2_store_col.pop();
          matmul_num_2--;
        } 
      }
    
      if(0) {
        if(1) {
          inst = new Instruction();
          VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, left_reg_1.id);
          inst->SetOperationState(Instruction::VECTORTWO, &move_1);
          VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, left_reg_2.id);
          inst->SetOperationState(Instruction::VECTORTWO, &move_2);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        
        //load left mat
        if(input_col_1 == input_col_2) {
          uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col_1; 
          Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr, input.dims[3]);
        }
        else{
          left_reg_2 =inst2.AllocVReg("");
          uint32_t base_addr_1 = input.addr + input_row * input.dims[3] + input_col_1; 
          Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr_1, input.dims[3]);
          uint32_t base_addr_2 = input.addr + input_row * input.dims[3] + input_col_2; 
          Load8_128(inst2, left_reg_2, min_use_input_row_1, min_use_input_col_2, base_addr_2, input.dims[3]);
        }
      }
    }

  }

  if(matmul_1_store_addr.size() < 0) {
    for(;matmul_1_store_addr.size()>0;) {
      auto result_reg_1 = inst2.AllocVReg("");
      auto old_result_1 = inst2.AllocVReg("");

      if(1) {
        inst = new Instruction();
        MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
        inst->SetOperationState(Instruction::MTR, &pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(addr_store.count(matmul_1_store_addr.front())) {
        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load old result
        if(1) {
          Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
        }

        if(1) {
          inst = new Instruction();
          VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
          inst->SetOperationState(Instruction::VECTORTWO, &add);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);

      addr_store[matmul_1_store_addr.front()] = true;
      matmul_1_store_addr.pop();
      matmul_1_store_row.pop();
      matmul_1_store_col.pop();
    }
  }
        
  if(matmul_2_store_addr.size() < 0) {
    for(; matmul_2_store_addr.size() > 0; ) {
      auto result_reg_2 = inst2.AllocVReg("");
      auto old_result_2 = inst2.AllocVReg("");

      if(1) {
        inst = new Instruction();
        MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
        inst->SetOperationState(Instruction::MTR, &pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(addr_store.count(matmul_2_store_addr.front())) {
        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load old result
        if(1) {
          Load8_128(inst2, old_result_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
        }

        if(1) {
          inst = new Instruction();
          VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
          inst->SetOperationState(Instruction::VECTORTWO, &add);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
    
      Store8_128(inst2, result_reg_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
    
      addr_store[matmul_2_store_addr.front()] = true;
      matmul_2_store_addr.pop();
      matmul_2_store_row.pop();
      matmul_2_store_col.pop();
    }
  } 
        
  return output;
}

data<4> matmulIvWv_v3__(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  uint32_t min_use_input_col_1;
  uint32_t min_use_input_row_1;
  uint32_t min_use_weight_col_1;
  uint32_t min_use_weight_row_1;
  uint32_t min_use_input_col_2;
  uint32_t min_use_input_row_2;
  uint32_t min_use_weight_col_2;
  uint32_t min_use_weight_row_2;
  uint32_t matmul_num_1 = 0;
  uint32_t matmul_num_2 = 0;
  std::queue<uint32_t> matmul_1_store_addr;
  std::queue<uint32_t> matmul_2_store_addr;
  std::queue<uint32_t> matmul_1_store_row;
  std::queue<uint32_t> matmul_2_store_row;
  std::queue<uint32_t> matmul_1_store_col;
  std::queue<uint32_t> matmul_2_store_col;
  std::unordered_map<uint32_t, bool> addr_store;
  uint32_t input_col_1 = 0;
  uint32_t input_col_2 = 0;
  uint32_t weights_col_1 = 0;
  uint32_t weights_col_2 = 0;

 
  for(; input_col_1 < weights.dims[0] || input_col_2 < weights.dims[0];){
    min_use_input_col_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);
    min_use_weight_row_1 = min_use_input_col_1;
    min_use_input_col_2 = std::min( weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);
    min_use_weight_row_2 = min_use_input_col_2;

    auto right_reg_1 = inst2.AllocVReg("");
    auto right_reg_2 = inst2.AllocVReg("");


    if(input_col_1 == 0 && input_col_2 == 0 && weights_col_1 == 0 && weights_col_2 == 0){
      min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
      weights_col_2 = min_use_weight_col_1;
      min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
      //load right mat and push gain
      for(uint32_t i = min_use_weight_row_1; i > 0; ) {
        uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
        i -= one_use_row;
        uint32_t base_addr  = weights.addr + (input_col_1 + i) * weights.dims[1] + weights_col_1;

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

        if(1) {
          inst = new Instruction();
          MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &push);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      //fake mul
      if(1) {
        if(1) {
          inst = new Instruction();
          MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
          inst->SetOperationState(Instruction::MTI, &fake_mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
          inst->SetOperationState(Instruction::MTR, &pop);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      
      if(min_use_weight_col_2 > 0) {
        //load right mat and push gain
        for(uint32_t i = min_use_weight_row_2; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;
          uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        } 
      }
      else if(input_col_1 + min_use_input_col_1 < weights.dims[0]) {
        input_col_2 = std::min(weights.dims[0] - input_col_1 - min_use_input_col_1, (uint32_t)kNumberOfCores);
        weights_col_2 = 0;
        min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
        //load right mat and push gain
        for(uint32_t i = min_use_weight_row_2; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;
          uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        } 
      
      }
      else{
        min_use_weight_row_2 = 0;
        min_use_weight_col_2 = 0;
      }
    }

    for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
      min_use_input_row_1 = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
      auto left_reg_1 = inst2.AllocVReg("");
      auto& left_reg_2 = left_reg_1;
      auto result_reg_1 = inst2.AllocVReg("");
      auto old_result_1 = inst2.AllocVReg("");
      auto result_reg_2 = inst2.AllocVReg("");
      auto old_result_2 = inst2.AllocVReg("");

      if(1) {
        inst = new Instruction();
        VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, left_reg_1.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move_1);
        VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, left_reg_2.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move_2);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col_1;
        matmul_1_store_addr.push(store_addr);
        matmul_1_store_row.push(min_use_input_row_1);
        matmul_1_store_col.push(min_use_weight_col_1);
      }

      if(min_use_weight_col_2 > 0) {
        uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col_2;
        matmul_2_store_addr.push(store_addr);
        matmul_2_store_row.push(min_use_input_row_1);
        matmul_2_store_col.push(min_use_weight_col_2);
      }
    

      //load left mat
      if(input_col_1 == input_col_2) {
        uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col_1; 
        Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr, input.dims[3]);
      }
      else{
        left_reg_2 =inst2.AllocVReg("");
        uint32_t base_addr_1 = input.addr + input_row * input.dims[3] + input_col_1; 
        Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr_1, input.dims[3]);
        uint32_t base_addr_2 = input.addr + input_row * input.dims[3] + input_col_2; 
        Load8_128(inst2, left_reg_2, min_use_input_row_1, min_use_input_col_2, base_addr_2, input.dims[3]);
      }

      //Matrix multiplication
      if(min_use_weight_col_1 > 0 && min_use_weight_row_1 > 0) {
        if(input_row + min_use_input_row_1 == input.dims[2]) {
          if(weights_col_1 + min_use_weight_col_1*2 <  weights.dims[1]) {
            weights_col_1 = weights_col_1 + min_use_weight_col_1*2;
            min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
          }
          else if((weights_col_1 + min_use_weight_col_1*2 >= weights.dims[1]) && (input_col_1 + min_use_input_col_1 < weights.dims[0])) {
            input_col_1 = min_use_input_col_1 + input_col_1;
            min_use_weight_row_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);

            if(input_col_1 == input_col_2) {
              if(weights_col_2 + min_use_weight_col_2 < weights.dims[1]){
                weights_col_1 = weights_col_2 + min_use_weight_col_2;
                min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
              }
              else if(input_col_1 + min_use_weight_row_1 < weights.dims[0]) {
                input_col_1 = min_use_input_col_1 + input_col_1;
                min_use_weight_row_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);
                weights_col_1 = 0;
                min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
              }
              else{
                input_col_1 += min_use_weight_row_1;
                min_use_weight_row_1 = 0;
                min_use_weight_col_1 = 0;
              }
            }
            else{
              weights_col_1 = 0;
              min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
            }
          }
          else{
            input_col_1 += min_use_weight_row_1;
            min_use_weight_row_1 = 0;
            min_use_weight_col_1 = 0;
          }

          //load right mat and push gain
          for(uint32_t i = min_use_weight_row_1; i > 0 && min_use_weight_col_1 > 0; ) {
            uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
            i -= one_use_row;
            uint32_t base_addr  = weights.addr + (input_col_1 + i) * weights.dims[1] + weights_col_1;

            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &push);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, left_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        else {
          inst = new Instruction();
          MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_1.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        
        matmul_num_1++;

        if(matmul_num_1 == 2) {
          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(addr_store.count(matmul_1_store_addr.front())) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            if(1) {
              Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
        
          addr_store[matmul_1_store_addr.front()] = true;
          matmul_1_store_addr.pop();
          matmul_1_store_row.pop();
          matmul_1_store_col.pop();
          matmul_num_1--;
        }
      }
    
      //Matrix multiplication
      if(min_use_weight_col_2 > 0 && min_use_weight_row_2 > 0) {
        if(input_row + min_use_input_row_1 == input.dims[2]) {
          if(weights_col_2 + min_use_weight_col_2*2 <  weights.dims[1]) {
            weights_col_2 = weights_col_2 + min_use_weight_col_2*2;
            min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
          }
          else if((weights_col_2 + min_use_weight_col_2*2 >= weights.dims[1]) && (input_col_2 + min_use_input_col_2 < weights.dims[0])) {
            input_col_2 = min_use_input_col_2 + input_col_2;
            min_use_weight_row_2 = std::min( weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);

            if(input_col_1 == input_col_2) {
              if(weights_col_1 + min_use_weight_col_1 < weights.dims[1]){
                weights_col_2 = weights_col_1 + min_use_weight_col_1;
                min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
              }
              else if(input_col_2 + min_use_weight_row_2 < weights.dims[0]) {
                input_col_2 = min_use_input_col_2 + input_col_2;
                min_use_weight_row_2 = std::min(weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);
                weights_col_2 = 0;
                min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
              }
              else{
                input_col_2 += min_use_weight_row_2;
                min_use_weight_row_2 = 0;
                min_use_weight_col_2 = 0;
              }
            }
            else{
              weights_col_2 = 0;
              min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
            }
          }
          else{
            input_col_2 += min_use_weight_row_2;
            min_use_weight_row_2 = 0;
            min_use_weight_col_2 = 0;
          }

          //load right mat and push gain
          for(uint32_t i = min_use_weight_row_2; i > 0 && min_use_weight_col_2 > 0; ) {
            uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
            i -= one_use_row;
            uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
              inst->SetOperationState(Instruction::MTI, &push);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
          
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, left_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        else {
          inst = new Instruction();
          MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_2.id, 0, 1);
          inst->SetOperationState(Instruction::MTI, &mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        
        matmul_num_2++;

        if(matmul_num_2 == 2) {
          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(addr_store.count(matmul_2_store_addr.front())) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            if(1) {
              Load8_128(inst2, old_result_2, matmul_2_store_row.front(), matmul_2_store_col.front(),matmul_2_store_addr.front(), weights.dims[1]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          Store8_128(inst2, result_reg_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
        
          addr_store[matmul_2_store_addr.front()] = true;
          matmul_2_store_addr.pop();
          matmul_2_store_row.pop();
          matmul_2_store_col.pop();
          matmul_num_2--;
        } 
      }
    
    }

  }

  if(matmul_1_store_addr.size() != 0) {
    for(;matmul_1_store_addr.size()>0;) {
      auto result_reg_1 = inst2.AllocVReg("");
      auto old_result_1 = inst2.AllocVReg("");

      if(1) {
        inst = new Instruction();
        MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
        inst->SetOperationState(Instruction::MTR, &pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(addr_store.count(matmul_1_store_addr.front())) {
        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load old result
        if(1) {
          Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
        }

        if(1) {
          inst = new Instruction();
          VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
          inst->SetOperationState(Instruction::VECTORTWO, &add);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);

      addr_store[matmul_1_store_addr.front()] = true;
      matmul_1_store_addr.pop();
      matmul_1_store_row.pop();
      matmul_1_store_col.pop();
    }
  }
        
  if(matmul_2_store_addr.size() != 0) {
    for(; matmul_2_store_addr.size() > 0; ) {
      auto result_reg_2 = inst2.AllocVReg("");
      auto old_result_2 = inst2.AllocVReg("");

      if(1) {
        inst = new Instruction();
        MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
        inst->SetOperationState(Instruction::MTR, &pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(addr_store.count(matmul_2_store_addr.front())) {
        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        //load old result
        if(1) {
          Load8_128(inst2, old_result_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
        }

        if(1) {
          inst = new Instruction();
          VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
          inst->SetOperationState(Instruction::VECTORTWO, &add);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
    
      Store8_128(inst2, result_reg_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
    
      addr_store[matmul_2_store_addr.front()] = true;
      matmul_2_store_addr.pop();
      matmul_2_store_row.pop();
      matmul_2_store_col.pop();
    }
  } 
        
  return output;
}

data<4> matmulIvWv_v3(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  uint32_t min_use_input_col_1;
  uint32_t min_use_input_row_1;
  uint32_t min_use_weight_col_1;
  uint32_t min_use_weight_row_1;
  uint32_t min_use_input_col_2;
  uint32_t min_use_input_row_2;
  uint32_t min_use_weight_col_2;
  uint32_t min_use_weight_row_2;
  uint32_t matmul_num_1 = 0;
  uint32_t matmul_num_2 = 0;
  std::queue<uint32_t> matmul_1_store_addr;
  std::queue<uint32_t> matmul_2_store_addr;
  std::queue<uint32_t> matmul_1_store_row;
  std::queue<uint32_t> matmul_2_store_row;
  std::queue<uint32_t> matmul_1_store_col;
  std::queue<uint32_t> matmul_2_store_col;
  std::unordered_map<uint32_t, bool> addr_store;
  uint32_t input_col_1 = 0;
  uint32_t input_col_2 = 0;
  uint32_t weights_col_1 = 0;
  uint32_t weights_col_2 = 0;

 
  for(; input_col_1 < weights.dims[0] || input_col_2 < weights.dims[0];){
    min_use_input_col_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);
    min_use_weight_row_1 = min_use_input_col_1;
    min_use_input_col_2 = std::min( weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);
    min_use_weight_row_2 = min_use_input_col_2;

    auto right_reg_1 = inst2.AllocVReg("");
    auto right_reg_2 = inst2.AllocVReg("");

    if(input_col_1 == 0 && input_col_2 == 0 && weights_col_1 == 0 && weights_col_2 == 0){
      min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
      weights_col_2 = min_use_weight_col_1;
      min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
      //load right mat and push gain
      for(uint32_t i = min_use_weight_row_1; i > 0; ) {
        uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
        i -= one_use_row;
        uint32_t base_addr  = weights.addr + (input_col_1 + i) * weights.dims[1] + weights_col_1;

        if(1) {
          inst = new Instruction();
          VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
          inst->SetOperationState(Instruction::VECTORONE, &move);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

        if(1) {
          inst = new Instruction();
          MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &push);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      //fake mul
      if(1) {
        if(1) {
          inst = new Instruction();
          MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 0);
          inst->SetOperationState(Instruction::MTI, &fake_mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 0);
          inst->SetOperationState(Instruction::MTR, &pop);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      
      if(min_use_weight_col_2 > 0) {
        //load right mat and push gain
        for(uint32_t i = min_use_weight_row_2; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;
          uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        } 
      }
      else if(input_col_1 + min_use_input_col_1 < weights.dims[0]) {
        input_col_2 = std::min(weights.dims[0] - input_col_1 - min_use_input_col_1, (uint32_t)kNumberOfCores);
        weights_col_2 = 0;
        min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
        //load right mat and push gain
        for(uint32_t i = min_use_weight_row_2; i > 0; ) {
          uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
          i -= one_use_row;
          uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

          if(1) {
            inst = new Instruction();
            VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
            inst->SetOperationState(Instruction::VECTORONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

          if(1) {
            inst = new Instruction();
            MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &push);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }

        //fake mul
        if(1) {
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, 0, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, 0, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        } 
      
      }
      else{
        min_use_weight_row_2 = 0;
        min_use_weight_col_2 = 0;
      }
    }
    
    for(uint32_t input_row = 0; input_row < input.dims[2]; input_row+=kNumberOfSubcoresPerCore) {
      min_use_input_row_1 = std::min(input.dims[2] - input_row, (uint32_t)kNumberOfSubcoresPerCore);
      auto left_reg_1 = inst2.AllocVReg("");
      auto& left_reg_2 = left_reg_1;
      auto result_reg_1 = inst2.AllocVReg("");
      auto old_result_1 = inst2.AllocVReg("");
      auto result_reg_2 = inst2.AllocVReg("");
      auto old_result_2 = inst2.AllocVReg("");

      if(1) {
        inst = new Instruction();
        VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, left_reg_1.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move_1);
        VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, left_reg_2.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move_2);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(min_use_weight_col_1 > 0 && min_use_weight_row_1 > 0) {
        uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col_1;
        matmul_1_store_addr.push(store_addr);
        matmul_1_store_row.push(min_use_input_row_1);
        matmul_1_store_col.push(min_use_weight_col_1);
      }

      if(min_use_weight_col_2 > 0 && min_use_weight_row_2 > 0) {
        uint32_t store_addr = output_addr + input_row * weights.dims[1] + weights_col_2;
        matmul_2_store_addr.push(store_addr);
        matmul_2_store_row.push(min_use_input_row_1);
        matmul_2_store_col.push(min_use_weight_col_2);
      }
    

      //load left mat
      if(input_col_1 == input_col_2) {
        uint32_t base_addr = input.addr + input_row * input.dims[3] + input_col_1; 
        Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr, input.dims[3]);
      }
      else{
        left_reg_2 =inst2.AllocVReg("");
        uint32_t base_addr_1 = input.addr + input_row * input.dims[3] + input_col_1; 
        Load8_128(inst2, left_reg_1, min_use_input_row_1, min_use_input_col_1, base_addr_1, input.dims[3]);
        uint32_t base_addr_2 = input.addr + input_row * input.dims[3] + input_col_2; 
        Load8_128(inst2, left_reg_2, min_use_input_row_1, min_use_input_col_2, base_addr_2, input.dims[3]);
      }

      //Matrix multiplication
      if(min_use_weight_col_1 > 0 && min_use_weight_row_1 > 0) {
        if(input_row + min_use_input_row_1 == input.dims[2]) {
          if(weights_col_1 + min_use_weight_col_1*2 <  weights.dims[1]) {
            weights_col_1 = weights_col_1 + min_use_weight_col_1*2;
            min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
          }
          else if((weights_col_1 + min_use_weight_col_1*2 >= weights.dims[1]) && (input_col_1 + min_use_input_col_1 < weights.dims[0])) {
            input_col_1 = min_use_input_col_1 + input_col_1;
            min_use_weight_row_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);

            if(input_col_1 == input_col_2) {
              if(weights_col_2 + min_use_weight_col_2 < weights.dims[1]){
                weights_col_1 = weights_col_2 + min_use_weight_col_2;
                min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
              }
              else if(input_col_1 + min_use_weight_row_1 < weights.dims[0]) {
                input_col_1 = min_use_input_col_1 + input_col_1;
                min_use_weight_row_1 = std::min( weights.dims[0] - input_col_1, (uint32_t)kNumberOfCores);
                weights_col_1 = 0;
                min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
              }
              else{
                input_col_1 += min_use_weight_row_1;
                min_use_weight_row_1 = 0;
                min_use_weight_col_1 = 0;
              }
            }
            else{
              weights_col_1 = 0;
              min_use_weight_col_1 = std::min(weights.dims[1] - weights_col_1, (uint32_t)kNumberOfCores);
            }
          }
          else{
            input_col_1 += min_use_weight_row_1;
            min_use_weight_row_1 = 0;
            min_use_weight_col_1 = 0;
          }

          //load right mat and push gain
          for(uint32_t i = min_use_weight_row_1; i > 0 && min_use_weight_col_1 > 0; ) {
            uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
            i -= one_use_row;
            uint32_t base_addr  = weights.addr + (input_col_1 + i) * weights.dims[1] + weights_col_1;

            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            Load8_128(inst2, right_reg_1, one_use_row, min_use_weight_col_1, base_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_1.id, 0, 0);
              inst->SetOperationState(Instruction::MTI, &push);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }

          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, left_reg_1.id, 0, 0);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        else {
          inst = new Instruction();
          MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_1.id, 0, 0);
          inst->SetOperationState(Instruction::MTI, &mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        
        matmul_num_1++;

        if(matmul_num_1 == 1) {
          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(addr_store.count(matmul_1_store_addr.front())) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            if(1) {
              Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
        
          addr_store[matmul_1_store_addr.front()] = true;
          if(matmul_1_store_addr.empty())
            std::cout << "matmul_1_store_addr.empty\n";
          matmul_1_store_addr.pop();
          matmul_1_store_row.pop();
          matmul_1_store_col.pop();
          matmul_num_1--;
        }
      }

      //Matrix multiplication
      if(min_use_weight_col_2 > 0 && min_use_weight_row_2 > 0) {
        if(input_row + min_use_input_row_1 == input.dims[2]) {
          if(weights_col_2 + min_use_weight_col_2*2 <  weights.dims[1]) {
            weights_col_2 = weights_col_2 + min_use_weight_col_2*2;
            min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
          }
          else if((weights_col_2 + min_use_weight_col_2*2 >= weights.dims[1]) && (input_col_2 + min_use_input_col_2 < weights.dims[0])) {
            input_col_2 = min_use_input_col_2 + input_col_2;
            min_use_weight_row_2 = std::min( weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);

            if(input_col_1 == input_col_2) {
              if(weights_col_1 + min_use_weight_col_1 < weights.dims[1]){
                weights_col_2 = weights_col_1 + min_use_weight_col_1;
                min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
              }
              else if(input_col_2 + min_use_weight_row_2 < weights.dims[0]) {
                input_col_2 = min_use_input_col_2 + input_col_2;
                min_use_weight_row_2 = std::min(weights.dims[0] - input_col_2, (uint32_t)kNumberOfCores);
                weights_col_2 = 0;
                min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
              }
              else{
                input_col_2 += min_use_weight_row_2;
                min_use_weight_row_2 = 0;
                min_use_weight_col_2 = 0;
              }
            }
            else{
              weights_col_2 = 0;
              min_use_weight_col_2 = std::min(weights.dims[1] - weights_col_2, (uint32_t)kNumberOfCores);
            }
          }
          else{
            input_col_2 += min_use_weight_row_2;
            min_use_weight_row_2 = 0;
            min_use_weight_col_2 = 0;
          }

          //load right mat and push gain
          for(uint32_t i = min_use_weight_row_2; i > 0 && min_use_weight_col_2 > 0; ) {
            uint32_t one_use_row = i % kNumberOfSubcoresPerCore == 0 ? kNumberOfSubcoresPerCore : i % kNumberOfSubcoresPerCore;
            i -= one_use_row;
            uint32_t base_addr  = weights.addr + (input_col_2 + i) * weights.dims[1] + weights_col_2;

            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, right_reg_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            Load8_128(inst2, right_reg_2, one_use_row, min_use_weight_col_2, base_addr, weights.dims[1]);

            if(1) {
              inst = new Instruction();
              MTIOperationState push(MTI_PUSHGAIN_FLOAT_ROUNDED, 0, right_reg_2.id, 0, 1);
              inst->SetOperationState(Instruction::MTI, &push);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
          
          if(1) {
            inst = new Instruction();
            MTIOperationState fake_mul(MTI_MUL_GSNF_ROUNDED, 0, left_reg_2.id, 0, 1);
            inst->SetOperationState(Instruction::MTI, &fake_mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        else {
          inst = new Instruction();
          MTIOperationState mul(MTI_MUL_FLOAT_ROUNDED, 0, left_reg_2.id, 0, 1);
          inst->SetOperationState(Instruction::MTI, &mul);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        
        matmul_num_2++;

        if(matmul_num_2 == 1) {
          if(1) {
            inst = new Instruction();
            MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
            inst->SetOperationState(Instruction::MTR, &pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(addr_store.count(matmul_2_store_addr.front())) {
            if(1) {
              inst = new Instruction();
              VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
              inst->SetOperationState(Instruction::VECTORONE, &move);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }

            //load old result
            if(1) {
              Load8_128(inst2, old_result_2, matmul_2_store_row.front(), matmul_2_store_col.front(),matmul_2_store_addr.front(), weights.dims[1]);
            }

            if(1) {
              inst = new Instruction();
              VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
              inst->SetOperationState(Instruction::VECTORTWO, &add);
              CompleteInstruction(inst);
              instruction_list.push_back(inst);
            }
          }
        
          Store8_128(inst2, result_reg_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
        
          addr_store[matmul_2_store_addr.front()] = true;
          if(matmul_2_store_addr.empty()) 
            std::cout << "matmul_2_store_addr.empty\n";
          matmul_2_store_addr.pop();
          matmul_2_store_row.pop();
          matmul_2_store_col.pop();
          matmul_num_2--;
        } 
      }
    }

  }

  // if(matmul_1_store_addr.size() != 0) {
  //   for(;matmul_1_store_addr.size()>0;) {
  //     auto result_reg_1 = inst2.AllocVReg("");
  //     auto old_result_1 = inst2.AllocVReg("");

  //     if(1) {
  //       inst = new Instruction();
  //       MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_1.id, 0);
  //       inst->SetOperationState(Instruction::MTR, &pop);
  //       CompleteInstruction(inst);
  //       instruction_list.push_back(inst);
  //     }

  //     if(addr_store.count(matmul_1_store_addr.front())) {
  //       if(1) {
  //         inst = new Instruction();
  //         VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_1.id);
  //         inst->SetOperationState(Instruction::VECTORONE, &move);
  //         CompleteInstruction(inst);
  //         instruction_list.push_back(inst);
  //       }

  //       //load old result
  //       if(1) {
  //         Load8_128(inst2, old_result_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);
  //       }

  //       if(1) {
  //         inst = new Instruction();
  //         VectorOperationState add(V_F32_ADDITION, 0, old_result_1.id, result_reg_1.id, result_reg_1.id);
  //         inst->SetOperationState(Instruction::VECTORTWO, &add);
  //         CompleteInstruction(inst);
  //         instruction_list.push_back(inst);
  //       }
  //     }

  //     // Store8_128(inst2, result_reg_1, matmul_1_store_row.front(), matmul_1_store_col.front(), matmul_1_store_addr.front(), weights.dims[1]);

  //     addr_store[matmul_1_store_addr.front()] = true;
  //     // matmul_1_store_addr.pop();
  //     matmul_1_store_row.pop();
  //     matmul_1_store_col.pop();
  //   }
  // }
        
  // if(matmul_2_store_addr.size() != 0) {
  //   for(; matmul_2_store_addr.size() > 0; ) {
  //     auto result_reg_2 = inst2.AllocVReg("");
  //     auto old_result_2 = inst2.AllocVReg("");

  //     if(1) {
  //       inst = new Instruction();
  //       MTROperationState pop(MTR_READ_MATRIX_RESULT, 0, result_reg_2.id, 1);
  //       inst->SetOperationState(Instruction::MTR, &pop);
  //       CompleteInstruction(inst);
  //       instruction_list.push_back(inst);
  //     }

  //     if(addr_store.count(matmul_2_store_addr.front())) {
  //       if(1) {
  //         inst = new Instruction();
  //         VectorOperationState move(V_U32_MOVE, 0, 0, 46, old_result_2.id);
  //         inst->SetOperationState(Instruction::VECTORONE, &move);
  //         CompleteInstruction(inst);
  //         instruction_list.push_back(inst);
  //       }

  //       //load old result
  //       if(1) {
  //         Load8_128(inst2, old_result_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
  //       }

  //       if(1) {
  //         inst = new Instruction();
  //         VectorOperationState add(V_F32_ADDITION, 0, old_result_2.id, result_reg_2.id, result_reg_2.id);
  //         inst->SetOperationState(Instruction::VECTORTWO, &add);
  //         CompleteInstruction(inst);
  //         instruction_list.push_back(inst);
  //       }
  //     }
    
  //     // Store8_128(inst2, result_reg_2, matmul_2_store_row.front(), matmul_2_store_col.front(), matmul_2_store_addr.front(), weights.dims[1]);
    
  //     addr_store[matmul_2_store_addr.front()] = true;
  //     matmul_2_store_addr.pop();
  //     matmul_2_store_row.pop();
  //     matmul_2_store_col.pop();
  //   }
  // } 
        
  return output;
}

data<4> matmulIbWb(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr)
{
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {input.dims[0], input.dims[1], input.dims[2], weights.dims[1]});

  assert(input.dims[3] == weights.dims[0]);

  uint32_t input_hbmaddr = input.hbmaddr;
  uint32_t weights_hbmaddr = weights.hbmaddr;

  std::cout << "input_hbmaddr: " << input.hbmaddr << std::endl;
  std::cout << "weights_hbmaddr: " << weights_hbmaddr << std::endl;

  uint32_t free_vmem = kVectorDataMemorySize - (output_addr + output.size());
  std::cout << "Available vmem: " << free_vmem << std::endl;
  uint32_t common = free_vmem / (input.dims[2] + weights.dims[1]);
  std::cout << "Available common: " << common << std::endl;
  common = (common & (~127));
  
  std::cout << "used common: " << common << std::endl;
  uint32_t use_common = common;
  uint32_t common_end = input.dims[3] % common;
  uint32_t split_num = input.dims[3] / common;


  uint32_t input_hbmaddr_new = input_hbmaddr;
  uint32_t input_addr_start = output.addr + output.size();
  uint32_t input_addr = input_addr_start;
  uint32_t weights_hbmaddr_new = weights_hbmaddr;
  uint32_t weights_addr = input_addr;
  
  for (uint32_t i = 0; i <= split_num; i++)
  {
    if (i == split_num)
    {
      use_common = common_end;
      std::cout << "use_common: " << use_common << std::endl;
    }
    /*
    for (uint32_t j = 0; j < input.dims[2]; j++)
    {
      HBM_TO_VMEM(instruction_list, input_hbmaddr_new, input_addr, use_common);
      std::cout << "use input_addr: " << input_addr << std::endl;
      std::cout << "use input_hbmaddr_new: " << input_hbmaddr_new << std::endl;
      input_addr += use_common;
      input_hbmaddr_new += input.dims[3];
    }
    */
    // uint32_t numsplit = use_common / 128;
    for (uint32_t j = 0; j < input.dims[2]; j++)
    {
      // HBM_TO_VMEM_Stride(instruction_list, input_hbmaddr_new, input_addr, 128 * input.dims[2], input.dims[3] / 128, use_common / 128);
      // HBM_TO_VMEM_Stride(instruction_list, input_hbmaddr_new, input_addr, input.dims[3] * input.dims[2], input.dims[3] / 128, use_common / 128);
      HBM_TO_VMEM(instruction_list, input_hbmaddr_new, input_addr, use_common);
      std::cout << "use input_addr: " << input_addr << std::endl;
      std::cout << "use input_hbmaddr_new: " << input_hbmaddr_new << std::endl;
      std::cout << "use_common: " << use_common << std::endl;
      input_addr += use_common;
      input_hbmaddr_new += input.dims[3];
    }
    weights_addr = input_addr_start + use_common * input.dims[2];
    // HBM_TO_VMEM(instruction_list, weights_hbmaddr_new, weights_addr, use_common * weights.dims[1]);
    // std::cout << "use weights_addr: " << weights_addr << std::endl;
    // std::cout << "use weights_hbmaddr_new: " << weights_hbmaddr_new << std::endl;
    // weights_hbmaddr_new += use_common * weights.dims[1];
    input_hbmaddr_new = input_hbmaddr + (i + 1) * common;
    input_addr = input_addr_start;

    data<4> inputsplit(input_addr, {1, 1, input.dims[2], use_common});
    // data<2> weightsplit(weights_addr, {use_common, weights.dims[1]});
    data<2> weightsplit;
    weightsplit.hbmaddr = weights_hbmaddr_new;
    weightsplit.dims = {use_common, weights.dims[1]};

    // data<4> outputsplit(output.addr + output.size(), {1, 1, 41, 4096});
    if (i == 0)
    {
      data<4> outputsplit = matmul(inst2, inputsplit, weightsplit, inputsplit.addr + inputsplit.size());
      if (output_addr != outputsplit.addr)
      {
        data<4> src(output_addr, outputsplit.dims);
        INST_TYPE inst2;
        Memcopy(outputsplit.asVMem(inst2), src.asVMem(inst2));
        instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
      }
    } else {
      data<4> outputsplit = matmul(inst2, inputsplit, weightsplit, inputsplit.addr + inputsplit.size());
      AddVector(instruction_list, outputsplit, output, output.addr);
    }
    weights_hbmaddr_new += use_common * weights.dims[1];
  }


  return output;
}

data<4> Addmm(INST_TYPE &inst2,
              const data<4> &hidden_states,
              uint32_t hbmWeightAddr,
              const std::array<uint32_t, 2> &weightSize,
              uint32_t hbmBiasAddr,
              uint32_t biasSize,
              uint32_t output_addr,
              uint32_t beta,
              uint32_t alpha)
{
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  auto alignTo128 = [](uint32_t addr) { return ((addr + 127) / 128) * 128; };
  data<4> output(output_addr, {hidden_states.dims[0], hidden_states.dims[1], hidden_states.dims[2], weightSize[1]});
  uint32_t usable_addr = alignTo128(output_addr + output.size());
  const std::array<uint32_t, 4> inputSize = {hidden_states.dims[0],
                                             hidden_states.dims[1],
                                             hidden_states.dims[2],
                                             hidden_states.dims[3]};

  // LinearExVMemIO(instruction_list,
  //                {},{},{},{},
  //                128,
  //                128,
  //                hidden_states.addr,
  //                inputSize,
  //                hbmWeightAddr,
  //                weightSize,
  //                output.addr);

  data<2> weights;
  weights.dims = weightSize;
  weights.hbmaddr = hbmWeightAddr;
  matmul(inst2, hidden_states, weights, output.addr);

  data<1> bias(usable_addr, {biasSize});
  usable_addr += alignTo128(bias.size());
  HBM_TO_VMEM(instruction_list, hbmBiasAddr, bias.addr, biasSize);

  data<4> extend_bias(usable_addr, {output.dims[0], output.dims[1], output.dims[2], biasSize});
  usable_addr += alignTo128(extend_bias.size());

  for(int i = 0; i < extend_bias.dims[0]; i++)
  {
    for(int j = 0; j < extend_bias.dims[1]; j++)
    {
      for(int k = 0; k < extend_bias.dims[2]; k++)
      {
        Memcopy(bias.asVMem(inst2), extend_bias[i][j][k].asVMem(inst2));
      }
    }
  }
  uint32_t bias_reg = 1;
  uint32_t output_reg = 2;
  for(uint32_t i = output.addr, j = extend_bias.addr; 
               i < output.addr + output.size(),
               j < extend_bias.addr + extend_bias.size();
               i+=kNumberOfSubcores, j+=kNumberOfSubcores)
  {
    if(1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(i / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(i / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::IMMEDIATE3, 1);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, output_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if(1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0 * alpha).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0 * alpha).first);
      VectorOperationState mul1(V_F32_MULTIPLICATION, 0, output_reg, 44, output_reg);
      inst->SetOperationState(Instruction::VECTORONE, &mul1);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if(1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(j / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(j / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::IMMEDIATE3, 1);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, bias_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if(1) 
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0 * beta).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0 * beta).first);
      VectorOperationState mul1(V_F32_MULTIPLICATION, 0, bias_reg, 44, bias_reg);
      inst->SetOperationState(Instruction::VECTORONE, &mul1);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      VectorOperationState add(V_F32_ADDITION, 0, output_reg, bias_reg, output_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if(1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(i / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(i / kVMemSeg).first);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, output_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  return output;
}

data<4> Conv1D(INST_TYPE &inst2,
               const data<3> &hidden_states,
               const data<2> weight,
               const data<1> bias,
               uint32_t output_addr)
{
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> input;
  input.dims = {1, hidden_states.dims[0], hidden_states.dims[1], hidden_states.dims[2]};
  input.addr = hidden_states.addr;
  assert(weight.hbmaddr != -1);
  assert(bias.hbmaddr != -1);
  data<4> output = Addmm(inst2,
                         input,
                         weight.hbmaddr,
                         weight.dims,
                         bias.hbmaddr,
                         bias.dims[0],
                         output_addr);

  return output;
}

data<3> LayerNorm_spare(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps) {
  uint32_t weights_addr = weights.addr;
  uint32_t bias_addr = bias.addr;
  const uint32_t sum_reg = 10;
  const uint32_t mean_reg = 11;
  const uint32_t var_reg = 12;
  const uint32_t core_id_reg = 13;
  std::vector<uint32_t> weights_reg;
  std::vector<uint32_t> bias_reg;
  const uint32_t ch_num = (input.dims[2] + 1024)/1024;
  uint32_t min_load_size;
  data<3> output(output_addr, input.dims);
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  // get core id
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }


  // load weights and bias to reg
  for(uint32_t i = 0; i < input.dims[2]; i += kNumberOfSubcores) {
    min_load_size = std::min(input.dims[2] - i, uint32_t(kNumberOfSubcores));
    uint32_t _weight_addr = weights_addr + i;
    uint32_t _bias_addr = bias_addr + i;
    uint32_t _weights_reg = core_id_reg + 1 + i / kNumberOfSubcores;
    uint32_t _bias_reg = core_id_reg + ch_num + 1 + i / kNumberOfSubcores;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_weight_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_weight_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _weights_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_bias_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_bias_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _bias_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }   

    weights_reg.push_back(_weights_reg);
    bias_reg.push_back(_bias_reg); 
  }

  for(uint32_t i = 0; i < input.dims[0]*input.dims[1]; i++) {
    // reset sum_reg and var_reg
    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_reg);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    // sum += x
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t input_addr = input.addr + i * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      /**/
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        //set 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 1);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 1, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_reg, 1, sum_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    // sum/ch
    { 
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(input.dims[2]).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }       

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_reg, 1, mean_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }    

    // var += pow(x-mear, 2);
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t load_addr = input.addr + i * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 1);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 1, mean_reg, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, 2, 3);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 4);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState select_vm0(V_SELECT_VMASK0, 0, 4, 3, 4);
        inst->SetOperationState(Instruction::VECTORONE, &select_vm0);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 4, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 4, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 4, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_reg, 4, var_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
  
    // var/ch
    if(1) { 
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(input.dims[2]).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }       

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_reg, 1, var_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);          
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, 1, var_reg, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, 2, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst); 
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_reg, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }   
    }   
  
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t ch_num = j / 1024;
      uint32_t load_addr = input.addr + i * input.dims[2] + j;
      uint32_t store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + i * input.dims[2] + j;

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 1, mean_reg, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, var_reg, 3);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 3, weights_reg[ch_num], 4);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, 4, bias_reg[ch_num], 5);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      
      if(1) {
        Instruction *instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
        instr->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
        instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        instr->SetOperationState(Instruction::SCALARONE, &set_base);
        instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 5, 1, 2, 4, 0, 0);
        instr->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }      
    }
  }

  if(bias_addr + input.dims[2] != output_addr) {
    data<3> src(bias_addr+input.dims[2], input.dims);
    Memcopy(src.asVMem(inst2), output.asVMem(inst2));
  }

  return output;
}

data<3> LayerNorm_V1(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps) {
  uint32_t weights_addr = weights.addr;
  uint32_t bias_addr = bias.addr;
  const uint32_t sum_1_reg = 10;
  const uint32_t mean_1_reg = 11;
  const uint32_t var_1_reg = 12;
  const uint32_t sum_2_reg = 13;
  const uint32_t mean_2_reg = 14;
  const uint32_t var_2_reg = 15;
  const uint32_t core_id_reg = 16;
  std::vector<uint32_t> weights_reg;
  std::vector<uint32_t> bias_reg;
  const uint32_t ch_num = (input.dims[2] + 1024)/1024;
  uint32_t min_load_size;
  data<3> output(output_addr, input.dims);
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  // get core id
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }


  // load weights and bias to reg
  for(uint32_t i = 0; i < input.dims[2]; i += kNumberOfSubcores) {
    min_load_size = std::min(input.dims[2] - i, uint32_t(kNumberOfSubcores));
    uint32_t _weight_addr = weights_addr + i;
    uint32_t _bias_addr = bias_addr + i;
    uint32_t _weights_reg = core_id_reg + 1 + i / kNumberOfSubcores;
    uint32_t _bias_reg = core_id_reg + ch_num + 1 + i / kNumberOfSubcores;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_weight_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_weight_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _weights_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_bias_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_bias_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _bias_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }   

    weights_reg.push_back(_weights_reg);
    bias_reg.push_back(_bias_reg); 
  }

  for(uint32_t i = 0; i < input.dims[0]*input.dims[1]; i+=2) {
    // reset sum_reg and var_reg
    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_1_reg);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_1_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_2_reg);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_2_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    // sum += x
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t btach_1_input_addr = input.addr + i * input.dims[2] + j;
      uint32_t btach_2_input_addr = input.addr + (i+1) * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      /**/
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        //set reg_1 = 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 1);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) {
        inst = new Instruction();
        // set reg_2 = 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 2);
        inst->SetOperationState(Instruction::VECTORONE, &move);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(btach_1_input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(btach_1_input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(btach_2_input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(btach_2_input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 2, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 2, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      } 

      /**/
      if(1){
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 1, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 2, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 2, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_1_reg, 1, sum_1_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_2_reg, 2, sum_2_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    // sum/ch
    { 
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(input.dims[2]).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }       

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_1_reg, 1, mean_1_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_2_reg, 1, mean_2_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }    

    // var += pow(x-mear, 2);
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t batch_1_load_addr = input.addr + i * input.dims[2] + j;
      uint32_t batch_2_load_addr = input.addr + (i+1) * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_1_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_1_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 1, mean_1_reg, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_2_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_2_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 5, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 5, mean_2_reg, 5);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, 2, 3);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 4);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 5, 5, 6);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 7);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState select_vm0(V_SELECT_VMASK0, 0, 4, 3, 4);
        inst->SetOperationState(Instruction::VECTORONE, &select_vm0);
        VectorOperationState select_vm1(V_SELECT_VMASK0, 0, 7, 6, 7);
        inst->SetOperationState(Instruction::VECTORTWO, &select_vm1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 4, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 7, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      /**/
      if(1){
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 4, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 7, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 7, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 4, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 7, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 7, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_1_reg, 4, var_1_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 7, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_2_reg, 7, var_2_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
  
    // var/ch
    if(1){
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(input.dims[2]).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }       

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_1_reg, 1, var_1_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &move);

        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_2_reg, 1, var_2_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);          
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, 2, var_1_reg, 3);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, 3, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);

        VectorOperationState add(V_F32_ADDITION, 0, 2, var_2_reg, 4);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst); 
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_1_reg, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );

        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }   

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_2_reg, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }  
    }   
  
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t ch_num = j / 1024;
      uint32_t batch_1_load_addr = input.addr + i * input.dims[2] + j;
      uint32_t batch_2_load_addr = input.addr + (i+1) * input.dims[2] + j;
      uint32_t batch_1_store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + i * input.dims[2] + j;
      uint32_t batch_2_store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + (i+1) * input.dims[2] + j;

      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_1_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_1_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 1, mean_1_reg, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_2_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_2_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 6, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, var_1_reg, 3);
        inst->SetOperationState(Instruction::VECTORONE, &mul);

        VectorOperationState sub(V_F32_SUBTRACTION, 0, 6, mean_2_reg, 7);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 7, var_2_reg, 8);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 3, weights_reg[ch_num], 4);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, 4, bias_reg[ch_num], 5);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 8, weights_reg[ch_num], 9);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_1_store_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_1_store_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, 5, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);

        VectorOperationState add(V_F32_ADDITION, 0, 9, bias_reg[ch_num], 9);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }  

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_2_store_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_2_store_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, 9, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }        
    }
  }

  if(bias_addr + input.dims[2] != output_addr) {
    data<3> src(bias_addr+input.dims[2], input.dims);
    Memcopy(src.asVMem(inst2), output.asVMem(inst2));
    instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
  }

  return output;
}

data<3> LayerNorm_V2(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps) {
  uint32_t weights_addr = weights.addr;
  uint32_t bias_addr = bias.addr;
  const uint32_t sum_1_reg = 12;
  const uint32_t mean_1_reg = 13;
  const uint32_t var_1_reg = 14;
  const uint32_t sum_2_reg = 15;
  const uint32_t mean_2_reg = 16;
  const uint32_t var_2_reg = 17;
  const uint32_t sum_3_reg = 18;
  const uint32_t mean_3_reg = 19;
  const uint32_t var_3_reg = 20;
  const uint32_t sum_4_reg = 21;
  const uint32_t mean_4_reg = 22;
  const uint32_t var_4_reg = 23;
  const uint32_t core_id_reg = 24;
  std::vector<uint32_t> weights_reg;
  std::vector<uint32_t> bias_reg;
  const uint32_t ch_num = (input.dims[2] + 1024)/1024;
  const uint32_t ch_total_size = input.dims[0]*input.dims[1];
  uint32_t min_load_size;
  data<3> output(output_addr, input.dims);
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  // get core id
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  // load weights and bias to reg
  for(uint32_t i = 0; i < input.dims[2]; i += kNumberOfSubcores) {
    min_load_size = std::min(input.dims[2] - i, uint32_t(kNumberOfSubcores));
    uint32_t _weight_addr = weights_addr + i;
    uint32_t _bias_addr = bias_addr + i;
    uint32_t _weights_reg = core_id_reg + 1 + i / kNumberOfSubcores;
    uint32_t _bias_reg = core_id_reg + ch_num + 1 + i / kNumberOfSubcores;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_weight_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_weight_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _weights_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_bias_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_bias_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _bias_reg, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }   

    weights_reg.push_back(_weights_reg);
    bias_reg.push_back(_bias_reg); 
  }

  for(uint32_t i = 0; i < input.dims[0]*input.dims[1]; i+=4) {
    // reset sum_reg and var_reg
    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_1_reg);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_1_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_2_reg);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_2_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_3_reg);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_3_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_4_reg);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_4_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    // sum += x
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t btach_1_input_addr = input.addr + i * input.dims[2] + j;
      uint32_t btach_2_input_addr = input.addr + (i+1) * input.dims[2] + j;
      uint32_t btach_3_input_addr = input.addr + (i+2) * input.dims[2] + j;
      uint32_t btach_4_input_addr = input.addr + (i+3) * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      /**/
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        //set reg_1 = 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 1);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) {
        inst = new Instruction();
        // set reg = 0
        VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, 2);
        inst->SetOperationState(Instruction::VECTORONE, &move_1);
        VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, 3);
        inst->SetOperationState(Instruction::VECTORTWO, &move_2);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(btach_1_input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(btach_1_input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (i+2 < ch_total_size) {
        inst = new Instruction();
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 5);
        inst->SetOperationState(Instruction::VECTORONE, &move);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(btach_3_input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(btach_3_input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 3, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(btach_2_input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(btach_2_input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 2, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < ch_total_size) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 3, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(btach_4_input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(btach_4_input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 5, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 2, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      } 

      if(i+2 < ch_total_size){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 3, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 5, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      } 

      /**/
      if(1){
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 1, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < ch_total_size){
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 3, 4, 1);
        inst->SetOperationState(Instruction::MTI, &transpose);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 5, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 2, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(i+2 < ch_total_size){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 3, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 5, 4, 1);
        inst->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(i+2 < ch_total_size) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 3, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 5, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 2, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(i+2 < ch_total_size){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 3, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 5, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_1_reg, 1, sum_1_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(i+2 < ch_total_size) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_3_reg, 3, sum_3_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 5, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_2_reg, 2, sum_2_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(i+3 < ch_total_size) {
        inst = new Instruction();
        VectorOperationState move(V_U32_MOVE, 0, 0, 5, 4);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(i+3 < ch_total_size) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_4_reg, 4, sum_4_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    // sum/ch
    if(1) { 
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(input.dims[2]).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }       

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_1_reg, 1, mean_1_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_2_reg, 1, mean_2_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_3_reg, 1, mean_3_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_4_reg, 1, mean_4_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }    

    // var += pow(x-mear, 2);
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t batch_1_load_addr = input.addr + i * input.dims[2] + j;
      uint32_t batch_2_load_addr = input.addr + (i+1) * input.dims[2] + j;
      uint32_t batch_3_load_addr = input.addr + (i+2) * input.dims[2] + j;
      uint32_t batch_4_load_addr = input.addr + (i+3) * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_1_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_1_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 1, mean_1_reg, 1);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_2_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_2_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 2, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_1_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_1_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 3, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 3, mean_3_reg, 3);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_2_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_2_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, 4, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 2, mean_2_reg, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 4, mean_4_reg, 4);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 1, 1, 1);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 5);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, 2, 2);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 6);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 3, 3, 3);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 7);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 4, 4, 4);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 8);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState select_vm0(V_SELECT_VMASK0, 0, 4, 3, 4);
        inst->SetOperationState(Instruction::VECTORONE, &select_vm0);
        VectorOperationState select_vm1(V_SELECT_VMASK0, 0, 7, 6, 7);
        inst->SetOperationState(Instruction::VECTORTWO, &select_vm1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState select_vm0(V_SELECT_VMASK0, 0, 3, 7, 3);
        inst->SetOperationState(Instruction::VECTORONE, &select_vm0);
        VectorOperationState select_vm1(V_SELECT_VMASK0, 0, 4, 8, 4);
        inst->SetOperationState(Instruction::VECTORTWO, &select_vm1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 3, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 2, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(i+2 < input.dims[0]*input.dims[1]){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 3, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 4, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      /**/
      if(1){
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 1, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]){
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 3, 4, 1);
        inst->SetOperationState(Instruction::MTI, &transpose);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 2, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(i+2 < input.dims[0]*input.dims[1]){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 3, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, 4, 4, 1);
        inst->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 3, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 2, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(i+2 < input.dims[0]*input.dims[1]){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 3, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);

        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, 4, 0, 1);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_1_reg, 1, var_1_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 2, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_3_reg, 3, var_3_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);

        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, 4, 1);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_2_reg, 2, var_2_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_4_reg, 4, var_4_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
  
    // var/ch
    if(1){
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(input.dims[2]).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }       

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_1_reg, 1, var_1_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 44, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &move);

        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_2_reg, 1, var_2_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);          
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_3_reg, 1, var_3_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_4_reg, 1, var_4_reg);
        inst->SetOperationState(Instruction::VECTORONE, &mul);

        VectorOperationState add(V_F32_ADDITION, 0, 2, var_1_reg, var_1_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, var_1_reg, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);

        VectorOperationState add(V_F32_ADDITION, 0, 2, var_2_reg, var_2_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst); 
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, 2, var_3_reg, var_3_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, var_3_reg, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);

        VectorOperationState add(V_F32_ADDITION, 0, 2, var_4_reg, var_4_reg);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst); 
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_1_reg, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );

        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, var_2_reg, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }   

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_3_reg, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );

        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, var_4_reg, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }  

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_2_reg, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      } 

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_4_reg, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }   
    }   
  
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t ch_num = j / 1024;
      uint32_t batch_1_load_addr = input.addr + i * input.dims[2] + j;
      uint32_t batch_2_load_addr = input.addr + (i+1) * input.dims[2] + j;
      uint32_t batch_3_load_addr = input.addr + (i+2) * input.dims[2] + j;
      uint32_t batch_4_load_addr = input.addr + (i+3) * input.dims[2] + j;
      uint32_t batch_1_store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + i * input.dims[2] + j;
      uint32_t batch_2_store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + (i+1) * input.dims[2] + j;
      uint32_t batch_3_store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + (i+2) * input.dims[2] + j;
      uint32_t batch_4_store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + (i+3) * input.dims[2] + j;

      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_1_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_1_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 1, mean_1_reg, 1);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_2_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_2_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 2, mean_2_reg, 2);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_3_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_3_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 3, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 1, var_1_reg, 1);
        inst->SetOperationState(Instruction::VECTORONE, &mul);

        //sub inst x and y can't eq dest
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 3, mean_3_reg, 5);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_4_load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_4_load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 4, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);   
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, var_2_reg, 2);
        inst->SetOperationState(Instruction::VECTORONE, &mul);

        VectorOperationState sub(V_F32_SUBTRACTION, 0, 4, mean_4_reg, 4);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 5, var_3_reg, 3);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 4, var_4_reg, 4);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 1, weights_reg[ch_num], 1);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, 1, bias_reg[ch_num], 1);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, weights_reg[ch_num], 2);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_1_store_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_1_store_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, 1, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);

        VectorOperationState add(V_F32_ADDITION, 0, 2, bias_reg[ch_num], 2);
        inst->SetOperationState(Instruction::VECTORTWO, &add);

        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 3, weights_reg[ch_num], 3);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }  

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_2_store_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_2_store_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, 2, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);

        VectorOperationState add(V_F32_ADDITION, 0, 3, bias_reg[ch_num], 3);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 4, weights_reg[ch_num], 4);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_3_store_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_3_store_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, 3, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);

        VectorOperationState add(V_F32_ADDITION, 0, 4, bias_reg[ch_num], 4);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      } 

      if(i+2 < input.dims[0]*input.dims[1]) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(batch_4_store_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(batch_4_store_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, 4, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }           
    }
  
  }

  if(bias_addr + input.dims[2] != output_addr) {
    data<3> src(bias_addr+input.dims[2], input.dims);
    Memcopy(src.asVMem(inst2), output.asVMem(inst2));
    instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
  }

  return output;
}

data<3> LayerNorm_V3(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps) {
  uint32_t weights_addr = weights.addr;
  uint32_t bias_addr = bias.addr;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  auto sum_reg = inst2.AllocVReg("");
  auto mean_reg = inst2.AllocVReg("");
  auto var_reg = inst2.AllocVReg("");
  auto core_id_reg = inst2.AllocVReg("");
  std::vector<VReg> weights_reg;
  std::vector<VReg> bias_reg;
  const uint32_t ch_num = (input.dims[2] + 1024)/1024;
  uint32_t min_load_size;
  data<3> output(output_addr, input.dims);
  Instruction *inst;

  // std::cout << "core_id_reg: " << core_id_reg.id << std::endl;
  // get core id
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }


  // load weights and bias to reg
  for(uint32_t i = 0; i < input.dims[2]; i += kNumberOfSubcores) {
    min_load_size = std::min(input.dims[2] - i, uint32_t(kNumberOfSubcores));
    uint32_t _weight_addr = weights_addr + i;
    uint32_t _bias_addr = bias_addr + i;
    auto _weights_reg = inst2.AllocVReg("");
    auto _bias_reg = inst2.AllocVReg("");

    // std::cout << "_weights_reg: " << _weights_reg.id << std::endl;
    // std::cout << "core_id__bias_regreg: " << _bias_reg.id << std::endl;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_weight_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_weight_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _weights_reg.id, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_bias_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_bias_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, _bias_reg.id, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }   

    weights_reg.push_back(std::move(_weights_reg));
    bias_reg.push_back(std::move(_bias_reg)); 
  }

  for(uint32_t i = 0; i < input.dims[0]*input.dims[1]; i++) {
    // std::cout << "for_" << i << ": \n" << std::endl;

    // reset sum_reg and var_reg
    if(1) {
      inst = new Instruction();
      VectorOperationState set_sum_0(V_U32_MOVE, 0, 0, 46, sum_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &set_sum_0);
      VectorOperationState set_var_0(V_U32_MOVE, 0, 0, 46, var_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &set_var_0);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    // sum += x
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      auto temp_1_reg = inst2.AllocVReg("");
      // std::cout << "temp_1_reg: " << temp_1_reg.id << std::endl;
      uint32_t input_addr = input.addr + i * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      /**/
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        //set 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(input_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(input_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, temp_1_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, temp_1_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_1_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, temp_1_reg.id, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_1_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, temp_1_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_1_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_reg.id, temp_1_reg.id, sum_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    // sum/ch
    {     
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0/input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0/input.dims[2]).first);
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_reg.id, 44, mean_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }    

    // var += pow(x-mear, 2);
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      auto temp_1_reg = inst2.AllocVReg("");
      auto zero_reg = inst2.AllocVReg("");
      // std::cout << "temp_1_reg: " << temp_1_reg.id << std::endl;
      // std::cout << "zero_reg: " << zero_reg.id << std::endl;
      uint32_t load_addr = input.addr + i * input.dims[2] + j;
      min_load_size = std::min(input.dims[2] - j, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, temp_1_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, temp_1_reg.id, mean_reg.id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, temp_1_reg.id, temp_1_reg.id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState select_vm0(V_SELECT_VMASK0, 0, zero_reg.id, temp_1_reg.id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &select_vm0);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, temp_1_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_1_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, temp_1_reg.id, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_1_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, temp_1_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_1_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, var_reg.id, temp_1_reg.id, var_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
  
    // var/ch
    if(1) {       
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0/input.dims[2]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0/input.dims[2]).first);
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, var_reg.id, 44, var_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
        VectorOperationState add(V_F32_ADDITION, 0, var_reg.id, 44, var_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, var_reg.id, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &sqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst); 
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, var_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop );
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }   
    }   
  
    for(uint32_t j = 0; j < input.dims[2]; j += kNumberOfSubcores) {
      uint32_t ch_num = j / 1024;
      uint32_t load_addr = input.addr + i * input.dims[2] + j;
      uint32_t store_addr = (bias_addr + input.dims[2] != output_addr ? bias_addr + input.dims[2] : output_addr) + i * input.dims[2] + j;
      auto temp_1_reg = inst2.AllocVReg("");
      // std::cout << "temp_1_reg: " << temp_1_reg.id << std::endl; 

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, temp_1_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, temp_1_reg.id, mean_reg.id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, temp_1_reg.id, var_reg.id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, temp_1_reg.id, weights_reg[ch_num].id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, temp_1_reg.id, bias_reg[ch_num].id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      
      if(1) {
        Instruction *instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
        instr->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
        instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        instr->SetOperationState(Instruction::SCALARONE, &set_base);
        instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, temp_1_reg.id, 1, 2, 4, 0, 0);
        instr->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }      
    }
  }

  if(bias_addr + input.dims[2] != output_addr) {
    data<3> src(bias_addr+input.dims[2], input.dims);
    INST_TYPE inst2;
    Memcopy(src.asVMem(inst2), output.asVMem(inst2));
    instruction_list.insert(instruction_list.end() ,inst2.inst.insts.begin(), inst2.inst.insts.end()); 
  }

  return output;
}


data<3> LayerNorm(INST_TYPE &inst2,
               data<3> hidden_states,
               data<1> weights,
               data<1> bias,
               uint32_t output_addr,
               float eps)
{ 
    uint32_t input_addr = hidden_states.addr;
    uint32_t input_row = hidden_states.dims[0];
    uint32_t input_col = hidden_states.dims[1];
    uint32_t input_ch = hidden_states.dims[2];
    uint32_t weights_addr = weights.addr;
    uint32_t bias_addr = bias.addr;
    uint32_t variance_reg = 10;
    uint32_t weights_reg = 11;
    uint32_t sum_reg = 20;
    uint32_t sum_for_mean_reg = 25;
    uint32_t eps_reg = 13;
    uint32_t input_addr_reg = 14;
    uint32_t output_addr_reg = 15;
    uint32_t weights_addr_reg = 16;
    uint32_t bias_addr_reg = 7;
    uint32_t id_mask =
        ((input_ch / 128) < 8) ? pow(2, (input_ch / 128)) - 1 : pow(2, 8) - 1;
    uint32_t one_use_ch = (input_ch > 1024) ? 1024 : input_ch;
    Instruction *inst;
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    data<3> output(output_addr, hidden_states.dims);

    for (uint32_t num_no = 0; num_no < 1; num_no++)
    {
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                    (input_row * input_col) - 1);
            ScalarOperationState move1(S_U32_MOVE, 0, 0, 46, 28);
            inst->SetOperationState(Instruction::SCALARONE, &move1);
            ScalarOperationState move2(S_U32_MOVE, 0, 0, 32, 29);
            inst->SetOperationState(Instruction::SCALARTWO, &move2);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                    HelperGetAddress(input_addr).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                    HelperGetAddress(input_addr).first);
            ScalarOperationState move(S_U32_MOVE, 0, 0, 44, input_addr_reg);
            inst->SetOperationState(Instruction::SCALARONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                    HelperGetAddress(output_addr).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                    HelperGetAddress(output_addr).first);
            ScalarOperationState move(S_U32_MOVE, 0, 0, 44, output_addr_reg);
            inst->SetOperationState(Instruction::SCALARONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        for (uint32_t ch_no = 0; ch_no < 1; ch_no += 1024)
        {
            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        (input_ch - one_use_ch) / one_use_ch);
                ScalarOperationState move1(S_U32_MOVE, 0, 0, 46, 30);
                inst->SetOperationState(Instruction::SCALARONE, &move1);
                ScalarOperationState move2(S_U32_MOVE, 0, 0, 32, 31);
                inst->SetOperationState(Instruction::SCALARTWO, &move2);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                ScalarOperationState move(S_U32_MOVE, 0, 0, input_addr_reg, 10);
                inst->SetOperationState(Instruction::SCALARONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT, 0, 10, 32, 0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorLoadOperationState
                    vload(V_LOAD_WITH_OFFSET, 0, 30, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORLOAD, &vload);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0, one_use_ch);
                ScalarOperationState add(S_S32_ADDITION, 0, 10, 32, 10);
                inst->SetOperationState(Instruction::SCALARONE, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }
            if (1)
            {
                inst = new Instruction();
                VectorOperationState mul(V_F32_MULTIPLICATION,
                                         0,
                                         30,
                                         30,
                                         sum_reg);
                inst->SetOperationState(Instruction::VECTORONE, &mul);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, sum_reg, 0, 0);
                inst->SetOperationState(Instruction::MTI, &sum);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT,
                                        0,
                                        sum_reg,
                                        0);
                instr->SetOperationState(Instruction::MTR, &t_pop);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
                MTIOperationState transpose(MTI_TRANSPOSE_START_END,
                                            0,
                                            sum_reg,
                                            4,
                                            0);
                instr->SetOperationState(Instruction::MTI, &transpose);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT,
                                        0,
                                        sum_reg,
                                        0);
                instr->SetOperationState(Instruction::MTR, &t_pop);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, sum_reg, 0, 0);
                inst->SetOperationState(Instruction::MTI, &sum);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT,
                                        0,
                                        sum_reg,
                                        0);
                instr->SetOperationState(Instruction::MTR, &t_pop);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            //比较reg_30是否小于reg_31
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState cmp(S_S32_LESSER,
                                         0,
                                         30,
                                         31,
                                         kVMemSegShift);
                instr->SetOperationState(Instruction::SCALARONE, &cmp);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            // reg+30 += 1
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState sadd(S_S32_ADDITION, 0, 30, 48, 30);
                instr->SetOperationState(Instruction::SCALARONE, &sadd);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            //分支循环
            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0, -12);
                ScalarOperationState branch(S_BRANCH, 5, 0, 0, 1);
                instr->SetOperationState(Instruction::SCALARONE, &branch);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }
        }

        //求每个通道的和(+14)
        for (uint32_t ch_no = 0; ch_no < 1; ch_no += 1024)
        {
            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        (input_ch - one_use_ch) / one_use_ch);
                ScalarOperationState move1(S_U32_MOVE, 0, 0, 46, 30);
                inst->SetOperationState(Instruction::SCALARONE, &move1);
                ScalarOperationState move2(S_U32_MOVE, 0, 0, 32, 31);
                inst->SetOperationState(Instruction::SCALARTWO, &move2);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                ScalarOperationState move(S_U32_MOVE, 0, 0, input_addr_reg, 10);
                inst->SetOperationState(Instruction::SCALARONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT, 0, 10, 32, 0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorLoadOperationState
                    vload(V_LOAD_WITH_OFFSET, 0, 30, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORLOAD, &vload);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0, one_use_ch);
                ScalarOperationState add(S_S32_ADDITION, 0, 10, 32, 10);
                inst->SetOperationState(Instruction::SCALARONE, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }
            // if (1)
            // {
            //     inst = new Instruction();
            //     VectorOperationState mul(V_F32_MULTIPLICATION,
            //                              0,
            //                              30,
            //                              30,
            //                              sum_reg);
            //     inst->SetOperationState(Instruction::VECTORONE, &mul);
            //     CompleteInstruction(inst);
            //     instruction_list.push_back(inst);
            // }
            if (1)
            {
                inst = new Instruction();
                VectorOperationState set_v(V_U32_MOVE, 0, 0, 30, sum_for_mean_reg);
                inst->SetOperationState(Instruction::VECTORONE, &set_v);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, sum_for_mean_reg, 0, 0);
                inst->SetOperationState(Instruction::MTI, &sum);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT,
                                        0,
                                        sum_for_mean_reg,
                                        0);
                instr->SetOperationState(Instruction::MTR, &t_pop);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
                MTIOperationState transpose(MTI_TRANSPOSE_START_END,
                                            0,
                                            sum_for_mean_reg,
                                            4,
                                            0);
                instr->SetOperationState(Instruction::MTI, &transpose);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT,
                                        0,
                                        sum_for_mean_reg,
                                        0);
                instr->SetOperationState(Instruction::MTR, &t_pop);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, sum_for_mean_reg, 0, 0);
                inst->SetOperationState(Instruction::MTI, &sum);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT,
                                        0,
                                        sum_for_mean_reg,
                                        0);
                instr->SetOperationState(Instruction::MTR, &t_pop);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            //比较reg_30是否小于reg_31
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState cmp(S_S32_LESSER,
                                         0,
                                         30,
                                         31,
                                         kVMemSegShift);
                instr->SetOperationState(Instruction::SCALARONE, &cmp);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            // reg+30 += 1
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState sadd(S_S32_ADDITION, 0, 30, 48, 30);
                instr->SetOperationState(Instruction::SCALARONE, &sadd);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            //分支循环
            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0, -12);
                ScalarOperationState branch(S_BRANCH, 5, 0, 0, 1);
                instr->SetOperationState(Instruction::SCALARONE, &branch);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }
        }

        //对每个通道的和求均值之后的平方根，最后再求个倒数：8条指令
        {
            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        HelperGetFloatingBits(input_ch).second);
                inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                        HelperGetFloatingBits(input_ch).first);
                VectorOperationState move(V_U32_MOVE, 0, 0, 44, 1);
                inst->SetOperationState(Instruction::VECTORONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                VectorOperationState rcp(V_F32_RECIPROCAL, 0, 1, 0, 0);
                inst->SetOperationState(Instruction::VECTORONE, &rcp);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT,
                                          0,
                                          1,
                                          0);
                inst->SetOperationState(Instruction::MTR, &urf_pop);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                VectorOperationState mul(V_F32_MULTIPLICATION,
                                         0,
                                         sum_reg,
                                         1,
                                         variance_reg);
                inst->SetOperationState(Instruction::VECTORONE, &mul);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        HelperGetFloatingBits(eps).second);
                inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                        HelperGetFloatingBits(eps).first);
                VectorOperationState move(V_U32_MOVE, 0, 0, 44, eps_reg);
                inst->SetOperationState(Instruction::VECTORONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                VectorOperationState add(V_F32_ADDITION,
                                         0,
                                         eps_reg,
                                         variance_reg,
                                         1);
                inst->SetOperationState(Instruction::VECTORTWO, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL,
                                          0,
                                          variance_reg,
                                          0,
                                          0);
                inst->SetOperationState(Instruction::VECTORONE, &sqrt);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT,
                                          0,
                                          1,
                                          0);
                inst->SetOperationState(Instruction::MTR, &urf_pop);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }
        }
        //sum_for_mean_reg / dim_3(+1)
        {
            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        HelperGetFloatingBits(1.0 / input_ch).second);
                inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                        HelperGetFloatingBits(1.0 / input_ch).first);
                VectorOperationState move(V_F32_MULTIPLICATION, 0, sum_for_mean_reg, 44, sum_for_mean_reg);
                inst->SetOperationState(Instruction::VECTORONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }
        }
        //使通道中的每一个值都进行标准化：11条指令(+1)
        for (uint32_t ch_no = 0; ch_no < 1; ch_no += 1024)
        {

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        (input_ch - one_use_ch) / one_use_ch);
                ScalarOperationState move1(S_U32_MOVE, 0, 0, 46, 30);
                inst->SetOperationState(Instruction::SCALARONE, &move1);
                ScalarOperationState move2(S_U32_MOVE, 0, 0, 32, 31);
                inst->SetOperationState(Instruction::SCALARTWO, &move2);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                ScalarOperationState move(S_U32_MOVE, 0, 0, input_addr_reg, 10);
                inst->SetOperationState(Instruction::SCALARONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                ScalarOperationState move(S_U32_MOVE,
                                          0,
                                          0,
                                          output_addr_reg,
                                          11);
                inst->SetOperationState(Instruction::SCALARONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT, 0, 10, 32, 0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorLoadOperationState
                    vload(V_LOAD_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORLOAD, &vload);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0, one_use_ch);
                ScalarOperationState add(S_S32_ADDITION, 0, 10, 32, 10);
                inst->SetOperationState(Instruction::SCALARONE, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)//V_F32_SUBTRACTION
            {
                inst = new Instruction();
                VectorOperationState mul(V_F32_SUBTRACTION, 0, 2, sum_for_mean_reg, 2);
                inst->SetOperationState(Instruction::VECTORTWO, &mul);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                VectorOperationState mul(V_F32_MULTIPLICATION, 0, 1, 2, 2);
                inst->SetOperationState(Instruction::VECTORONE, &mul);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT, 0, 11, 32, 0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorStoreOperationState
                    vstore(V_STORE_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORSTORE, &vstore);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0, one_use_ch);
                ScalarOperationState add(S_S32_ADDITION, 0, 11, 32, 11);
                inst->SetOperationState(Instruction::SCALARONE, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            //比较reg_31是否小于reg_30
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState cmp(S_S32_LESSER, 0, 30, 31, 5);
                instr->SetOperationState(Instruction::SCALARONE, &cmp);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            // reg+30 += 1
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState sadd(S_S32_ADDITION, 0, 30, 48, 30);
                instr->SetOperationState(Instruction::SCALARONE, &sadd);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            //分支循环
            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0, -9);
                ScalarOperationState branch(S_BRANCH, 5, 0, 0, 1);
                instr->SetOperationState(Instruction::SCALARONE, &branch);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }
        }

        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, input_ch);
            ScalarOperationState add(S_S32_ADDITION,
                                     0,
                                     input_addr_reg,
                                     32,
                                     input_addr_reg);
            inst->SetOperationState(Instruction::SCALARONE, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, input_ch);
            ScalarOperationState add(S_S32_ADDITION,
                                     0,
                                     output_addr_reg,
                                     32,
                                     output_addr_reg);
            inst->SetOperationState(Instruction::SCALARONE, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        //比较reg_31是否小于reg_30
        if (1)
        {
            Instruction *instr = new Instruction();
            ScalarOperationState cmp(S_S32_LESSER, 0, 28, 29, 5);
            instr->SetOperationState(Instruction::SCALARONE, &cmp);
            CompleteInstruction(instr);
            instruction_list.push_back(instr);
        }

        // reg+31 += 1
        if (1)
        {
            Instruction *instr = new Instruction();
            ScalarOperationState sadd(S_S32_ADDITION, 0, 28, 48, 28);
            instr->SetOperationState(Instruction::SCALARONE, &sadd);
            CompleteInstruction(instr);
            instruction_list.push_back(instr);
        }

        //分支循环
        if (1)
        {
            Instruction *instr = new Instruction();
            instr->SetImmediateValue(Instruction::IMMEDIATE0, /*-38*/-53);
            ScalarOperationState branch(S_BRANCH, 5, 0, 0, 1);
            instr->SetOperationState(Instruction::SCALARONE, &branch);
            CompleteInstruction(instr);
            instruction_list.push_back(instr);
        }
    }

    for (uint32_t num_no = 0; num_no < 1; num_no++)
    {
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                    (input_row * input_col) - 1);
            ScalarOperationState move1(S_U32_MOVE, 0, 0, 46, 28);
            inst->SetOperationState(Instruction::SCALARONE, &move1);
            ScalarOperationState move2(S_U32_MOVE, 0, 0, 32, 29);
            inst->SetOperationState(Instruction::SCALARTWO, &move2);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                    HelperGetAddress(output_addr).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                    HelperGetAddress(output_addr).first);
            ScalarOperationState move(S_U32_MOVE, 0, 0, 44, output_addr_reg);
            inst->SetOperationState(Instruction::SCALARONE, &move);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        //对单个通道进行weights点乘：12条指令(+3)
        for (uint32_t ch_no = 0; ch_no < 1; ch_no += 1024)
        {

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        (input_ch - one_use_ch) / one_use_ch);
                ScalarOperationState move1(S_U32_MOVE, 0, 0, 46, 30);
                inst->SetOperationState(Instruction::SCALARONE, &move1);
                ScalarOperationState move2(S_U32_MOVE, 0, 0, 32, 31);
                inst->SetOperationState(Instruction::SCALARTWO, &move2);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        HelperGetAddress(weights_addr).second);
                inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                        HelperGetAddress(weights_addr).first);
                ScalarOperationState move(S_U32_MOVE,
                                          0,
                                          0,
                                          44,
                                          weights_addr_reg);
                inst->SetOperationState(Instruction::SCALARONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                        HelperGetAddress(bias_addr).second);
                inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                        HelperGetAddress(bias_addr).first);
                ScalarOperationState move(S_U32_MOVE,
                                          0,
                                          0,
                                          44,
                                          bias_addr_reg);//**********
                inst->SetOperationState(Instruction::SCALARONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                ScalarOperationState move(S_U32_MOVE,
                                          0,
                                          0,
                                          output_addr_reg,
                                          10);
                inst->SetOperationState(Instruction::SCALARONE, &move);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT, 0, 10, 32, 0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorLoadOperationState
                    vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORLOAD, &vload);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }
            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT,
                                              0,
                                              weights_addr_reg,
                                              32,
                                              0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorLoadOperationState
                    vload(V_LOAD_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORLOAD, &vload);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }
            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT,
                                              0,
                                              bias_addr_reg,
                                              32,
                                              0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorLoadOperationState
                    vload(V_LOAD_WITH_OFFSET, 0, 3, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORLOAD, &vload);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                VectorOperationState mul(V_F32_MULTIPLICATION, 0, 1, 2, 2);
                inst->SetOperationState(Instruction::VECTORONE, &mul);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                VectorOperationState mul(V_F32_ADDITION, 0, 2, 3, 2);
                inst->SetOperationState(Instruction::VECTORTWO, &mul);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0,
                                         kVMemSegShift);
                instr->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
                instr->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                         0);
                ScalarOperationState set_base(S_U32_SHIFTRIGHT, 0, 10, 32, 0);
                instr->SetOperationState(Instruction::SCALARTWO, &set_base);
                instr->SetImmediateValue(Instruction::IMMEDIATE4, 0);
                instr->SetImmediateValue(Instruction::IMMEDIATE2, 1);
                VectorStoreOperationState
                    vload(V_STORE_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 5);
                instr->SetOperationState(Instruction::VECTORSTORE, &vload);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0, one_use_ch);
                ScalarOperationState add(S_S32_ADDITION,
                                         0,
                                         weights_addr_reg,
                                         32,
                                         weights_addr_reg);
                inst->SetOperationState(Instruction::SCALARONE, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0, one_use_ch);
                ScalarOperationState add(S_S32_ADDITION,
                                         0,
                                         bias_addr_reg,
                                         32,
                                         bias_addr_reg);
                inst->SetOperationState(Instruction::SCALARONE, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            if (1)
            {
                inst = new Instruction();
                inst->SetImmediateValue(Instruction::IMMEDIATE0, one_use_ch);
                ScalarOperationState add(S_S32_ADDITION, 0, 10, 32, 10);
                inst->SetOperationState(Instruction::SCALARONE, &add);
                CompleteInstruction(inst);
                instruction_list.push_back(inst);
            }

            //�Ƚ�reg_30�Ƿ�С��reg_31
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState cmp(S_S32_LESSER, 0, 30, 31, 5);
                instr->SetOperationState(Instruction::SCALARONE, &cmp);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            // reg+30 += 1
            if (1)
            {
                Instruction *instr = new Instruction();
                ScalarOperationState sadd(S_S32_ADDITION, 0, 30, 48, 30);
                instr->SetOperationState(Instruction::SCALARONE, &sadd);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }

            //��֧ѭ��
            if (1)
            {
                Instruction *instr = new Instruction();
                instr->SetImmediateValue(Instruction::IMMEDIATE0, /*-9*/-12);
                ScalarOperationState branch(S_BRANCH, 5, 0, 0, 1);
                instr->SetOperationState(Instruction::SCALARONE, &branch);
                CompleteInstruction(instr);
                instruction_list.push_back(instr);
            }
        }

        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, input_ch);
            ScalarOperationState add(S_S32_ADDITION,
                                     0,
                                     output_addr_reg,
                                     32,
                                     output_addr_reg);
            inst->SetOperationState(Instruction::SCALARONE, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }

        //�Ƚ�reg_31�Ƿ�С��reg_30
        if (1)
        {
            Instruction *instr = new Instruction();
            ScalarOperationState cmp(S_S32_LESSER, 0, 28, 29, 5);
            instr->SetOperationState(Instruction::SCALARONE, &cmp);
            CompleteInstruction(instr);
            instruction_list.push_back(instr);
        }

        // reg+31 += 1
        if (1)
        {
            Instruction *instr = new Instruction();
            ScalarOperationState sadd(S_S32_ADDITION, 0, 28, 48, 28);
            instr->SetOperationState(Instruction::SCALARONE, &sadd);
            CompleteInstruction(instr);
            instruction_list.push_back(instr);
        }

        //��֧ѭ��
        if (1)
        {
            Instruction *instr = new Instruction();
            instr->SetImmediateValue(Instruction::IMMEDIATE0, /*-16*/-19);
            ScalarOperationState branch(S_BRANCH, 5, 0, 0, 1);
            instr->SetOperationState(Instruction::SCALARONE, &branch);
            CompleteInstruction(instr);
            instruction_list.push_back(instr);
        }
    }

    return output;
}

data<4> Cosh(INST_TYPE &inst2, data<4> input, uint32_t output_addr) {
  Instruction* inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, input.dims);
  uint32_t total_size = input.size();

  for(uint32_t index = 0; index < total_size; index += 2 * kNumberOfSubcores) {
    const uint32_t reg_1 = 10;
    const uint32_t reg_1_1 = 11;
    const uint32_t reg_1_2 = 12;
    const uint32_t reg_2 = 13; 
    const uint32_t reg_2_1 = 14;
    const uint32_t reg_2_2 = 15;
    uint32_t load_addr_1 = input.addr + index;
    uint32_t load_addr_2 = input.addr + index + kNumberOfSubcores;
    uint32_t store_addr_1 = output_addr + index;
    uint32_t store_addr_2 = output_addr + index + kNumberOfSubcores;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState less(V_F32_LESSER, 0, reg_1, 46, 0);
      inst->SetOperationState(Instruction::VECTORONE, &less);

      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState exp(V_F32_EXPONENT, 0, reg_1, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &exp);

      VectorOperationState less(V_F32_LESSER, 0, reg_2, 46, 1);
      inst->SetOperationState(Instruction::VECTORTWO, &less);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, reg_1, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);

      VectorOperationState exp(V_F32_EXPONENT, 0, reg_2, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &exp);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState select_mask_1(V_SELECT_VMASK0, 0, reg_1, 46, reg_1_1);
      inst->SetOperationState(Instruction::VECTORONE, &select_mask_1);

      VectorOperationState select_mask_2(V_SELECT_VMASK0, 0, 0, reg_1, reg_1_2);
      inst->SetOperationState(Instruction::VECTORTWO, &select_mask_2);

      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, reg_2, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);      
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState select_mask_1(V_SELECT_VMASK1, 0, reg_2, 46, reg_2_1);
      inst->SetOperationState(Instruction::VECTORONE, &select_mask_1);

      VectorOperationState select_mask_2(V_SELECT_VMASK1, 0, 0, reg_2, reg_2_2);
      inst->SetOperationState(Instruction::VECTORTWO, &select_mask_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();   
      VectorOperationState rcp(V_F32_RECIPROCAL, 0, reg_1, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, reg_1, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);  

      VectorOperationState rcp(V_F32_RECIPROCAL, 0, reg_2, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);   
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState select_mask_1(V_SELECT_VMASK0, 0, reg_1_1, reg_1, reg_1_1);
      inst->SetOperationState(Instruction::VECTORONE, &select_mask_1);

      VectorOperationState select_mask_2(V_SELECT_VMASK0, 0, reg_1, reg_1_2, reg_1_2);
      inst->SetOperationState(Instruction::VECTORTWO, &select_mask_2);

      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, reg_2, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);  
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState select_mask_1(V_SELECT_VMASK1, 0, reg_2_1, reg_2, reg_2_1);
      inst->SetOperationState(Instruction::VECTORONE, &select_mask_1);

      VectorOperationState select_mask_2(V_SELECT_VMASK1, 0, reg_2, reg_2_2, reg_2_2);
      inst->SetOperationState(Instruction::VECTORTWO, &select_mask_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState add(V_F32_ADDITION, 0, reg_1_1, reg_1_2, reg_1);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_1, 50, reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      VectorOperationState add(V_F32_ADDITION, 0, reg_2_1, reg_2_2, reg_2);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);

      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_2, 50, reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }  

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }

  return output;
}

data<4> NewGELUActivationBackward(INST_TYPE &inst2, data<4> forward, data<4> backward, uint32_t output_addr) {
  Instruction* inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, forward.dims);
  data<4> _x(output.addr + output.size(), output.dims);
  data<4> tanh(_x.addr + _x.size(), forward.dims); 
  data<4> cosh(tanh.addr + tanh.size(), forward.dims);
  const uint32_t total_size = forward.size();
  const uint32_t sqrt_2_reg = 20;
  const uint32_t sqrt_PI_reg = 21;
  const uint32_t rsqrt_PI_reg = 22;
  
  //sqrt(2) sqrt(PI) rsqrt(PI)
  if(1) {
    if(1) {
      inst = new Instruction();
      VectorOperationState move_1(V_U32_MOVE, 0, 0, 51, sqrt_2_reg);
      inst->SetOperationState(Instruction::VECTORONE, &move_1);
      VectorOperationState move_2(V_U32_MOVE, 0, 0, 52, sqrt_PI_reg);
      inst->SetOperationState(Instruction::VECTORTWO, &move_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState rsqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, sqrt_2_reg, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rsqrt);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);         
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, sqrt_2_reg, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);

      VectorOperationState rsqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, sqrt_PI_reg, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rsqrt);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState rcp(V_F32_RECIPROCAL, 0, sqrt_2_reg, 0 , 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);

      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, sqrt_PI_reg, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, sqrt_2_reg, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);

      VectorOperationState rcp(V_F32_RECIPROCAL, 0, sqrt_PI_reg, 0 , 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, sqrt_PI_reg, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);

      VectorOperationState move(V_U32_MOVE, 0, 0, 52, rsqrt_PI_reg);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState rsqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, rsqrt_PI_reg, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rsqrt);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, rsqrt_PI_reg, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }
  }

  // _x= (sqrt(2) * (0.044715*pow(x, 3) + x)) / sqrt(PI)
  for(uint32_t index = 0; index < total_size; index += 2 * kNumberOfSubcores){
    uint32_t reg_1 = 10;
    uint32_t _reg_1 = 11;
    uint32_t reg_2 = 12;
    uint32_t _reg_2 = 13;
    uint32_t temp_reg = 14;
    uint32_t load_addr_1 = forward.addr + index;
    uint32_t load_addr_2 = forward.addr + index + kNumberOfSubcores;
    uint32_t store_addr_1 = _x.addr + index;
    uint32_t store_addr_2 = _x.addr + index + kNumberOfSubcores;
    
    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_1, reg_1, _reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, _reg_1, reg_1, _reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_2, reg_2, _reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, _reg_2, reg_2, _reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(0.044715).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(0.044715).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, _reg_1, 44, _reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);        
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(0.044715).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(0.044715).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, _reg_2, 44, _reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      VectorOperationState add(V_F32_ADDITION, 0, _reg_1, reg_1, reg_1);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);        
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_1, sqrt_2_reg, reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      VectorOperationState add(V_F32_ADDITION, 0, _reg_2, reg_2, reg_2);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);       
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_2, sqrt_2_reg, reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);       
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_1, rsqrt_PI_reg, reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_2, rsqrt_PI_reg, reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }
  }

  tanh = Tanh(inst2, _x, tanh.addr);
  cosh = Cosh(inst2, _x, cosh.addr);

  //_x = pow(1/cosh(x), 2)
  for(uint32_t index = 0; index < total_size; index += 2 * kNumberOfSubcores) {
    uint32_t load_addr_1 = cosh.addr + index;
    uint32_t load_addr_2 = cosh.addr + index + kNumberOfSubcores;
    uint32_t store_addr_1 = cosh.addr + index;
    uint32_t store_addr_2 = cosh.addr + index + kNumberOfSubcores;
    uint32_t reg_1 = 10;
    uint32_t reg_2 = 11;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState rcp(V_F32_RECIPROCAL, 0, reg_1, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);

      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState rcp(V_F32_RECIPROCAL, 0, reg_2, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);      

      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, reg_1, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_1, reg_1, reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, reg_2, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);     
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, reg_2, reg_2, reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }
  }

  //{[sqrt(2) * x * (0.134145 * pow(x, 2) + 1) * _x) / sqrt(PI) + 1 + tanh(x)] / 2} * backward
  for(uint32_t index = 0; index < total_size; index += 2 * kNumberOfSubcores) {
    uint32_t input_addr_1 = forward.addr + index;
    uint32_t input_addr_2 = forward.addr + kNumberOfSubcores + index;
    uint32_t _x_addr_1 = cosh.addr + index;
    uint32_t _x_addr_2 = cosh.addr + kNumberOfSubcores + index;
    uint32_t tanh_addr_1 = tanh.addr + index;
    uint32_t tanh_addr_2 = tanh.addr + kNumberOfSubcores + index;
    uint32_t backward_addr_1 = backward.addr + index;
    uint32_t backward_addr_2 = backward.addr + kNumberOfSubcores + index;    
    uint32_t store_addr_1 = output_addr + index;
    uint32_t store_addr_2 = output_addr + kNumberOfSubcores + index;
    uint32_t cosh_reg_1 = 10;
    uint32_t cosh_reg_2 = 11;
    uint32_t input_reg_1 = 12;
    uint32_t input_reg_2 = 13;
    uint32_t tanh_reg_1 = 14;
    uint32_t tanh_reg_2 = 15;
    uint32_t temp_reg_1 = 16;
    uint32_t temp_reg_2 = 17;
    uint32_t backward_reg_1 = 18;
    uint32_t backward_reg_2 = 19;

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(input_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(input_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, input_reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(input_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(input_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, input_reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);

      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, sqrt_2_reg, temp_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_x_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_x_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, cosh_reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);

      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, sqrt_2_reg, temp_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(_x_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(_x_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, cosh_reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);

      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, input_reg_1, input_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(tanh_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(tanh_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, tanh_reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);

      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, input_reg_2, input_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(tanh_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(tanh_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, tanh_reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE2, HelperGetFloatingBits(0.134145).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE3, HelperGetFloatingBits(0.134145).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, 45, input_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(backward_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(backward_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, backward_reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);

      VectorOperationState add(V_F32_ADDITION, 0, input_reg_1, 49, input_reg_1);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE2, HelperGetFloatingBits(0.134145).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE3, HelperGetFloatingBits(0.134145).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, 45, input_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(backward_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(backward_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, backward_reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);

      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, temp_reg_1, input_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      VectorOperationState add(V_F32_ADDITION, 0, input_reg_2, 49, input_reg_2);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, temp_reg_2, input_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, cosh_reg_1, input_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, cosh_reg_2, input_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, rsqrt_PI_reg, input_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, rsqrt_PI_reg, input_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      VectorOperationState add(V_F32_ADDITION, 0, input_reg_1, 49, input_reg_1);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState add(V_F32_ADDITION, 0, input_reg_1, tanh_reg_1, input_reg_1);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState add(V_F32_ADDITION, 0, input_reg_2, 49, input_reg_2);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, 50, input_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      VectorOperationState add(V_F32_ADDITION, 0, input_reg_2, tanh_reg_2, input_reg_2);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, 50, input_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1) {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_1, backward_reg_1, input_reg_1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_1 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_1 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, input_reg_1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);

      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg_2, backward_reg_2, input_reg_2);
      inst->SetOperationState(Instruction::VECTORONE, &mul);      
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr_2 / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr_2 / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, input_reg_2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);   
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }
  }
  
  return output;
}

//avg_erro : 0.137%
data<4> LayerNormDxBackward(INST_TYPE &inst2, data<4> forward_input, data<4> backward_input, data<1> weights, uint32_t output_addr, float eps) {
  Instruction* inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, forward_input.dims);
  const uint32_t ch_num = forward_input.dims[0] * forward_input.dims[1] * forward_input.dims[2];
  uint32_t min_load_size;
  auto core_id_reg = inst2.AllocVReg("");
  auto zero_reg = inst2.AllocVReg("");

  // get core id and get zero
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_reg.id);
    inst->SetOperationState(Instruction::VECTORTWO, &move);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  
  for(uint32_t batch = 0; batch < ch_num; batch++) {
    auto sum_reg = inst2.AllocVReg("");
    auto mean_reg = inst2.AllocVReg("");
    auto variance_reg = inst2.AllocVReg("");
    auto input_reg = inst2.AllocVReg("");
    auto weight_reg = inst2.AllocVReg("");
    auto _x_reg = inst2.AllocVReg("");
    auto temp_1_reg = inst2.AllocVReg("");
    auto temp_2_reg = inst2.AllocVReg("");
    auto _temp_1_reg = inst2.AllocVReg("");
    auto _temp_2_reg = inst2.AllocVReg("");
    auto backward_reg = inst2.AllocVReg("");

    //reset 0
    if(1) {
      inst = new Instruction();
      VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, sum_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move_1);
      VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, mean_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &move_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, variance_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move_1);
      VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, temp_1_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &move_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, temp_2_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &move_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }
    
    if(1) {
      inst = new Instruction();
      VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, _temp_1_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move_1);
      VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, _temp_2_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &move_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    // sum += x
    for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t load_addr = forward_input.addr + batch * forward_input.dims[3] + ch;
      min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      /**/
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        //set 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, input_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, input_reg.id, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_reg.id, input_reg.id, sum_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }



    // mean_reg = sum/ch
    if(1) { 
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0/forward_input.dims[3]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0/forward_input.dims[3]).first);
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_reg.id, 44, mean_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }    

    // var += pow(x-mean, 2);
    for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t load_addr = forward_input.addr + batch * forward_input.dims[3] + ch;
      min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, input_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, input_reg.id, mean_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg.id, input_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState select_vm0(V_SELECT_VMASK0, 0, zero_reg.id, input_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &select_vm0);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, input_reg.id, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, variance_reg.id, input_reg.id, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }

    // sqrt(var/ch)
    if (1) {
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0/forward_input.dims[3]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0/forward_input.dims[3]).first);
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, variance_reg.id, 44, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }  

      if(1) {
        inst = new Instruction();
        VectorOperationState rsqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, variance_reg.id, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rsqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, variance_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, variance_reg.id, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, variance_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    //1/sqrt(pow(var, 2) + eps)
    if(1) {
      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, variance_reg.id, variance_reg.id, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
        VectorOperationState add(V_F32_ADDITION, 0, variance_reg.id, 44, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rsqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, variance_reg.id, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rsqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, variance_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t load_addr = forward_input.addr + batch * forward_input.dims[3] + ch;
      uint32_t weights_addr = weights.addr + ch;
      uint32_t backward_addr = backward_input.addr + batch * forward_input.dims[3] + ch;
      uint32_t store_addr = output.addr + batch * output.dims[3] + ch;
      min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();  
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, input_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, input_reg.id, mean_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(weights_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(weights_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, weight_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(backward_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(backward_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, backward_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);

        VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg.id, variance_reg.id, _x_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, backward_reg.id, weight_reg.id, _temp_2_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, _temp_2_reg.id, _x_reg.id, _temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);              
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, _temp_1_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, _temp_1_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, _temp_2_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      /**/
      if(1){
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, _temp_1_reg.id, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, _temp_2_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, _temp_1_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, _temp_2_reg.id, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, _temp_1_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, _temp_2_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        inst = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, _temp_1_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, _temp_2_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, temp_1_reg.id, _temp_1_reg.id, temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, _temp_2_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, temp_2_reg.id, _temp_2_reg.id, temp_2_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }   
    }

    for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t load_addr = forward_input.addr + batch * forward_input.dims[3] + ch;
      uint32_t weights_addr = weights.addr + ch;
      uint32_t backward_addr = backward_input.addr + batch * forward_input.dims[3] + ch;
      uint32_t store_addr = output.addr + batch * output.dims[3] + ch;
      min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();  
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, input_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, input_reg.id, mean_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(weights_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(weights_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, weight_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(backward_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(backward_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, backward_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);

        VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg.id, variance_reg.id, _x_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, temp_1_reg.id, _x_reg.id, _temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);              
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, _temp_1_reg.id, temp_2_reg.id, _temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);           
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, backward_reg.id, weight_reg.id, _temp_2_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);            
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0/forward_input.dims[3]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0/forward_input.dims[3]).first);
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, _temp_1_reg.id, 44, _temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);              
      }  

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, _temp_2_reg.id, _temp_1_reg.id, _temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);           
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, _temp_1_reg.id, variance_reg.id, _temp_1_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);              
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, _temp_1_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }      
    }
  }

  return output;
}

//avg_erro: 0.0361347%
data<1> LayerNormDwBackward(INST_TYPE &inst2, data<4> forward_input, data<4> backward_input, uint32_t output_addr, float eps) {
  Instruction* inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<1> output(output_addr, {forward_input.dims[3]});
  const uint32_t ch_num = forward_input.dims[0] * forward_input.dims[1] * forward_input.dims[2];
  uint32_t min_load_size;
  auto core_id_reg = inst2.AllocVReg("");
  auto zero_reg = inst2.AllocVReg("");
  const uint32_t result_num = (forward_input.dims[3] + kNumberOfSubcores) / kNumberOfSubcores;
  std::vector<VReg> result;

  for(uint32_t i = 0; i < result_num; i++) {
    result.emplace_back(inst2.AllocVReg(""));
  }

  // get core id and get zero
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_reg.id);
    inst->SetOperationState(Instruction::VECTORTWO, &move);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  // reset 0 
  for(uint32_t i = 0; i < result_num; i++) {
    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, result[i].id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  
  for(uint32_t batch = 0; batch < ch_num; batch++) {
    auto sum_reg = inst2.AllocVReg("");
    auto mean_reg = inst2.AllocVReg("");
    auto variance_reg = inst2.AllocVReg("");
    auto input_reg = inst2.AllocVReg("");
    auto _x_reg = inst2.AllocVReg("");
    auto backward_reg = inst2.AllocVReg("");

    //reset 0
    if(1) {
      inst = new Instruction();
      VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, sum_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move_1);
      VectorOperationState move_2(V_U32_MOVE, 0, 0, 46, mean_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &move_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState move_1(V_U32_MOVE, 0, 0, 46, variance_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move_1);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }

    // sum += x
    for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t load_addr = forward_input.addr + batch * forward_input.dims[3] + ch;
      min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      /**/
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        //set 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, input_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, input_reg.id, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_reg.id, input_reg.id, sum_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    // mean_reg = sum/ch
    if(1) { 
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0/forward_input.dims[3]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0/forward_input.dims[3]).first);
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_reg.id, 44, mean_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }    

    // var += pow(x-mean, 2);
    for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t load_addr = forward_input.addr + batch * forward_input.dims[3] + ch;
      min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, input_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, input_reg.id, mean_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, input_reg.id, input_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState select_vm0(V_SELECT_VMASK0, 0, zero_reg.id, input_reg.id, input_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &select_vm0);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      /**/
      if(1){
        Instruction* instr = new Instruction();
        instr->SetImmediateValue(Instruction::IMMEDIATE2, 7);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, input_reg.id, 4, 0);
        instr->SetOperationState(Instruction::MTI, &transpose);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        MTIOperationState sum(MTI_REDUCTION_V_SUM, 0, input_reg.id, 0, 0);
        inst->SetOperationState(Instruction::MTI, &sum);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);    
      }

      if(1){
        Instruction* instr = new Instruction();
        MTROperationState t_pop(MTR_READ_TRANSPOSE_RESULT, 0, input_reg.id, 0);
        instr->SetOperationState(Instruction::MTR, &t_pop);
        CompleteInstruction(instr);
        instruction_list.push_back(instr);  
      } 

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, variance_reg.id, input_reg.id, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }

    // sqrt(var/ch)
    if (1) {
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(1.0/forward_input.dims[3]).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(1.0/forward_input.dims[3]).first);
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, variance_reg.id, 44, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }  

      if(1) {
        inst = new Instruction();
        VectorOperationState rsqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, variance_reg.id, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rsqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, variance_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rcp(V_F32_RECIPROCAL, 0, variance_reg.id, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rcp);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, variance_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }

    //1/sqrt(pow(var, 2) + eps)
    if(1) {
      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, variance_reg.id, variance_reg.id, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
        VectorOperationState add(V_F32_ADDITION, 0, variance_reg.id, 44, variance_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState rsqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, variance_reg.id, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &rsqrt);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }

      if(1) {
        inst = new Instruction();
        MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, variance_reg.id, 0);
        inst->SetOperationState(Instruction::MTR, &urf_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);  
      }
    }
  
    for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t result_index = ch / kNumberOfSubcores;
      uint32_t load_addr = forward_input.addr + batch * forward_input.dims[3] + ch;
      uint32_t backward_addr = backward_input.addr + batch * forward_input.dims[3] + ch;
      min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, input_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if (1) {
        inst = new Instruction();
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, backward_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &move);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, input_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, input_reg.id, mean_reg.id, _x_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(backward_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(backward_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, backward_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, _x_reg.id, variance_reg.id, _x_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, backward_reg.id, _x_reg.id, _x_reg.id);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);   
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, result[result_index].id, _x_reg.id, result[result_index].id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);   
      }      
    }
  }

  for(uint32_t ch = 0; ch < forward_input.dims[3]; ch += kNumberOfSubcores) {
    uint32_t store_addr = output_addr + ch;
    uint32_t result_index = ch / kNumberOfSubcores;
    min_load_size = std::min(forward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
    
    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
      VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
      inst->SetOperationState(Instruction::VECTORONE, &less);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, result[result_index].id, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }  
  }

  return output;
}

//avg_erro: 0%
data<1> LayerNormDbBackward(INST_TYPE &inst2, data<4> backward_input, uint32_t output_addr) {
  Instruction* inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  const uint32_t ch_num = backward_input.dims[0] * backward_input.dims[1] * backward_input.dims[2];
  data<1> output(output_addr, {backward_input.dims[3]});
  uint32_t min_load_size;
  auto core_id_reg = inst2.AllocVReg("");
  const uint32_t result_num = (backward_input.dims[3] + kNumberOfSubcores) / kNumberOfSubcores;
  std::vector<VReg> result;

  for(uint32_t i = 0; i < result_num; i++) {
    result.emplace_back(inst2.AllocVReg(""));
  }

  // get core id and get zero
  if(1) {
    inst = new Instruction();
    VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
    inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  // reset 0 
  for(uint32_t i = 0; i < result_num; i++) {
    if(1) {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, result[i].id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
    
  for(uint32_t batch = 0; batch < ch_num; batch++) {
    for(uint32_t ch = 0; ch < backward_input.dims[3]; ch += kNumberOfSubcores) {
      uint32_t load_addr = backward_input.addr + batch * backward_input.dims[3] + ch;
      uint32_t result_index = ch / kNumberOfSubcores;
      auto load_reg = inst2.AllocVReg("");
      min_load_size = std::min(backward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
      
      /**/
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
        VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
        inst->SetOperationState(Instruction::VECTORONE, &less);
        //set 0
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, load_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(load_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(load_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0, 0, load_reg.id, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if(1) {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, result[result_index].id, load_reg.id, result[result_index].id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);        
      }
    }
  }

  for(uint32_t ch = 0; ch < backward_input.dims[3]; ch += kNumberOfSubcores) {
    uint32_t store_addr = output_addr + ch;
    uint32_t result_index = ch / kNumberOfSubcores;
    min_load_size = std::min(backward_input.dims[3] - ch, uint32_t(kNumberOfSubcores));
    
    /**/
    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(min_load_size).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(min_load_size).first);
      VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 44, 0);
      inst->SetOperationState(Instruction::VECTORONE, &less);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(store_addr / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(store_addr / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_VMASK0, 0, result[result_index].id, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }  
  }

  return output;
}

data<4> Conv1DDxBackward(INST_TYPE &inst2, data<2> weight, data<4> backward_input, uint32_t backward_output_addr) {
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    auto oldRes = inst2.resource;
    inst2.resource = Resource();
    auto outdatasize = weight.dims[0] * backward_input.dims[2];

    inst2.resource.AllocVMem(backward_output_addr + outdatasize);

    auto tInput =
        Alloc<2>(inst2, {backward_input.dims[3], backward_input.dims[2]});

    Transpose(inst2,
              {},
              {},
              {},
              {},
              backward_input.addr,
              backward_input.dims[2],
              backward_input.dims[3],
              tInput.addr);

    auto output = Alloc<2>(inst2, {weight.dims[0], tInput.dims[1]});

    LinearExHiVwVo(inst2,
                   {},
                   {},
                   {},
                   {},
                   128,
                   128,
                   weight.addr,
                   {1, 1, weight.dims[0], weight.dims[1]},
                   tInput.addr,
                   tInput.dims,
                   output.addr);

    Transpose(inst2,
              {},
              {},
              {},
              {},
              output.addr,
              output.dims[0],
              output.dims[1],
              backward_output_addr);

    inst2.resource = oldRes;
    instruction_list.insert(instruction_list.end(), inst2.inst.insts.begin(), inst2.inst.insts.end());
    return data<4>(backward_output_addr,
                   {1, 1, output.dims[1], output.dims[0]});
}


data<4>
Conv1DBackDx(INST_TYPE &inst2,
             data<2> weight,
             data<4> backward_input,
             uint32_t backward_output_addr)
{
    printf("%d[%d, %d]\n", weight.hbmaddr, weight.dims[0], weight.dims[1]);
    printf("%d[%d, %d, %d]\n", backward_input.addr, backward_input.dims[1], backward_input.dims[2], backward_input.dims[3]);
    printf("%d\n", backward_output_addr);
    auto oldRes = inst2.resource;
    inst2.resource = Resource();
    auto outdatasize = weight.dims[0] * backward_input.dims[2];

    inst2.resource.AllocVMem(backward_output_addr + outdatasize);

    auto tInput =
        Alloc<2>(inst2, {backward_input.dims[3], backward_input.dims[2]});

    Transpose(inst2,
              {},
              {},
              {},
              {},
              backward_input.addr,
              backward_input.dims[2],
              backward_input.dims[3],
              tInput.addr);

    auto output = Alloc<2>(inst2, {weight.dims[0], tInput.dims[1]});

    if (true) {
        LinearExHiVwVo(inst2,
                    {},
                    {},
                    {},
                    {},
                    128,
                    128,
                    weight.hbmaddr ,
                    {1, 1, weight.dims[0], weight.dims[1]},
                    tInput.addr,
                    tInput.dims,
                    output.addr);
    }

    Transpose(inst2,
              {},
              {},
              {},
              {},
              output.addr,
              output.dims[0],
              output.dims[1],
              backward_output_addr);

    inst2.resource = oldRes;
    return data<4>(backward_output_addr,
                   {1, 1, output.dims[1], output.dims[0]});
}

data<4>
Conv1DBackDw(INST_TYPE &inst2,
             data<4> forward_input,
             data<4> backward_input,
             uint32_t backward_output_addr)
{
    printf("%d[%d, %d, %d]\n", forward_input.addr, forward_input.dims[1], forward_input.dims[2], forward_input.dims[3]);
    printf("%d[%d, %d, %d]\n", backward_input.addr, backward_input.dims[1], backward_input.dims[2], backward_input.dims[3]);
    printf("%d\n", backward_output_addr);
    std::vector<Instruction *> &instruction_list = inst2;
    Instruction *inst;
    uint32_t usable_addr = backward_output_addr +
                           forward_input.dims[0] * forward_input.dims[1] *
                               forward_input.dims[2] * backward_input.dims[3];
    if (1)
    {
        std::vector<uint32_t> dimSize = {forward_input.dims[0],
                                         forward_input.dims[1],
                                         forward_input.dims[2],
                                         forward_input.dims[3]};
        std::vector<uint32_t> newDims = {0, 1, 3, 2};
        Permute(instruction_list,
                {},
                {},
                {},
                {},
                forward_input.addr,
                dimSize,
                newDims,
                usable_addr);
        uint32_t temp = forward_input.dims[2];
        forward_input.dims[2] = forward_input.dims[3];
        forward_input.dims[3] = temp;
        forward_input.addr = usable_addr;
        usable_addr += forward_input.size();
    }
    data<4> backward_output;
    if (1)
    {
        std::array<uint32_t, 4> leftMatSize = {forward_input.dims[0],
                                               forward_input.dims[1],
                                               forward_input.dims[2],
                                               forward_input.dims[3]};
        std::array<uint32_t, 4> rightMatAddr = {backward_input.dims[0],
                                                backward_input.dims[1],
                                                backward_input.dims[2],
                                                backward_input.dims[3]};
        backward_output.dims[0] = forward_input.dims[0];
        backward_output.dims[1] = forward_input.dims[1];
        backward_output.dims[2] = forward_input.dims[2];
        backward_output.dims[3] = backward_input.dims[3];
        backward_output.addr = usable_addr;
        usable_addr += backward_output.size();
        MatMul(instruction_list,
               {},
               {},
               {},
               {},
               forward_input.addr,
               leftMatSize,
               backward_input.addr,
               rightMatAddr,
               backward_output.addr);
    }
    // move to given addr
    uint32_t move_num;
    if (backward_output.size() % kNumberOfSubcores == 0)
    {
        move_num = backward_output.size() / kNumberOfSubcores;
    }
    else
    {
        move_num = backward_output.size() / kNumberOfSubcores + 1;
    }
    if (move_num == 0)
        move_num++;
    for (uint32_t i = 0; i < move_num; i++)
    {
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(
                Instruction::IMMEDIATE0,
                HelperGetValue((backward_output.addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .second);
            inst->SetImmediateValue(
                Instruction::IMMEDIATE1,
                HelperGetValue((backward_output.addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .first);
            ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
            inst->SetOperationState(Instruction::SCALARONE, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorLoadOperationState
                vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(
                Instruction::IMMEDIATE0,
                HelperGetValue((backward_output_addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .second);
            inst->SetImmediateValue(
                Instruction::IMMEDIATE1,
                HelperGetValue((backward_output_addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .first);
            ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
            inst->SetOperationState(Instruction::SCALARONE, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorStoreOperationState
                vstore(V_STORE_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
            inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
    }
    backward_output.addr = backward_output_addr;
    return backward_output;
}

data<1>
Conv1DBackDb(INST_TYPE &inst2,
             data<4> dx,
             uint32_t output_addr)
{
  std::vector<Instruction *> &instruction_list = inst2;
  Instruction *inst;
  uint32_t size = dx.dims[3];
  data<1> out(output_addr,{size});
  uint32_t time = (size + 1023) / 1024;
  for(uint32_t t = 0; t < time ; t++){
    uint32_t id_mask = ((dx.dims[3] / 128) < 8) ? pow(2, (dx.dims[3] / 128)) - 1 : pow(2, 8) - 1;
    if(t != time - 1) id_mask = pow(2, 8) - 1;
    VReg first_reg = inst2.AllocVReg("");
    if (1)
    {
      Instruction *inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((dx.addr + t * kNumberOfSubcores) / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((dx.addr + t * kNumberOfSubcores)/ kVMemSeg).first);
      inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARTWO, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, first_reg.id, 1, 2, 4, 0, 5);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    
    for(int i = 0; i < dx.dims[0]; i++){
      for(int j = 0; j < dx.dims[1]; j++){
        for(int k = 0; k < dx.dims[2]; k++){
          if(i == 0 && j == 0 && k == 0) continue;
          VReg second_reg = inst2.AllocVReg("");
          if (1)
          {
            Instruction *inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((dx[i][j][k].addr + t * kNumberOfSubcores) / kVMemSeg).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((dx[i][j][k].addr + t * kNumberOfSubcores) / kVMemSeg).first);
            inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
            ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
            inst->SetOperationState(Instruction::SCALARTWO, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, second_reg.id, 1, 2, 4, 0, 5);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
          if(1) {
            inst = new Instruction();
            VectorOperationState add(V_F32_ADDITION, 0, first_reg.id, second_reg.id, first_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst); 
          }
        }
      }
    }

    if (1)
    {
      Instruction *inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((out.addr + t * kNumberOfSubcores) / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((out.addr + t * kNumberOfSubcores) / kVMemSeg).first);
      inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARTWO, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, first_reg.id, 1, 2, 4, 0, 5);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  
  return out;
}


data<4>
MatMulDxBackward(INST_TYPE &inst2,
                 data<4> forward_input,
                 data<4> backward_input,
                 uint32_t backward_output_addr)
{
    Instruction *inst;
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    uint32_t usable_addr = backward_output_addr +
                           forward_input.dims[0] * forward_input.dims[1] *
                               forward_input.dims[2] * backward_input.dims[3];
    usable_addr = AlignTo128Bytes(usable_addr);
    if (1)
    {
        std::vector<uint32_t> dimSize = {forward_input.dims[0],
                                         forward_input.dims[1],
                                         forward_input.dims[2],
                                         forward_input.dims[3]};
        std::vector<uint32_t> newDims = {0, 1, 3, 2};
        Permute(instruction_list,
                {},
                {},
                {},
                {},
                forward_input.addr,
                dimSize,
                newDims,
                usable_addr);
        uint32_t temp = forward_input.dims[2];
        forward_input.dims[2] = forward_input.dims[3];
        forward_input.dims[3] = temp;
        forward_input.addr = usable_addr;
        usable_addr += forward_input.size();
        usable_addr = AlignTo128Bytes(usable_addr);
    }

    data<4> backward_output;
    if (1)
    {
        std::array<uint32_t, 4> leftMatSize = {backward_input.dims[0],
                                               backward_input.dims[1],
                                               backward_input.dims[2],
                                               backward_input.dims[3]};
        std::array<uint32_t, 4> rightMatAddr = {forward_input.dims[0],
                                                forward_input.dims[1],
                                                forward_input.dims[2],
                                                forward_input.dims[3]};
        backward_output.dims[0] = forward_input.dims[0];
        backward_output.dims[1] = forward_input.dims[1];
        backward_output.dims[2] = backward_input.dims[2];
        backward_output.dims[3] = forward_input.dims[3];
        backward_output.addr = usable_addr;
        usable_addr += backward_output.size();
        usable_addr = AlignTo128Bytes(usable_addr);

        MatMul(instruction_list,
               {},
               {},
               {},
               {},
               backward_input.addr, // 1 12 13 64
               leftMatSize,
               forward_input.addr, // 1 12 13 13
               rightMatAddr,
               backward_output.addr);
        
    }
    // move to given addr
    uint32_t move_num;
    if (backward_output.size() % kNumberOfSubcores == 0)
    {
        move_num = backward_output.size() / kNumberOfSubcores;
    }
    else
    {
        move_num = backward_output.size() / kNumberOfSubcores + 1;
    }
    if (move_num == 0)
        move_num++;
    for (uint32_t i = 0; i < move_num; i++)
    {
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(
                Instruction::IMMEDIATE0,
                HelperGetValue((backward_output.addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .second);
            inst->SetImmediateValue(
                Instruction::IMMEDIATE1,
                HelperGetValue((backward_output.addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .first);
            ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
            inst->SetOperationState(Instruction::SCALARONE, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorLoadOperationState
                vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(
                Instruction::IMMEDIATE0,
                HelperGetValue((backward_output_addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .second);
            inst->SetImmediateValue(
                Instruction::IMMEDIATE1,
                HelperGetValue((backward_output_addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .first);
            ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
            inst->SetOperationState(Instruction::SCALARONE, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorStoreOperationState
                vstore(V_STORE_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
            inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
    }
    backward_output.addr = backward_output_addr;
    return backward_output;
}

data<4>
MatMulDwBackward(INST_TYPE &inst2,
                 data<4> forward_input,
                 data<4> backward_input,
                 uint32_t backward_output_addr)
{
    Instruction *inst;
    std::vector<Instruction *> &instruction_list = inst2.inst.insts;
    uint32_t usable_addr = backward_output_addr +
                           forward_input.dims[0] * forward_input.dims[1] *
                               forward_input.dims[2] * backward_input.dims[3];
    usable_addr = AlignTo128Bytes(usable_addr);
    if (1)
    {
        std::vector<uint32_t> dimSize = {forward_input.dims[0],
                                         forward_input.dims[1],
                                         forward_input.dims[2],
                                         forward_input.dims[3]};
        std::vector<uint32_t> newDims = {0, 1, 3, 2};
        Permute(inst2,
                {},
                {},
                {},
                {},
                forward_input.addr,
                dimSize,
                newDims,
                usable_addr);
        uint32_t temp = forward_input.dims[2];
        forward_input.dims[2] = forward_input.dims[3];
        forward_input.dims[3] = temp;
        forward_input.addr = usable_addr;
        usable_addr += forward_input.size();
        usable_addr = AlignTo128Bytes(usable_addr);
    }

    data<4> backward_output;
    if (1)
    {
        std::array<uint32_t, 4> leftMatSize = {forward_input.dims[0],
                                               forward_input.dims[1],
                                               forward_input.dims[2],
                                               forward_input.dims[3]};
        std::array<uint32_t, 4> rightMatAddr = {backward_input.dims[0],
                                                backward_input.dims[1],
                                                backward_input.dims[2],
                                                backward_input.dims[3]};
        backward_output.dims[0] = forward_input.dims[0];
        backward_output.dims[1] = forward_input.dims[1];
        backward_output.dims[2] = forward_input.dims[2];
        backward_output.dims[3] = backward_input.dims[3];
        backward_output.addr = usable_addr;
        usable_addr += backward_output.size();
        usable_addr = AlignTo128Bytes(usable_addr);
        MatMul(inst2,
               {},
               {},
               {},
               {},
               forward_input.addr,
               leftMatSize,
               backward_input.addr,
               rightMatAddr,
               backward_output.addr);
    }
    // move to given addr
    uint32_t move_num;
    if (backward_output.size() % kNumberOfSubcores == 0)
    {
        move_num = backward_output.size() / kNumberOfSubcores;
    }
    else
    {
        move_num = backward_output.size() / kNumberOfSubcores + 1;
    }
    if (move_num == 0)
        move_num++;
    for (uint32_t i = 0; i < move_num; i++)
    {
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(
                Instruction::IMMEDIATE0,
                HelperGetValue((backward_output.addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .second);
            inst->SetImmediateValue(
                Instruction::IMMEDIATE1,
                HelperGetValue((backward_output.addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .first);
            ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
            inst->SetOperationState(Instruction::SCALARONE, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorLoadOperationState
                vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(
                Instruction::IMMEDIATE0,
                HelperGetValue((backward_output_addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .second);
            inst->SetImmediateValue(
                Instruction::IMMEDIATE1,
                HelperGetValue((backward_output_addr + i * kNumberOfSubcores) /
                               kVMemSeg)
                    .first);
            ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
            inst->SetOperationState(Instruction::SCALARONE, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorStoreOperationState
                vstore(V_STORE_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
            inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
    }
    backward_output.addr = backward_output_addr;
    return backward_output;
}

namespace __Infra
{
void MemcopyEx(INST_TYPE &inst2,
               const VMem &src,
               uint32_t srcOffset,
               uint32_t srcLineWidth,
               const VMem &dest,
               uint32_t destOffset,
               uint32_t destLineWidth,
               uint32_t singleWidth,
               uint32_t copyTimes);
}

data<2>
Diag(INST_TYPE &inst2, const data<1> &in)
{
    auto len = in.dims[0];
    data<2> out = Alloc<2>(inst2, {len, len});
    out.asVMem(inst2) = 0.0f;
    for (int i = 0; i < len; i++)
    {
        Memcopy(in.asVMem(inst2)[Range(i, i + 1)],
                out[i].asVMem(inst2)[Range(i, i + 1)]);
    }
    return out;
}

data<2>
Trans(INST_TYPE &inst2, const data<2> &in)
{
    data<2> out = Alloc<2>(inst2, {in.dims[1], in.dims[0]});
    Transpose(inst2.inst.insts,
              {},
              {},
              {},
              {},
              in.addr,
              in.dims[0],
              in.dims[1],
              out.addr);
    return out;
}

data<2>
Mul(INST_TYPE &inst2, const data<2> &ina, const data<2> &inb)
{
    data<2> out = Alloc<2>(inst2, {ina.dims[0], inb.dims[1]});
    MatMul(inst2.inst.insts,
           {},
           {},
           {},
           {},
           ina.addr,
           ina.as<4>().dims,
           inb.addr,
           inb.as<4>().dims,
           out.addr);
    return out;
}

template <uint32_t T>
data<T>
Sub(INST_TYPE &inst2, const data<T> &ina, const data<T> &inb)
{
    assert(ina.dims == inb.dims);

    data<T> out = Alloc<T>(inst2, ina.dims);

    auto a = inst2.AllocVReg("ina");
    auto b = inst2.AllocVReg("inb");
    a.isFloat = true;
    b.isFloat = true;

    auto i = 0;
    for (; i < ina.size() / 1024; i++)
    {
        a = ina.asVMem(inst2)[Range(i * 1024, (i + 1) * 1024)];
        b = inb.asVMem(inst2)[Range(i * 1024, (i + 1) * 1024)];
        inst2(VSubF, a.id, b.id, a.id);
        out.asVMem(inst2)[Range(i * 1024, (i + 1) * 1024)] = a;
    }

    auto size = ina.size();
    a[Range(0, size % 1024)] = ina.asVMem(inst2)[Range(i * 1024, size)];
    b[Range(0, size % 1024)] = inb.asVMem(inst2)[Range(i * 1024, size)];
    inst2(VSubF, a.id, b.id, a.id);
    out.asVMem(inst2)[Range(i * 1024, size)] = a[Range(0, size % 1024)];

    return out;
}

template<uint32_t T>
void
Free(Inst2 &inst2, data<T> &d)
{
    auto vmem = d.asVMem(inst2);
    inst2.FreeVMem(&vmem);
}

data<4>
SoftmaxBack(INST_TYPE &inst2,
            const data<4> &forward_output,
            const data<4> &backword_input,
            uint32_t output_addr)
{
    auto oldRes = inst2.resource;
    inst2.resource = Resource();
    inst2.resource.AllocVMem(output_addr + forward_output.size());

    uint32_t d0 = forward_output.dims[0];
    uint32_t d1 = forward_output.dims[1];
    uint32_t d2 = forward_output.dims[2];
    uint32_t d3 = forward_output.dims[3];

    data<4> output(output_addr, forward_output.dims);

    for (auto i = 0; i < d0; i++)
    {
        for (auto j = 0; j < d1; j++)
        {
            for (auto k = 0; k < d2; k++)
            {
                // [d3]
                auto temp = forward_output[i][j][k];
                // [d3, d3]
                auto diag_y = Diag(inst2, temp);
                // [d3, 1] * [1, d3] => [d3, d3]
                auto y_trans_mul_y =
                    Mul(inst2, Trans(inst2, temp.as<2>()), temp.as<2>());
                // [d3, d3]
                auto dw_ds = Sub(inst2, diag_y, y_trans_mul_y);
                // [1, bd3] x [d3, d3] => [1, d3]
                auto out = Mul(inst2, backword_input[i][j][k].as<2>(), dw_ds);

                Memcopy(out[0].asVMem(inst2), output[i][j][k].asVMem(inst2));

                Free(inst2, diag_y);
                Free(inst2, y_trans_mul_y);
                Free(inst2, dw_ds);
                Free(inst2, out);
            }
        }
    }

    inst2.resource = oldRes;

    return output;
}

void UpdateWeight(INST_TYPE &inst2,
                  float lr,
                  data<2> weight, // weight.addr为hbm地址
                  data<2> weight_grad,
                  uint32_t addr) // addr后的地址为可操作空间
{
    std::vector<Instruction *> &instruction_list = inst2;
    Instruction *inst;
    float lr_test = lr * std::sqrt(1 - 0.999) / (1 - 0.9);
    float m = 1.0 - 0.9;
    float v = 1.0 - 0.999;
    float eps = 0.00000001;
    inst2.PushResource();
    inst2.Alloc(addr);
    weight.addr = weight.hbmaddr;
    std::cout << "weight.hbmaddr: " << weight.hbmaddr << std::endl;
    // std::cout << "addr: " << addr << std::endl;
    auto size = inst2.resource.BiggestContinuousAvailableVMemSize(128);
    auto tmp = inst2.Alloc(size);
    auto wsize = weight_grad.size();
    // std::cout << "size: " << size << std::endl;
    for (int i = 0; i < (wsize + size - 1) / size; i++)
    {
        auto curSize = ((i + 1) * size > wsize) ? (wsize % size) : size;
        // std::cout << "weight hbm: " << weight.addr + i * size << std::endl;
        // std::cout << "curSize: " << curSize << std::endl;
        uint32_t gradaddr = weight_grad.addr + i * size;
        HBM_TO_VMEM(inst2.inst.insts, weight.addr + i * size, tmp.startAddr, curSize);
        assert(curSize % 1024 == 0);
        uint32_t time = (curSize + 1023) / 1024;
        // std::cout << "time: " << time << std::endl;
        for(int t = 0; t < time; t++){
          uint32_t id_mask = ((size / 128) < 8) ? pow(2, (size / 128)) - 1 : pow(2, 8) - 1;
          if(t != time - 1) id_mask = pow(2, 8) - 1;
          VReg m_reg = inst2.AllocVReg("");
          VReg v_reg = inst2.AllocVReg("");
          VReg bias_reg = inst2.AllocVReg("");
          VReg grad_reg = inst2.AllocVReg("");

          if (1)
          {
            Instruction *inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((tmp.startAddr + t * kNumberOfSubcores) / kVMemSeg).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((tmp.startAddr + t * kNumberOfSubcores)/ kVMemSeg).first);
            inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
            ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
            inst->SetOperationState(Instruction::SCALARTWO, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, bias_reg.id, 1, 2, 4, 0, 5);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
          if (1)
          {
            Instruction *inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((gradaddr + t * kNumberOfSubcores) / kVMemSeg).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((gradaddr + t * kNumberOfSubcores) / kVMemSeg).first);
            inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
            ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
            inst->SetOperationState(Instruction::SCALARTWO, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, grad_reg.id, 1, 2, 4, 0, 5);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
          if (1)
          {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(m).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(m).first);
            VectorOperationState mul(V_F32_MULTIPLICATION, 0, grad_reg.id, 44, m_reg.id);
            inst->SetOperationState(Instruction::VECTORONE, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if (1)
          {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(lr_test).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(lr_test).first);
            VectorOperationState mul(V_F32_MULTIPLICATION, 0, m_reg.id, 44, m_reg.id);
            inst->SetOperationState(Instruction::VECTORONE, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if (1)
          {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(v).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(v).first);
            VectorOperationState mul(V_F32_MULTIPLICATION, 0, grad_reg.id, 44, v_reg.id);
            inst->SetOperationState(Instruction::VECTORONE, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
          if (1)
          {
            inst = new Instruction();
            VectorOperationState mul(V_F32_MULTIPLICATION, 0, grad_reg.id, v_reg.id, v_reg.id);
            inst->SetOperationState(Instruction::VECTORONE, &mul);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, v_reg.id, 0, 0);
            inst->SetOperationState(Instruction::VECTORONE, &sqrt);
            CompleteInstruction(inst);
            instruction_list.push_back(inst); 
          }

          if(1) {
            inst = new Instruction();
            MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, v_reg.id, 0);
            inst->SetOperationState(Instruction::MTR, &urf_pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);            
          }  

          if(1) {
            inst = new Instruction();
            VectorOperationState rcp(V_F32_RECIPROCAL, 0, v_reg.id, 0, 0);
            inst->SetOperationState(Instruction::VECTORONE, &rcp);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, v_reg.id, 0);
            inst->SetOperationState(Instruction::MTR, &urf_pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);            
          }   

          if(1) {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
            VectorOperationState add(V_F32_ADDITION, 0, v_reg.id, 44, v_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &add);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);            
          }

          if(1) {
            inst = new Instruction();
            VectorOperationState rcp(V_F32_RECIPROCAL, 0, v_reg.id, 0, 0);
            inst->SetOperationState(Instruction::VECTORONE, &rcp);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if(1) {
            inst = new Instruction();
            MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, v_reg.id, 0);
            inst->SetOperationState(Instruction::MTR, &urf_pop);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);            
          }     

          if (1)
          {
            inst = new Instruction();
            VectorOperationState mul(V_F32_MULTIPLICATION, 0, v_reg.id, m_reg.id, v_reg.id);
            inst->SetOperationState(Instruction::VECTORONE, &mul);
            VectorOperationState sub(V_F32_SUBTRACTION, 0, bias_reg.id, v_reg.id, bias_reg.id);
            inst->SetOperationState(Instruction::VECTORTWO, &sub);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }

          if (1)
          {
            Instruction *inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((tmp.startAddr + t * kNumberOfSubcores) / kVMemSeg).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((tmp.startAddr + t * kNumberOfSubcores) / kVMemSeg).first);
            inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
            ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
            inst->SetOperationState(Instruction::SCALARTWO, &set_base);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
            VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, bias_reg.id, 1, 2, 4, 0, 5);
            inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
          }
        }
        
        // tmp[OffLen(0, curSize)].BinaryOpSelfAssign(weight_grad.asVMem(inst2)[OffLen(i * size, curSize)], [lr, &inst2](VReg &a, VReg &b, VReg &o)
        //                                            {
        //     inst2(VMulF, b.id, inst2.inst.ImmeF(lr), b.id);
        //     inst2(VSubF, a.id, b.id, o.id); });


        VMEM_TO_HBM(inst2.inst.insts, tmp.startAddr, weight.addr + i * size, curSize);
    }
    std::cout << "weight.hbmaddr: " << weight.hbmaddr << std::endl;
    inst2.PopResource();
}

void UpdateBias(INST_TYPE &inst2,
                float lr,
                data<1> bias,
                data<1> bias_grad,
                uint32_t addr)
{
  std::vector<Instruction *> &instruction_list = inst2;
  Instruction *inst;
  auto alignTo128 = [](uint32_t addr) { return ((addr + 127) / 128) * 128; };
  uint32_t size = bias.size();
  uint32_t usable_addr = addr;
  float lr_test = lr * std::sqrt(1 - 0.999) / (1 - 0.9);
  float m = 1.0 - 0.9;
  float v = 1.0 - 0.999;
  float eps = 0.00000001;
  std::cout << "bias.hbmaddr: " << bias.hbmaddr << std::endl;
  std::cout << "update_lr: " << lr_test << std::endl;
  // std::cout << "m: " << m << std::endl;
  // std::cout << "v: " << v << std::endl;
  // std::cout << "eps: " << eps << std::endl;
  HBM_TO_VMEM(instruction_list, bias.hbmaddr, usable_addr, size);
  bias.addr = usable_addr;
  usable_addr += alignTo128(bias.size());


  uint32_t time = (size + 1023) / 1024;
  for(int t = 0; t < time; t++){
    uint32_t id_mask = ((size / 128) < 8) ? pow(2, (size / 128)) - 1 : pow(2, 8) - 1;
    if(t != time - 1) id_mask = pow(2, 8) - 1;
    VReg m_reg = inst2.AllocVReg("");
    VReg v_reg = inst2.AllocVReg("");
    VReg bias_reg = inst2.AllocVReg("");
    VReg grad_reg = inst2.AllocVReg("");
    if (1)
    {
      Instruction *inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((bias.addr + t * 1024) / 32).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((bias.addr + t * 1024)/ 32).first);
      // inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARTWO, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, bias_reg.id, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((bias_grad.addr + t * 1024) / 32).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((bias_grad.addr + t * 1024) / 32).first);
      // inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARTWO, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, grad_reg.id, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(m).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(m).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, grad_reg.id, 44, m_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(lr_test).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(lr_test).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, m_reg.id, 44, m_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(v).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(v).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, grad_reg.id, 44, v_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, grad_reg.id, v_reg.id, v_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState sqrt(V_F32_SQUAREROOT_RECIPROCAL, 0, v_reg.id, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &sqrt);
      CompleteInstruction(inst);
      instruction_list.push_back(inst); 
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, v_reg.id, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }  

    if(1) {
      inst = new Instruction();
      VectorOperationState rcp(V_F32_RECIPROCAL, 0, v_reg.id, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, v_reg.id, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }   

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(eps).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(eps).first);
      VectorOperationState add(V_F32_ADDITION, 0, v_reg.id, 44, v_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }

    if(1) {
      inst = new Instruction();
      VectorOperationState rcp(V_F32_RECIPROCAL, 0, v_reg.id, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, v_reg.id, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);            
    }     

    if (1)
    {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, v_reg.id, m_reg.id, v_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1)
    {
      inst = new Instruction();
      VectorOperationState sub(V_F32_SUBTRACTION, 0, bias_reg.id, v_reg.id, bias_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &sub);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((bias.addr + t * 1024) / 32).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((bias.addr + t * 1024) / 32).first);
      // inst->SetImmediateValue(Instruction::IMMEDIATE3, id_mask);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, 0);
      inst->SetOperationState(Instruction::SCALARTWO, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, bias_reg.id, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  std::cout << "bias.hbmaddr: " << bias.hbmaddr << std::endl;
  VMEM_TO_HBM(instruction_list, bias.addr, bias.hbmaddr, size);
}


data<3>
linearAddVector(INST_TYPE &inst2,
          data<3> input1,
          data<1> input2,
          uint32_t output)
{
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<3> add_output;
  add_output.dims = input1.dims;
  add_output.addr = output;
  uint32_t move_num;
  if (add_output.size() % kNumberOfSubcores == 0)
  {
    move_num = add_output.size() / kNumberOfSubcores;
  }
  else
  {
    move_num = add_output.size() / kNumberOfSubcores + 1;
  }
  if (move_num == 0)
    move_num++;
  for (uint32_t i = 0; i < move_num; i++)
  {
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0,
          HelperGetValue((input1.addr + i * kNumberOfSubcores) / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1,
          HelperGetValue((input1.addr + i * kNumberOfSubcores) / kVMemSeg).first);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0,
          HelperGetValue((input2.addr + (i * kNumberOfSubcores) % input2.size()) / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1,
          HelperGetValue((input2.addr + (i * kNumberOfSubcores) % input2.size()) / kVMemSeg).first);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      VectorOperationState add(V_F32_ADDITION, 0, 1, 2, 3);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0,
          HelperGetAddress((output + i * kNumberOfSubcores) / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1,
          HelperGetAddress((output + i * kNumberOfSubcores) / kVMemSeg).first);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 3, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  return add_output;
}

data<3> linear(INST_TYPE &inst2, data<3> input, data<2> weights, data<1> bias, int output_addr) {

  Instruction *inst;

  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  data<3> output(output_addr, {input.dims[0], input.dims[1], weights.dims[0]});



  output = matmulT(inst2, input.as<4>(), weights, output.addr)[0];

  std::cout << "output: " << output.addr << std::endl;
  std::cout << "output: " << output.size() << std::endl;

  std::cout << "matmulT" << std::endl;
  data<1> _bias(output_addr + output.size(), bias.dims);
  std::cout << "_bias: " << _bias.addr << std::endl;
  HBM_TO_VMEM(instruction_list, bias.hbmaddr, _bias.addr, bias.size());

  output = linearAddVector(inst2, output, _bias, output.addr);



  return output;

}

data<3> linearNobias(INST_TYPE &inst2, data<3> input, data<2> weights, int output_addr) {

  Instruction *inst;

  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  data<3> output(output_addr, {input.dims[0], input.dims[1], weights.dims[0]});



  output = matmulT(inst2, input.as<4>(), weights, output.addr)[0];

  return output;

}

data<3> MatmulBackDx(INST_TYPE &inst2, data<3> input, data<2> weights, int output_addr)
{
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  std::cout << "MatmulBackDx" << std::endl;
  data<3> output(output_addr, {input.dims[0], input.dims[1], weights.dims[1]});
  output = matmul(inst2, input.as<4>(), weights, output.addr)[0];
  return output;

}

data<3> linearBackDx(INST_TYPE &inst2, data<3> input, data<2> weights, int output_addr)
{
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  data<3> dx(output_addr, {input.dims[0], input.dims[1], weights.dims[1]});
  dx = MatmulBackDx(inst2, input, weights, output_addr);
  return dx;
}

// data<3> linearBackDw(INST_TYPE &inst2, data<3> input, data<2> wegihts, int output_addr)
// {
//   Instruction *inst;
//   std::vector<Instruction *> &instruction_list = inst2.inst.insts;

//   data<3> dw(output_addr, {})
// }

// void build_alibi(INST_TYPE &inst2, data<2> input, uint32_t num_heads, uint32_t output_addr)
// {
//   uint32_t batch_size = input.dims[0];
//   uint32_t seq_length = input.dims[1];

//   uint32_t closest_power_of_2 = std::pow(2, std::floor(std::log2(num_heads)));
//   float base = std::pow(2, -(std::pow(2, -(std::log2(closest_power_of_2) - 3))));

//   std::vector<float> powers(closest_power_of_2);
//   std::iota(powers.begin(), powers.end(), 1);

//   std::vector<float> slopes(closest_power_of_2);
//   for (int i = 0; i < closest_power_of_2; i++)
//   {
//     slopes[i] = std::pow(base, powers[i]);
//   }
  
//   if (closest_power_of_2 != num_heads)
//   {
//     float extra_base = std::pow(2, -(std::pow(2, -(std::log2(2 * closest_power_of_2) - 3))));
//     int num_remaining_heads = std::min(closest_power_of_2, num_heads - closest_power_of_2);
//     std::vector<float> extra_powers(num_remaining_heads);
//     for (int i = 0; i < num_remaining_heads; i++)
//     {
//       extra_powers[i] = 2 * i + 1;
//     }
//     std::vector<float> extra_slopes(num_remaining_heads);
//     for (int i = 0; i < num_remaining_heads; i++)
//     {
//       extra_slopes[i] = std::pow(extra_base, extra_powers[i]);
//     }
//     slopes.insert(slopes.end(), extra_slopes.begin(), extra_slopes.end());
//   }

//   Instruction *inst;
//   std::vector<Instruction *> &instruction_list = inst2.inst.insts;

//   auto readAddr = inst2.AllocSReg();
//   readAddr = input.addr / kNumberOfCores;
//   uint32_t init_vreg = 10;
//   uint32_t sum_vreg = 11;
//   uint32_t test_vreg = 12;
//   uint32_t save_vreg = 13;
//   // 目前是认为attention_mask.size()不会超过128
//   int copyColCnt = 1;
    
//   inst2(VLoadBySRegWithMask, readAddr.id, init_vreg, (1 << copyColCnt) - 1);

//   //arange_tensor = cusum(input) - 1
//   if (1)
//   {
//     uint32_t core_rotate_vreg = 14;
//     if (1)
//     {
//       inst = new Instruction();
//       VectorOperationState move(V_U32_MOVE, 0, 0, init_vreg, sum_vreg);
//       inst->SetOperationState(Instruction::VECTORONE, &move);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }

//     for (int i = 0; i < seq_length - 1; i++)
//     {
//       if (1)
//       {
//         inst = new Instruction();
//         inst->SetImmediateValue(Instruction::IMMEDIATE3, i + 1);
//         MTIOperationState core_rotate(MTI_ROTATE, 0, init_vreg, 5, 0);
//         inst->SetOperationState(Instruction::MTI, &core_rotate);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }

//       if (1)
//       {
//         inst = new Instruction();
//         MTROperationState trg_pop(MTR_READ_TRANSPOSE_RESULT, 0, core_rotate_vreg, 0);
//         inst->SetOperationState(Instruction::MTR, &trg_pop);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }

//       if (1) 
//       {
//         inst = new Instruction();
//         VectorOperationState add(V_F32_ADDITION, 0, sum_vreg, core_rotate_vreg, sum_vreg);
//         inst->SetOperationState(Instruction::VECTORTWO, &add);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }

//     if (1)
//     {
//       inst = new Instruction();
//       VectorOperationState sub(V_F32_SUBTRACTION, 0, sum_vreg, 49, sum_vreg);
//       inst->SetOperationState(Instruction::VECTORTWO, &sub);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }

//     if (1)
//     {
//       inst = new Instruction();
//       VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_vreg, init_vreg, sum_vreg);
//       inst->SetOperationState(Instruction::VECTORONE, &mul);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }
//   }

//   if (1)
//   {
//     int num_load = (num_heads + 8 - 1) / 8;
//     std::vector<VReg> temp;
//     for (uint32_t i = 0; i < 8; i++)
//     {
//       temp.push_back(inst2.AllocVReg(""));
//     }

//     if (1)
//     {
//       inst = new Instruction();
//       VectorOperationState move(V_U32_MOVE, 0, 0, sum_vreg, temp[0].id);
//       inst->SetOperationState(Instruction::VECTORONE, &move);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }

//     for (int i = 0; i < 7; i++)
//     {
//       if (1)
//       {
//         inst = new Instruction();
//         VectorOperationState rotate(V_SUBCORE_ROTATE, 0, temp[i].id, 1, temp[i + 1].id);
//         inst->SetOperationState(Instruction::VECTORONE, &rotate);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }

//     for (int i = 0; i < num_load; i++)
//     {
//       auto writeAddr = inst2.AllocSReg();
//       writeAddr = (output_addr + i * kNumberOfSubcores) / kNumberOfCores; 
//       int Col = (i + 1) > (num_heads / 8) ? (num_heads % 8) : 8;

//       if (1)
//       {
//         inst = new Instruction();
//         VectorOperationState move(V_U32_MOVE, 0, 0, 46, save_vreg);
//         inst->SetOperationState(Instruction::VECTORONE, &move);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }

//       for (int j = 0; j < Col; j++)
//       {
//         if (1)
//         {
//           inst = new Instruction();
//           inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(slopes[i * 8 + j]).second);
//           inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(slopes[i * 8 + j]).first);
//           VectorOperationState mul(V_F32_MULTIPLICATION, 0, temp[j].id, 44, test_vreg);
//           inst->SetOperationState(Instruction::VECTORONE, &mul);
//           VectorOperationState add(V_F32_ADDITION, 0, save_vreg, test_vreg, save_vreg);
//           inst->SetOperationState(Instruction::VECTORTWO, &add);
//           CompleteInstruction(inst);
//           instruction_list.push_back(inst);
//         }
//       }
//       inst2(VStoreBySRegWithMask, save_vreg, writeAddr.id, (1 << Col) - 1);
//     }
//   }
//   return ;
// }


data<3> baddbmm(INST_TYPE &inst2, data<3> input, data<3> batch1, data<3> batch2, float beta, float alpha, int output_addr) {
  data<3> output =  matmulIvWv(inst2, batch1.as<4>(), batch2.as<4>(), output_addr)[0];

  for(int i = 0; i < batch1.dims[0]; i++) {
    for(int j = 0; j < batch1.dims[1]; j++) {
      AddVector(inst2, output[i][j], input[i][0], beta, alpha, output.addr + i * batch1.dims[1] * batch2.dims[2] + j * batch2.dims[2]);
    }
  }

  return output;
}

data<4> maskedFill(INST_TYPE &inst2, data<4> scores, data<4> mask, float value, int output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  // std::cout << "mask.addr: " << mask.addr << " mask.size: " << mask.size() << std::endl;

  int total_size = scores.size();
  int one_use_len;
  auto one_reg = inst2.AllocVReg("");
  auto value_reg = inst2.AllocVReg("");
  inst2(VMov, 48, one_reg.id); 
  inst2(VMov, inst2.inst.ImmeF(value), value_reg.id);

  for(int b1 = 0; b1 < scores.dims[0]; b1++ ) {
    for(int b2 = 0; b2 < scores.dims[1]; b2++) {
      for(int row = 0 ; row < scores.dims[2]; row += 8) {
        uint32_t one_use_row = std::min(scores.dims[2] - row, (uint32_t)8);
        for (int col = 0; col < scores.dims[3]; col += 128)
        {
          uint32_t one_use_col = std::min(scores.dims[3] - col, (uint32_t)128);
          uint32_t src_addr = scores.addr + b1 * scores[0].size() + b2 * scores[0][0].size() + row * scores[0][0][0].size() + col;
          uint32_t mask_addr = mask.addr + row * scores[0][0][0].size() + col;
          uint32_t res_addr = output_addr + b1 * scores[0].size() + b2 * scores[0][0].size() + row * scores[0][0][0].size() + col;

          auto src_reg = inst2.AllocVReg("");
          auto mask_reg = inst2.AllocVReg("");
          auto select_mask = inst2.AllocVMask();
          auto res = inst2.AllocVReg("");

          // std::cout << "mask_addr: " << mask_addr << std::endl;

          Load8_128(inst2, src_reg, one_use_row, one_use_col, src_addr, scores.dims[3]);
          Load8_128(inst2, mask_reg, one_use_row, one_use_col, mask_addr, scores.dims[3]);
          // inst2(VLoad, mask_addr/128, mask_reg.id);
          inst2(VEqS, one_reg.id, mask_reg.id, select_mask.id);
          inst2(VSel, select_mask.id, src_reg.id, value_reg.id, res.id);
          // inst2(VStore, mask_reg.id, res_addr/128);
          Store8_128(inst2, res, one_use_row, one_use_col, res_addr, mask.dims[3]);
        }
      }
    }
  }

  data<4> output(output_addr, scores.dims);
  return output;
}

void build_alibi(INST_TYPE &inst2, data<2> input, uint32_t num_heads, uint32_t output_addr) {
  uint32_t batch_size = input.dims[0];
  uint32_t seq_length = input.dims[1];

  uint32_t closest_power_of_2 = std::pow(2, std::floor(std::log2(num_heads)));
  float base = std::pow(2, -(std::pow(2, -(std::log2(closest_power_of_2) - 3))));

  std::vector<float> powers(closest_power_of_2);
  std::iota(powers.begin(), powers.end(), 1);

  std::vector<float> slopes(closest_power_of_2);
  for (int i = 0; i < closest_power_of_2; i++)
  {
    slopes[i] = std::pow(base, powers[i]);
  }
  
  if (closest_power_of_2 != num_heads)
  {
    float extra_base = std::pow(2, -(std::pow(2, -(std::log2(2 * closest_power_of_2) - 3))));
    int num_remaining_heads = std::min(closest_power_of_2, num_heads - closest_power_of_2);
    std::vector<float> extra_powers(num_remaining_heads);
    for (int i = 0; i < num_remaining_heads; i++)
    {
      extra_powers[i] = 2 * i + 1;
    }
    std::vector<float> extra_slopes(num_remaining_heads);
    for (int i = 0; i < num_remaining_heads; i++)
    {
      extra_slopes[i] = std::pow(extra_base, extra_powers[i]);
    }
    slopes.insert(slopes.end(), extra_slopes.begin(), extra_slopes.end());
  }

  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  auto readAddr = inst2.AllocSReg();
  auto _init_vreg = inst2.AllocVReg("");
  auto sum_vreg = inst2.AllocVReg("");
  auto test_vreg = inst2.AllocVReg("");
  auto save_vreg = inst2.AllocVReg("");
  auto init_vreg = inst2.AllocVReg("");
  auto core_id_reg = inst2.AllocVReg("");
  // 目前是认为attention_mask.size()不会超过128
  int copyColCnt = 1;
  inst2(VCoreId, core_id_reg.id);
  inst2(VLsS, core_id_reg.id, inst2.inst.ImmeS(seq_length), 0);
  inst2(VLoadEx, 0, inst2.inst.ImmeS(input.addr / kNumberOfCores), _init_vreg.id, 0b1111111);
  inst2(VS2F, _init_vreg.id, init_vreg.id);

  // arange_tensor = cusum(input) - 1
  if (1)
  {
    auto core_rotate_vreg = inst2.AllocVReg("");
    if (1)
    {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, init_vreg.id, sum_vreg.id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for (int i = 0; i < seq_length - 1; i++)
    {
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE3, i + 1);
        MTIOperationState core_rotate(MTI_ROTATE, 0, init_vreg.id, 5, 0);
        inst->SetOperationState(Instruction::MTI, &core_rotate);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1)
      {
        inst = new Instruction();
        MTROperationState trg_pop(MTR_READ_TRANSPOSE_RESULT, 0, core_rotate_vreg.id, 0);
        inst->SetOperationState(Instruction::MTR, &trg_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1) 
      {
        inst = new Instruction();
        VectorOperationState add(V_F32_ADDITION, 0, sum_vreg.id, core_rotate_vreg.id, sum_vreg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &add);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }

    if (1)
    {
      inst = new Instruction();
      VectorOperationState sub(V_F32_SUBTRACTION, 0, sum_vreg.id, 49, sum_vreg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &sub);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1)
    {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, sum_vreg.id, init_vreg.id, sum_vreg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }

  if (1)
  {
    int num_load = (num_heads + 8 - 1) / 8;
    std::vector<VReg> temp;
    for (uint32_t i = 0; i < 8; i++)
    {
      temp.push_back(inst2.AllocVReg(""));
    }

    if (1)
    {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, sum_vreg.id, temp[0].id);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    for (int i = 0; i < 7; i++)
    {
      if (1)
      {
        inst = new Instruction();
        VectorOperationState rotate(V_SUBCORE_ROTATE, 0, temp[i].id, 1, temp[i + 1].id);
        inst->SetOperationState(Instruction::VECTORONE, &rotate);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }

    for (int i = 0; i < num_load; i++)
    {
      auto writeAddr = inst2.AllocSReg();
      int Col = (i + 1) > (num_heads / 8) ? (num_heads % 8) : 8;

      inst2(SMov, inst2.inst.ImmeS((output_addr + i * kNumberOfSubcores) / kNumberOfCores), writeAddr.id);

      if (1)
      {
        inst = new Instruction();
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, save_vreg.id);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      for (int j = 0; j < Col; j++)
      {
        if (1)
        {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(slopes[i * 8 + j]).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(slopes[i * 8 + j]).first);
          VectorOperationState mul(V_F32_MULTIPLICATION, 0, temp[j].id, 44, test_vreg.id);
          inst->SetOperationState(Instruction::VECTORONE, &mul);
          VectorOperationState add(V_F32_ADDITION, 0, save_vreg.id, test_vreg.id, save_vreg.id);
          inst->SetOperationState(Instruction::VECTORTWO, &add);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      Store8_128(inst2, save_vreg, Col, seq_length, output_addr + i * kNumberOfSubcoresPerCore * seq_length, seq_length);
    }
  }
  return ;
}

data<4> _prepare_attn_mask(INST_TYPE &inst2, 
                        data<2> attention_mask, 
                        std::tuple<uint32_t, uint32_t> input_shape, 
                        uint32_t past_key_values_length, 
                        uint32_t output_addr) {
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<4> output(output_addr, {attention_mask.dims[0], 1, attention_mask.dims[1], attention_mask.dims[1]});

  auto one_reg = inst2.AllocVReg("");
  auto core_id_reg = inst2.AllocVReg("");
  inst2(VCoreId, core_id_reg.id);
  inst2(VMov, 48, one_reg.id);

  for(int i = 0; i < std::get<1>(input_shape); i += kNumberOfSubcoresPerCore) {
    int use_row = std::min(uint32_t(kNumberOfSubcoresPerCore), std::get<1>(input_shape) - i);

    auto i_reg = inst2.AllocVReg("");
    inst2(VMov, inst2.inst.ImmeS(i), i_reg.id);

    for (int j = 0; j < std::get<1>(input_shape); j += kNumberOfCores)
    {
      int use_col = std::min(uint32_t(kNumberOfCores), std::get<1>(input_shape) - j);
      auto cmp_data_reg = inst2.AllocVReg("");

      inst2(VAddS, core_id_reg.id, inst2.inst.ImmeS(j), cmp_data_reg.id);
      
      for(int now_i = 0; now_i < use_row; now_i++) {
        auto temp_mask = inst2.AllocVMask();
        auto res_reg = inst2.AllocVReg("");
        
        inst2(VGeS, i_reg.id, cmp_data_reg.id, temp_mask.id);
        inst2(VSel, temp_mask.id, one_reg.id, 46, res_reg.id);
        inst2(VAddS, i_reg.id, 48, i_reg.id);
        Store8_128(inst2, res_reg, 1, use_col, output_addr + (i + now_i) * std::get<1>(input_shape) + j, std::get<1>(input_shape));
      }
    }
  }

  auto _make_causal_mask = [&](std::tuple<int, int> input_shape, int past_key_values_length, int output_addr)
  {
    int batch_size = std::get<0>(input_shape), target_length = std::get<1>(input_shape);
    data<2> mask(output_addr, {target_length, target_length + past_key_values_length});

    for(int i = 0; i < target_length; i ++) {
      auto i_reg = inst2.AllocVReg("");
      inst2(VMov, inst2.inst.ImmeS(i), i_reg.id);

      for(int j = past_key_values_length; j < past_key_values_length + target_length; j += kNumberOfSubcores) {
        uint32_t use_col = std::min(kNumberOfSubcores, past_key_values_length + target_length - j);

        auto cmp_data_reg = inst2.AllocVReg("");
        auto temp_mask = inst2.AllocVMask();
        auto res_reg = inst2.AllocVReg("");

        inst2(VAddS, core_id_reg.id, inst2.inst.ImmeS(j), cmp_data_reg.id);

        if(use_col == kNumberOfSubcores) {
          inst2(VGeS, i_reg.id, cmp_data_reg.id, temp_mask.id);
          inst2(VSel, temp_mask.id, one_reg.id, 46, res_reg.id);
          Store8_128(inst2, res_reg, kNumberOfSubcoresPerCore, kNumberOfCores, output_addr + i * (past_key_values_length + target_length) + j, kNumberOfCores);
        }
        else {
          uint32_t col_mod = use_col % kNumberOfCores;
          uint32_t temp_col = use_col - col_mod;
          uint32_t now_row = temp_col / kNumberOfCores;
          inst2(VGeS, i_reg.id, cmp_data_reg.id, temp_mask.id);
          inst2(VSel, temp_mask.id, one_reg.id, 46, res_reg.id);
          Store8_128(inst2, res_reg, now_row, kNumberOfCores, output_addr + i * (past_key_values_length + target_length) + j, kNumberOfCores);
          for(int _x = 0; _x < col_mod; _x++)
            inst2(VSubRotL, res_reg.id, res_reg.id);
          Store8_128(inst2, res_reg, 1, col_mod, output_addr + i * (past_key_values_length + target_length) + j + temp_col, col_mod);
        }
      }
    }

    if(past_key_values_length > 0) {
      auto zero_reg = inst2.AllocVReg("");
      inst2(VMov, 46, zero_reg.id);
      for (int i = 0; i < target_length; i++)
      {
        for(int j = 0; j < kNumberOfSubcores; j += kNumberOfSubcores) {
          uint32_t use_col = std::min(kNumberOfSubcores, kNumberOfSubcores - j);

          if(use_col == kNumberOfSubcores) {
            // inst2(VStore, zero_reg.id, mask.addr + i * mask.dims[1] + j);
            Store8_128(inst2, zero_reg, kNumberOfSubcoresPerCore, kNumberOfCores, mask.addr + i * mask.dims[1] + j, kNumberOfCores);
          }
          else{
            auto temp_mask = inst2.AllocVMask();
            auto len_reg = inst2.AllocVReg("");
            inst2(VMov, inst2.inst.ImmeS(use_col), len_reg.id);
            inst2(VLsS, len_reg.id, core_id_reg.id, temp_mask.id);
            inst2(VStM, temp_mask.id, zero_reg.id, mask.addr + i * mask.dims[1] + j);
          }
        }
      }
    }

    for(int i = 0; i < batch_size; i++) {
      
    }
  };

  auto _expand_mask = [&one_reg, &core_id_reg, &inst2](data<2> mask, int tgt_length, uint32_t output_addr)
  {
    int batch_size = mask.dims[0], src_length = mask.dims[1];
    data<4> output(output_addr, {batch_size, 1, tgt_length, src_length});

    for(int i = 0; i < batch_size; i++)
    {
      for(int j = 0; j < tgt_length; j++)
      {
        for(int k = 0; k < src_length; k += kNumberOfCores)
        {
          auto mask_reg = inst2.AllocVReg("");
          int use_data = std::min(src_length - k, kNumberOfCores);
          uint32_t mask_addr = mask.addr + i * src_length + k;
          uint32_t dest_addr = output_addr + i * tgt_length * src_length + j * src_length + k;
          Load8_128(inst2, mask_reg, 1, use_data, mask_addr, src_length);
          inst2(VXorU, mask_reg.id, 48, mask_reg.id);
          Store8_128(inst2, mask_reg, 1, use_data, dest_addr, src_length);
        }
      }
    }
    return output;
  };

  return output;
}

// forward_input Vmem   backward_input Vmem     output HBM
void MatMulUpdateWeight(Inst2& inst2, data<3> forward_input, data<3> backward_input, data<2> weight, float update_lr, uint32_t output_addr)
{
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  uint32_t weights_use_vmem_size = (kVectorDataMemorySize - output_addr) / 2;
  uint32_t forward_use_row =  weights_use_vmem_size  / backward_input.dims[2];
  std::cout << "weights_use_vmem_size: " << weights_use_vmem_size << std::endl;
  std::cout << "forward_use_row: " << forward_use_row << std::endl;

  int num = forward_input.dims[1] / forward_use_row;
  for(int i = 0; i <= num; i++)
  {
    std::cout << "嘿嘿~" << std::endl;
    int forward_row_now = (forward_input.dims[1] - i * forward_use_row) >= forward_use_row ? forward_use_row : (forward_input.dims[1] - i * forward_use_row);
    std::cout << "forward_row_now: " << forward_row_now << std::endl;
    data<3> now_forward(forward_input.addr + i * forward_use_row * forward_input.dims[2], {forward_input.dims[0], forward_row_now, forward_input.dims[2]});

    data<3> output = matmulIvWv(inst2, now_forward.as<4>(), backward_input.as<4>(), output_addr)[0];
    data<2> now_weight;
    now_weight.hbmaddr = weight.hbmaddr + i * forward_use_row * backward_input.dims[2];
    now_weight.dims = {forward_row_now, backward_input.dims[2]};
    std::cout << "now weight: " << now_weight.dims[0] << ' ' << now_weight.dims[1] << std::endl;
    std::cout << "now weight: " << now_weight.hbmaddr << std::endl;
    std::cout << "output: " << output.dims[0] << ' ' << output.dims[1] << ' ' << output.dims[2] << std::endl; 
    // VMEM_TO_HBM(instruction_list, output.addr, weight.hbmaddr + i * forward_use_row * backward_input.dims[2], output.size());
    UpdateWeight(inst2, update_lr, now_weight, output[0], output.addr + output.size());
  }
  return ;
}

void Division(Inst2& inst2, data<3> input, float num)
{
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  uint32_t unum = *(uint32_t*)(&num);
  std::cout << "unum: " << unum << std::endl;
  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(unum).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(unum).first);
    VectorOperationState move(V_U32_MOVE, 0, 0, 44, 0);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  // if (1)
  // {
  //   inst = new Instruction();
  //   VectorOperationState rcp(V_F32_RECIPROCAL, 0, 0, 0, 0);
  //   inst->SetOperationState(Instruction::VECTORONE, &rcp);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }
  // if(1) {
  //   inst = new Instruction();
  //   MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 0, 0);
  //   inst->SetOperationState(Instruction::MTR, &urf_pop);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);            
  // }     
  uint32_t nums = input.dims[0] * input.dims[1] * input.dims[2] / 1024;
  for (uint32_t i = 0; i < nums; i++)
  {
    uint32_t vmem = input.addr + i * 1024;
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(vmem / 32).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(vmem / 32).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);      
    }
    if (1)
    {
      inst = new Instruction();
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, 1, 0, 1);
      inst->SetOperationState(Instruction::VECTORONE, &mul);

      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(vmem / 32).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(vmem / 32).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }
  }
  return ;
}

data<3> dropout(INST_TYPE inst2, data<3> input, float p, uint32_t output_addr) {
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;

  std::cout << "dropout float: " << p << std::endl;
  data<3> output(output_addr, {input.dims[0], input.dims[1], input.dims[2]});

  auto p_reg = inst2.AllocVReg("");
  auto core_id_reg = inst2.AllocVReg("");
  inst2(VCoreId, core_id_reg.id);
  inst2(VMov, inst2.inst.ImmeF(p), p_reg.id);

  //丢弃概率为 0 时，只需要将输入搬运至输出地址
  if(p == 0) {
    if  (input.addr != output_addr) Memcopy(input.asVMem(inst2), output.asVMem(inst2));
    return output;
  }
  //丢弃概率为 1 时，输出全部置为 0
  else if(p == 1.0){
    auto zero_reg = inst2.AllocVReg("");
    for (int i = 0; i < input.size(); i += kNumberOfSubcores) {
      int use_num = std::min(input.size() - i, (uint32_t)kNumberOfSubcores);
      auto store_mask = inst2.AllocVMask();
      inst2(VLsS, core_id_reg.id, inst2.inst.ImmeS(use_num), store_mask.id);
      inst2(VStoreEx, store_mask.id, zero_reg.id, inst2.inst.ImmeU((output_addr + i) / kNumberOfCores), 0b11111111);
    }
  }
  else {
    auto zero_reg = inst2.AllocVReg("");
    inst2(VMov, 46, zero_reg.id);
    for (int i = 0; i < input.size(); i += kNumberOfSubcores) {
      int use_num = std::min(input.size() - i, (uint32_t)kNumberOfSubcores);
      auto rand_reg = inst2.AllocVReg("");
      if (1) 
      {
        inst = new Instruction();
        VectorOperationState V_GET_COREID(V_GET_V_CORE_ID, 0, 0, 0, 1);
        inst->SetOperationState(Instruction::VECTORONE, &V_GET_COREID);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1)
      {
        inst = new Instruction();
        VectorOperationState set_seed(V_RNG_RESEED, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORTWO, &set_seed);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      if (1)
      {
        inst = new Instruction();
        VectorOperationState read_seed(V_RNG_READ_SEED, 0, 0, 0, rand_reg.id);
        inst->SetOperationState(Instruction::VECTORTWO, &read_seed);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }

      AddNoop(10, instruction_list);
      auto dropout_mask = inst2.AllocVMask();
      inst2(VRNG, rand_reg.id);
      inst2(VAndU, rand_reg.id, inst2.inst.ImmeS(0b1111111), rand_reg.id);
      inst2(VS2F, rand_reg.id, rand_reg.id);
      inst2(VMulF, rand_reg.id, inst2.inst.ImmeF(1.0 / 100), rand_reg.id);
      inst2(VLsF, rand_reg.id, inst2.inst.ImmeF(p), dropout_mask.id);

      if(use_num < kNumberOfSubcores) {
        auto load_store_mask = inst2.AllocVMask();
        auto load_reg = inst2.AllocVReg("");
        inst2(VLsS, core_id_reg.id, inst2.inst.ImmeS(use_num), load_store_mask.id);
        inst2(VLoadEx, load_store_mask.id, inst2.inst.ImmeS((input.addr + i) / kNumberOfCores), load_reg.id, 0b11111111);
        inst2(VSel, dropout_mask.id, load_reg.id, zero_reg.id, load_reg.id);
        inst2(VStoreEx, load_store_mask.id, load_reg.id, inst2.inst.ImmeU((output_addr + i) / kNumberOfCores), 0b11111111);
      }
      else{
        auto load_reg = inst2.AllocVReg("");
        inst2(VLoad, (input.addr + i) / kNumberOfCores, load_reg.id);
        inst2(VSel, dropout_mask.id, load_reg.id, zero_reg.id, load_reg.id);
        inst2(VStore, load_reg.id, (output_addr + i) / kNumberOfCores);
      }
    }
  }

  return output;
}

data<3> dropout_add(INST_TYPE inst2, BLOOMConfig config, data<3> input1, data<3> input2, uint32_t output_addr)
{
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  assert(input1.dims[0] == input2.dims[0]);
  assert(input1.dims[1] == input2.dims[1]);
  assert(input1.dims[2] == input2.dims[2]);
  data<3> output(output_addr, {input1.dims[0], input1.dims[1], input1.dims[2]});

  if (config.training) {
    std::cout << "dropout add" << std::endl;
    output = dropout(inst2, input1, config.hidden_dropout, output_addr);
  }
  output = AddVector(instruction_list, output.as<4>(), input2.as<4>(), output_addr)[0];
  
  return output;
}


void HBM_TO_SMEM(std::vector<Instruction*> &instruction_list, uint32_t input_addr,

  uint32_t dest_addr, uint32_t length, bool is_sync){

  // total_data+=length;

  Instruction* inst;

  int misc = 0b0001000101000000;
  int sync_register = 0;
  if(1){

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE0,

      HelperGetAddress(input_addr/128).second);

    inst->SetImmediateValue(Instruction::IMMEDIATE1,

      HelperGetAddress(input_addr/128).first);

    ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 6);

    inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }



  if(1){

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE0,

      HelperGetAddress(dest_addr/128).second);

    inst->SetImmediateValue(Instruction::IMMEDIATE1,

      HelperGetAddress(dest_addr/128).first);

    ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 7);

    inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }



  if(1){

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE0,

      HelperGetAddress(length/128).second);

    inst->SetImmediateValue(Instruction::IMMEDIATE1,

      HelperGetAddress(length/128).first);

    ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 8);

    inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  if(is_sync){

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE2, 1+sync_register);

    MiscOperationState set_sync(MISC_SET_SYNC_FLAG, 0, 0, 2, 4);

    inst->SetOperationState(Instruction::MISC, &set_sync);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);    

  }

  if(is_sync) {

    Instruction* inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE1, 16385+sync_register);

    ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 6, 8, 7, 33, misc);

    inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }else{

    Instruction* inst = new Instruction();

    ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 6, 8, 7, 46, misc);

    inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);    

  }

  if(is_sync) {

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE2, 1+sync_register);

    MiscOperationState sync(MISC_SYNC, 0, 0, 5, 4);

    inst->SetOperationState(Instruction::MISC, &sync);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);



    inst = new Instruction();

    ScalarOperationState fence(S_FENCE, 0, 0, 0, 0);

    inst->SetOperationState(Instruction::SCALARONE, &fence);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);
  }
  AddNoop(1, instruction_list);
}