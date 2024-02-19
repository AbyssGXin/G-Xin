#include "utils.h"

uint32_t AlignTo128Bytes(uint32_t _size) {
  return (~0x7F) & (_size + 0x7F);
}
void AddNoop(unsigned int number, std::vector<Instruction *> &bundle)
{
    for (int i = 0; i < number; ++i)
    {
        Instruction *inst = new Instruction();
        ScalarOperationState scalar;
        inst->SetOperationState(Instruction::SCALARONE, &scalar);
        inst->SetOperationState(Instruction::SCALARTWO, &scalar);
        VectorOperationState vector;
        inst->SetOperationState(Instruction::VECTORONE, &vector);
        inst->SetOperationState(Instruction::VECTORTWO, &vector);
        VectorLoadOperationState vectorload;
        inst->SetOperationState(Instruction::VECTORLOAD, &vectorload);
        VectorStoreOperationState vectorstore;
        inst->SetOperationState(Instruction::VECTORSTORE, &vectorstore);
        MTIOperationState mti;
        inst->SetOperationState(Instruction::MTI, &mti);
        MTROperationState mtr;
        inst->SetOperationState(Instruction::MTR, &mtr);
        MiscOperationState misc;
        inst->SetOperationState(Instruction::MISC, &misc);
        bundle.push_back(inst);
    }
}

void
CompleteInstruction(Instruction *instruction)
{
    for (unsigned int i = 0; i < Instruction::NUM_OPERATIONS_INSTRUCTION; i++)
    {
        if (instruction->GetOperationState(Instruction::OperationSequence(i)) !=
            nullptr)
            continue;
        ScalarOperationState scalar;
        VectorOperationState vector;
        VectorLoadOperationState vectorload;
        VectorStoreOperationState vectorstore;
        MTIOperationState mti;
        MTROperationState mtr;
        MiscOperationState misc;
        switch (Instruction::OperationSequence(i))
        {
        case Instruction::SCALARONE:
            instruction->SetOperationState(Instruction::SCALARONE, &scalar);
            break;
        case Instruction::SCALARTWO:
            instruction->SetOperationState(Instruction::SCALARTWO, &scalar);
            break;
        case Instruction::VECTORONE:
            instruction->SetOperationState(Instruction::VECTORONE, &vector);
            break;
        case Instruction::VECTORTWO:
            instruction->SetOperationState(Instruction::VECTORTWO, &vector);
            break;
        case Instruction::VECTORLOAD:
            instruction->SetOperationState(Instruction::VECTORLOAD,
                                           &vectorload);
            break;
        case Instruction::VECTORSTORE:
            instruction->SetOperationState(Instruction::VECTORSTORE,
                                           &vectorstore);
            break;
        case Instruction::MTI:
            instruction->SetOperationState(Instruction::MTI, &mti);
            break;
        case Instruction::MTR:
            instruction->SetOperationState(Instruction::MTR, &mtr);
            break;
        case Instruction::MISC:
            instruction->SetOperationState(Instruction::MISC, &misc);
            break;
        default:
            break;
        }
    }
    return;
}

void
HBM_TO_VMEM(std::vector<Instruction *> &instruction_list,
            uint32_t input_addr,
            uint32_t dest_addr,
            uint32_t length)
{
    const int callCnt = CallCount(__FUNCTION__);
    // std::cout << "HTV:  input_addr: " << input_addr << ", dest_addr: " << dest_addr << ", length: " << length << std::endl;
    bool safeCall = input_addr % 128 == 0 && dest_addr % 128 == 0;
    if (ShowFuncCallInfo() || (!safeCall))
    {
        std::clog << (safeCall ? COLOR::SHIORI : COLOR::SETSUNA)
                  << "FnCall: HbmToVMem#" << callCnt << "(@" << input_addr
                  << "[" << length << "]) => @" << dest_addr << COLOR::WHITE
                  << std::endl;
    }

    // std::cout << "input_addr: " << input_addr << std::endl;
    // std::cout << "dest_addr: " << dest_addr << std::endl; 
    // std::cout << "length: " << length << std::endl;
    assert(input_addr % 128 == 0);
    assert(dest_addr % 128 == 0);
    assert(length % 128 == 0);

    if (length % 128 != 0)
    {
        length = ((length + 127) / 128) * 128;
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON << "LENGTH UP-ALIGN TO 128: " << length
                      << COLOR::WHITE << std::endl;
        }
    }

    int sync_register = 0;
    // total_data+=length;
    Instruction *inst;
    int misc = 0b0001000100000000;
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(input_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(input_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 6);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(dest_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(dest_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 7);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(length / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(length / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 8);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    // 清空sync_register
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
        MiscOperationState set_sync(MISC_SET_SYNC_FLAG, 0, 0, 2, 4);
        inst->SetOperationState(Instruction::MISC, &set_sync);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    // 
    if (1)
    {
        Instruction *inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE1, 16385 + sync_register);
        ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 6, 8, 7, 33, misc);
        inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    for (int i = 0; i < 1; i++)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
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

void HBM_TO_VMEM_Stride(std::vector<Instruction *> &instruction_list,
                    uint32_t input_addr,
                    uint32_t dest_addr,
                    uint32_t length,
                    uint32_t hbmstride,
                    uint32_t vmemstride)
{
  const int callCnt = CallCount(__FUNCTION__);
  bool safeCall = input_addr % 128 == 0 && dest_addr % 128 == 0;
  if (ShowFuncCallInfo() || (!safeCall))
  {
    std::clog << (safeCall ? COLOR::SHIORI : COLOR::SETSUNA)
              << "FnCall: HBMTOVMEM#" << callCnt << "(@" << input_addr
              << "[" << length << "]) => @" << dest_addr << COLOR::WHITE
              << std::endl; 
  }

  assert(input_addr % 128 == 0);
  assert(dest_addr % 128 == 0);

  int sync_register = 0;
  Instruction *inst;
  int misc = 0b0001000100000000;

  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 
                            HelperGetAddress(input_addr / 128).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1,
                            HelperGetAddress(input_addr / 128).first);
    ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 6);
    inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0,
                            HelperGetAddress(dest_addr / 128).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1,
                            HelperGetAddress(dest_addr / 128).first);
    ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 7);
    inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  
  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0,
                          HelperGetAddress(length / 128).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1,
                          HelperGetAddress(length / 128).first);
    ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 8);
    inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  if(1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, hbmstride);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, vmemstride);
    ScalarOperationState load_reg_1(S_U32_MOVE, 0, 0, 32, 4);
    inst->SetOperationState(Instruction::SCALARONE, &load_reg_1);
    ScalarOperationState load_reg_2(S_U32_MOVE, 0, 0, 33, 5);
    inst->SetOperationState(Instruction::SCALARTWO, &load_reg_2);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  if(1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
    MiscOperationState set_sync(MISC_SET_SYNC_FLAG, 0, 0, 2, 4);
    inst->SetOperationState(Instruction::MISC, &set_sync);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  if(1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 4);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1, 5);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, 16385 + sync_register);
    ScalarOperationState load_dma_stride(S_STRIDED_DMA, 0, 6, 8, 7, 33, misc, 0, 0);
    inst->SetOperationState(Instruction::SCALARONE, &load_dma_stride);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  for (int i = 0; i < 1; i++)
  {
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
      MiscOperationState sync(MISC_SYNC, 0, 0 ,5, 4);
      inst->SetOperationState(Instruction::MISC, &sync);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      ScalarOperationState fence(S_FENCE, 0, 0, 0, 0);
      inst->SetOperationState(Instruction::SCALARONE, &fence);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  AddNoop(1, instruction_list);
}

void
VMEM_TO_HBM(std::vector<Instruction *> &instruction_list,
            uint32_t input_addr,
            uint32_t dest_addr,
            uint32_t length)
{
    const int callCnt = CallCount(__FUNCTION__);

    bool safeCall = input_addr % 128 == 0 && dest_addr % 128 == 0;

    if (ShowFuncCallInfo() || (!safeCall))
    {
        std::clog << (safeCall ? COLOR::SHIORI : COLOR::SETSUNA)
                  << "FnCall: VMemToHbm#" << callCnt << "(@" << input_addr
                  << "[" << length << "]) => @" << dest_addr << COLOR::WHITE
                  << std::endl;
    }

    assert(input_addr % 128 == 0);
    assert(dest_addr % 128 == 0);
    // assert(length % 128 == 0);

    if (length % 128 != 0)
    {
        length = ((length + 127) / 128) * 128;
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON << "LENGTH UP-ALIGN TO 128: " << length
                      << COLOR::WHITE << std::endl;
        }
    }

    int sync_register = 0;
    // total_data+=length;
    int misc = 0b0000101000000000;
    Instruction *inst;
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(input_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(input_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 10);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(dest_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(dest_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 11);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(length / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(length / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 12);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
        MiscOperationState set_sync(MISC_SET_SYNC_FLAG, 0, 0, 2, 4);
        inst->SetOperationState(Instruction::MISC, &set_sync);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    if (1)
    {
        Instruction *inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE1, 16385 + sync_register);
        ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 10, 12, 11, 33, misc);
        inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    for (int i = 0; i < 1; i++)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
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

data<4>
AddVector(std::vector<Instruction *> &instruction_list,
          data<4> input1,
          data<4> input2,
          uint32_t output)
{
  Instruction *inst;
  assert(input1.dims[0] == input2.dims[0]);
  assert(input1.dims[1] == input2.dims[1]);
  assert(input1.dims[2] == input2.dims[2]);
  assert(input1.dims[3] == input2.dims[3]);
  data<4> add_output;
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
          HelperGetValue((input2.addr + i * kNumberOfSubcores) / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1,
          HelperGetValue((input2.addr + i * kNumberOfSubcores) / kVMemSeg).first);
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

data<4>
AddVector(std::vector<Instruction *> &instruction_list,
          data<4> input1,
          data<1> input2,
          uint32_t output)
{
  Instruction *inst;
  assert(input1.dims[3] == input2.dims[0]);
  data<4> add_output;
  add_output.dims = input1.dims;
  add_output.addr = output;
  uint32_t move_num;
  uint32_t num = input2.size() / kNumberOfSubcores;

  std::cout << "input1.size: " << input1.size() << std::endl;
  std::cout << "input2.size: " << input2.size() << std::endl;
  std::cout << "num: " << num << std::endl;

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
          HelperGetValue((input2.addr + (i % num) * kNumberOfSubcores) / kVMemSeg).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1,
          HelperGetValue((input2.addr + (i % num) * kNumberOfSubcores) / kVMemSeg).first);
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

data<1>
AddVector(INST_TYPE &inst2,
          data<1> input1,
          data<1> input2,
          float beta,
          float alpha,
          uint32_t output)
{
  Instruction *inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  data<1> add_output;
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
    uint32_t use_one_col = std::min(uint32_t(kNumberOfSubcores), input1.dims[0] - i * kNumberOfSubcores);

    auto input1_reg = inst2.AllocVReg("");
    auto input2_reg = inst2.AllocVReg("");

    Load8_128(inst2, input1_reg, 1, use_one_col, input1.addr + i * kNumberOfSubcores, kNumberOfCores);
    Load8_128(inst2, input2_reg, 1, use_one_col, input2.addr + i * kNumberOfSubcores, kNumberOfCores);

    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(beta).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(beta).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input2_reg.id, 44, input2_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetFloatingBits(alpha).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetFloatingBits(alpha).first);
      VectorOperationState mul(V_F32_MULTIPLICATION, 0, input1_reg.id, 44, input1_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &mul);
      VectorOperationState add(V_F32_ADDITION, 0, input1_reg.id, input2_reg.id, input1_reg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &add);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    Store8_128(inst2, input1_reg, 1, use_one_col, output + i * kNumberOfSubcores, kNumberOfCores);
  }
  return add_output;
}

//当间隔小于128时，不能将原始数据放在 vmem 最开始，最好放在1024及之后的位置
// void Load8_128(Inst2& inst2, VReg& load_reg, uint32_t output_row, uint32_t output_col, uint32_t src_addr, uint32_t intervals) {
//   Instruction* inst;
//   std::vector<Instruction *> &instruction_list = inst2.inst.insts;

//   uint32_t stride = intervals / kNumberOfCores;
//   uint32_t offset = intervals % kNumberOfCores;
//   uint32_t addr_offset = src_addr % kNumberOfCores;
//   uint32_t load_num = 8;
//   uint32_t one_load_row;
//   auto vmask = inst2.AllocVMask();
//   std::vector<VReg> temp_reg_arr;
//   assert(intervals >= kNumberOfCores || src_addr >= kNumberOfSubcores);

//   if(1) {
//     inst = new Instruction();
//     inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(output_col).second);
//     inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(output_col).first);
//     VectorOperationState set_mask(V_SET_VMASK, 0, 0, 44, vmask.id);
//     inst->SetOperationState(Instruction::VECTORONE, &set_mask);
//     VectorOperationState move(V_U32_MOVE, 0, 0, 46, load_reg.id);
//     inst->SetOperationState(Instruction::VECTORTWO, &move);
//     CompleteInstruction(inst);
//     instruction_list.push_back(inst);
//   }

//   if(stride % 2 == 1) {
//     load_num = 1;
//     one_load_row = std::min((uint32_t)kNumberOfSubcoresPerCore, output_row);
//   }
//   else{
//     if(stride % kNumberOfSubcoresPerCore == 2 || stride % kNumberOfSubcoresPerCore == 6) {
//       one_load_row = std::min((uint32_t)4, output_row);
//       load_num = std::min((uint32_t)2, (output_row+3)/4);
//     }
//     else if(stride % kNumberOfSubcoresPerCore == 4) {
//       one_load_row = std::min((uint32_t)2, output_row);
//       load_num = std::min((uint32_t)4, (output_row+1)/2);
//     }
//     else{
//       one_load_row = 1;
//       load_num = std::min((uint32_t)8, output_row/1);
//     }
//   }

//   if(offset == 0 && addr_offset == 0) {
//     auto zero_vreg = inst2.AllocVReg("");
//     auto sreg = inst2.AllocSReg();

//     if(1) {
//       inst = new Instruction();
//       // inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(output_col).second);
//       // inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(output_col).first);
//       // VectorOperationState set_vmask(V_SET_VMASK, 0, 0, 44, vmask.id);
//       // inst->SetOperationState(Instruction::VECTORONE, &set_vmask);
//       VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_vreg.id);
//       inst->SetOperationState(Instruction::VECTORTWO, &move);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }

//     for(uint32_t i = 0; i < load_num; i++) {
//       temp_reg_arr.emplace_back(inst2.AllocVReg(""));
//     }

//     uint32_t _one_load_row = one_load_row;
//     for(uint32_t i = 0; i < output_row; i+=_one_load_row) {
//       uint32_t id_mask = 0;
//       _one_load_row = std::min(one_load_row, output_row - i);
//       for(uint32_t j = 0; j < _one_load_row; j++) {
//         id_mask += 1 << (i+j);
//       }

//       if(1) {
//         inst = new Instruction();
//         inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(src_addr / kVMemSeg).second);
//         inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(src_addr / kVMemSeg).first);
//         inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg.id);
//         ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg.id);
//         inst->SetOperationState(Instruction::SCALARONE, &set_base);
//         inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
//         inst->SetImmediateValue(Instruction::IMMEDIATE2, stride);
//         inst->SetImmediateValue(Instruction::IMMEDIATE5, id_mask);
//         VectorLoadOperationState vload(V_LOAD_WITH_VMASK0 + vmask.id, 0, temp_reg_arr[i/one_load_row].id, 1, 2, 4, 0, 7);
//         inst->SetOperationState(Instruction::VECTORLOAD, &vload);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }

//     _one_load_row = one_load_row;
//     for(uint32_t i = 0; i < output_row; i+=_one_load_row) {
//       _one_load_row = std::min(one_load_row, output_row - i);

//       if(1) {
//         inst = new Instruction();
//         VectorOperationState add(V_F32_ADDITION, 0, load_reg.id, temp_reg_arr[i/one_load_row].id, load_reg.id);
//         inst->SetOperationState(Instruction::VECTORTWO, &add);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }
  
//     if(1) {
//       inst = new Instruction();
//       VectorOperationState select(V_SELECT_VMASK0 + vmask.id, 0, zero_vreg.id, load_reg.id, load_reg.id);
//       inst->SetOperationState(Instruction::VECTORONE, &select);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }
  
//   }
//   else{
//     auto zero_vreg = inst2.AllocVReg("");
//     auto temp_reg = inst2.AllocVReg("");
//     if(1) {
//       inst = new Instruction();
//       inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(offset).second);
//       inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(offset).first);
//       VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_vreg.id);
//       inst->SetOperationState(Instruction::VECTORTWO, &move);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }

//     for(uint32_t i = 0; i < output_row * 2; i++) {
//       temp_reg_arr.push_back(inst2.AllocVReg(""));
//     }

//     uint32_t _src_addr = src_addr;
//     for(uint32_t i = 0; i < output_row; i++, src_addr+=intervals) {
//       uint32_t src_addr_128 = AlignTo128Bytes(src_addr) - i * kNumberOfCores;
//       auto sreg_1 = inst2.AllocSReg();
//       auto sreg_2 = inst2.AllocSReg();
    
//       if(1) {
//         if(1) {
//           inst = new Instruction();
//           inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((src_addr - i * kNumberOfCores) / kVMemSeg).second);
//           inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((src_addr - i * kNumberOfCores)/ kVMemSeg).first);
//           inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg_1.id);
//           ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg_1.id);
//           inst->SetOperationState(Instruction::SCALARONE, &set_base);
//           inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
//           inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
//           inst->SetImmediateValue(Instruction::IMMEDIATE5, 1 << i);
//           VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, temp_reg_arr[2*i].id, 1, 2, 4, 0, 7);
//           inst->SetOperationState(Instruction::VECTORLOAD, &vload);
//           CompleteInstruction(inst);
//           instruction_list.push_back(inst);
//         }

//         if(1) {
//           inst = new Instruction();
//           inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(src_addr_128 / kVMemSeg).second);
//           inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(src_addr_128 / kVMemSeg).first);
//           inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg_2.id);
//           ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg_2.id);
//           inst->SetOperationState(Instruction::SCALARONE, &set_base);
//           inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
//           inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
//           inst->SetImmediateValue(Instruction::IMMEDIATE5, 1 << i);
//           VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, temp_reg_arr[2*i + 1].id, 1, 2, 4, 0, 7);
//           inst->SetOperationState(Instruction::VECTORLOAD, &vload);
//           CompleteInstruction(inst);
//           instruction_list.push_back(inst);
//         }
//       }
//     }

//     for(uint32_t i = 0; i < output_row; i++) {
//       if(1) {
//         inst = new Instruction();
//         inst->SetImmediateValue(Instruction::IMMEDIATE3, -((i * offset + addr_offset) % kNumberOfCores));
//         MTIOperationState rotate(MTI_ROTATE, 0, temp_reg_arr[2*i].id, 5, 0);
//         inst->SetOperationState(Instruction::MTI, &rotate);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }

//       if(1) {
//         inst = new Instruction();
//         inst->SetImmediateValue(Instruction::IMMEDIATE3, -((i * offset + addr_offset) % kNumberOfCores));
//         MTIOperationState rotate(MTI_ROTATE, 0, temp_reg_arr[2*i + 1].id, 5, 0);
//         inst->SetOperationState(Instruction::MTI, &rotate);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }
    
//     for(uint32_t i = 0; i < output_row; i++) {
//       if(1) {
//         inst = new Instruction();
//         MTROperationState trf_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_reg_arr[2*i].id, 0);
//         inst->SetOperationState(Instruction::MTR, &trf_pop);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
      
//       if(1) {
//         inst = new Instruction();
//         MTROperationState trf_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_reg_arr[2*i + 1].id, 0);
//         inst->SetOperationState(Instruction::MTR, &trf_pop);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }

//     for(uint32_t i = 0; i < output_row; i++) {
//       auto select_mask = inst2.AllocVMask();
//       if(1) {
//         inst = new Instruction();
//         inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(kNumberOfCores -((i * offset + addr_offset) % kNumberOfCores)).second);
//         inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(kNumberOfCores -((i * offset + addr_offset) % kNumberOfCores)).first);
//         VectorOperationState set_mask(V_SET_VMASK, 0, 0, 44, select_mask.id);
//         inst->SetOperationState(Instruction::VECTORONE, &set_mask);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }

//       if(1) {
//         inst = new Instruction();
//         VectorOperationState select(V_SELECT_VMASK0 + select_mask.id, 0, temp_reg_arr[2*i + 1].id, temp_reg_arr[2*i].id, temp_reg_arr[2*i].id);
//         inst->SetOperationState(Instruction::VECTORONE, &select);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }

//     for(uint32_t i = 0; i < output_row; i++) {
//       // if(i == 0)
//       //   continue;
//       if(1) {
//         inst = new Instruction();
//         VectorOperationState add(V_F32_ADDITION, 0, temp_reg_arr[2*i].id, load_reg.id, load_reg.id);
//         inst->SetOperationState(Instruction::VECTORTWO, &add);
//         CompleteInstruction(inst);
//         instruction_list.push_back(inst);
//       }
//     }

//     if(1) {
//       inst = new Instruction();
//       VectorOperationState select(V_SELECT_VMASK0 + vmask.id, 0, zero_vreg.id, load_reg.id, load_reg.id);
//       inst->SetOperationState(Instruction::VECTORONE, &select);
//       CompleteInstruction(inst);
//       instruction_list.push_back(inst);
//     }
//   }
// }



//当间隔小于128时，不能将原始数据放在 vmem 最开始，最好放在1024及之后的位置

void Load8_128(Inst2& inst2, VReg& load_reg, uint32_t output_row, uint32_t output_col, uint32_t src_addr, uint32_t intervals) {

  Instruction* inst;

  std::vector<Instruction *> &instruction_list = inst2.inst.insts;



  uint32_t stride = intervals / kNumberOfCores;

  uint32_t offset = intervals % kNumberOfCores;

  uint32_t addr_offset = src_addr % kNumberOfCores;

  uint32_t load_num = 8;

  uint32_t one_load_row;

  auto vmask = inst2.AllocVMask();

  std::vector<VReg> temp_reg_arr;

  assert(intervals >= kNumberOfCores || src_addr >= kNumberOfSubcores);

  // std::cout << "======================================\n";

  // std::cout << "src_addr: \t" << src_addr << std::endl;

  // std::cout << "AlignTo128Bytes(src_addr): \t" << AlignTo128Bytes(src_addr) << std::endl;

  // std::cout << "intervals: \t" << intervals << std::endl;

  // std::cout << "output_row: \t" << output_row << std::endl;

  // std::cout << "output_col: \t" << output_col << std::endl;

  // std::cout << "======================================\n";



  if(1) {

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(output_col).second);

    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(output_col).first);

    VectorOperationState set_mask(V_SET_VMASK, 0, 0, 44, vmask.id);

    inst->SetOperationState(Instruction::VECTORONE, &set_mask);

    VectorOperationState move(V_U32_MOVE, 0, 0, 46, load_reg.id);

    inst->SetOperationState(Instruction::VECTORTWO, &move);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }



  if(stride % 2 == 1) {

    load_num = 1;

    one_load_row = std::min((uint32_t)kNumberOfSubcoresPerCore, output_row);

  }

  else{

    if(stride % kNumberOfSubcoresPerCore == 2 || stride % kNumberOfSubcoresPerCore == 6) {

      one_load_row = std::min((uint32_t)4, output_row);

      load_num = std::min((uint32_t)2, (output_row+3)/4);

    }

    else if(stride % kNumberOfSubcoresPerCore == 4) {

      one_load_row = std::min((uint32_t)2, output_row);

      load_num = std::min((uint32_t)4, (output_row+1)/2);

    }

    else{

      one_load_row = 1;

      load_num = std::min((uint32_t)8, output_row/1);

    }

  }



  if(offset == 0 && addr_offset == 0) {

    auto zero_vreg = inst2.AllocVReg("");

    auto sreg = inst2.AllocSReg();



    if(1) {

      inst = new Instruction();

      // inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(output_col).second);

      // inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(output_col).first);

      // VectorOperationState set_vmask(V_SET_VMASK, 0, 0, 44, vmask.id);

      // inst->SetOperationState(Instruction::VECTORONE, &set_vmask);

      VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_vreg.id);

      inst->SetOperationState(Instruction::VECTORTWO, &move);

      CompleteInstruction(inst);

      instruction_list.push_back(inst);

    }



    for(uint32_t i = 0; i < load_num; i++) {

      temp_reg_arr.emplace_back(inst2.AllocVReg(""));

    }



    uint32_t _one_load_row = one_load_row;

    for(uint32_t i = 0; i < output_row; i+=_one_load_row) {

      uint32_t id_mask = 0;

      _one_load_row = std::min(one_load_row, output_row - i);

      for(uint32_t j = 0; j < _one_load_row; j++) {

        id_mask += 1 << (i+j);

      }

      if (1)

      {

        inst = new Instruction();

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(src_addr / kVMemSeg).second);

        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(src_addr / kVMemSeg).first);

        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg.id);

        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg.id);

        inst->SetOperationState(Instruction::SCALARONE, &set_base);

        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);

        inst->SetImmediateValue(Instruction::IMMEDIATE2, stride);

        inst->SetImmediateValue(Instruction::IMMEDIATE5, id_mask);

        VectorLoadOperationState vload(V_LOAD_WITH_VMASK0 + vmask.id, 0, temp_reg_arr[i/one_load_row].id, 1, 2, 4, 0, 7);

        inst->SetOperationState(Instruction::VECTORLOAD, &vload);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }

    }



    _one_load_row = one_load_row;

    for(uint32_t i = 0; i < output_row; i+=_one_load_row) {

      _one_load_row = std::min(one_load_row, output_row - i);



      if(1) {

        inst = new Instruction();

        VectorOperationState add(V_F32_ADDITION, 0, load_reg.id, temp_reg_arr[i/one_load_row].id, load_reg.id);

        inst->SetOperationState(Instruction::VECTORTWO, &add);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }

    }

  

    if(1) {

      inst = new Instruction();

      VectorOperationState select(V_SELECT_VMASK0 + vmask.id, 0, zero_vreg.id, load_reg.id, load_reg.id);

      inst->SetOperationState(Instruction::VECTORONE, &select);

      CompleteInstruction(inst);

      instruction_list.push_back(inst);

    }

  }

  else{

    auto zero_vreg = inst2.AllocVReg("");

    auto temp_reg = inst2.AllocVReg("");



    if (1)

    {

      inst = new Instruction();

      VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_vreg.id);

      inst->SetOperationState(Instruction::VECTORTWO, &move);

      CompleteInstruction(inst);

      instruction_list.push_back(inst);

    }



    for(uint32_t i = 0; i < output_row * 2; i++) {

      temp_reg_arr.push_back(inst2.AllocVReg(""));

      if (1)

      {

        inst = new Instruction();

        VectorOperationState move(V_U32_MOVE, 0, 0, 46, temp_reg_arr[i].id);

        inst->SetOperationState(Instruction::VECTORTWO, &move);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }



    }



    uint32_t _src_addr = src_addr;

    for(uint32_t i = 0; i < output_row; i++, src_addr+=intervals) {

      uint32_t src_addr_128 = AlignTo128Bytes(src_addr) - i * kNumberOfCores;

      auto sreg_1 = inst2.AllocSReg();

      auto sreg_2 = inst2.AllocSReg();

    

      if(1) {

        if(1) {

          inst = new Instruction();

          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((src_addr - i * kNumberOfCores) / kVMemSeg).second);

          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((src_addr - i * kNumberOfCores) / kVMemSeg).first);

          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg_1.id);

          ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg_1.id);

          inst->SetOperationState(Instruction::SCALARONE, &set_base);

          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);

          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);

          inst->SetImmediateValue(Instruction::IMMEDIATE5, 1 << i);

          VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, temp_reg_arr[2*i].id, 1, 2, 4, 0, 7);

          inst->SetOperationState(Instruction::VECTORLOAD, &vload);

          CompleteInstruction(inst);

          instruction_list.push_back(inst);

        }



        if(1) {

          inst = new Instruction();

          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(src_addr_128 / kVMemSeg).second);

          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(src_addr_128 / kVMemSeg).first);

          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg_2.id);

          ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg_2.id);

          inst->SetOperationState(Instruction::SCALARONE, &set_base);

          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);

          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);

          inst->SetImmediateValue(Instruction::IMMEDIATE5, 1 << i);

          VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, temp_reg_arr[2*i + 1].id, 1, 2, 4, 0, 7);

          inst->SetOperationState(Instruction::VECTORLOAD, &vload);

          CompleteInstruction(inst);

          instruction_list.push_back(inst);

        }

      }

    }



    for(uint32_t i = 0; i < output_row; i++) {

      if(1) {

        inst = new Instruction();

        inst->SetImmediateValue(Instruction::IMMEDIATE3, -((i * offset + addr_offset) % kNumberOfCores));

        MTIOperationState rotate(MTI_ROTATE, 0, temp_reg_arr[2*i].id, 5, 0);

        inst->SetOperationState(Instruction::MTI, &rotate);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }



      if(1) {

        inst = new Instruction();

        inst->SetImmediateValue(Instruction::IMMEDIATE3, -((i * offset + addr_offset) % kNumberOfCores));

        MTIOperationState rotate(MTI_ROTATE, 0, temp_reg_arr[2*i + 1].id, 5, 0);

        inst->SetOperationState(Instruction::MTI, &rotate);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }

    }

    

    for(uint32_t i = 0; i < output_row; i++) {

      if(1) {

        inst = new Instruction();

        MTROperationState trf_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_reg_arr[2*i].id, 0);

        inst->SetOperationState(Instruction::MTR, &trf_pop);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }

      

      if(1) {

        inst = new Instruction();

        MTROperationState trf_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_reg_arr[2*i + 1].id, 0);

        inst->SetOperationState(Instruction::MTR, &trf_pop);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }

    }



    for(uint32_t i = 0; i < output_row; i++) {

      auto select_mask = inst2.AllocVMask();

      if(1) {

        inst = new Instruction();

        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(kNumberOfCores -((i * offset + addr_offset) % kNumberOfCores)).second);

        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(kNumberOfCores -((i * offset + addr_offset) % kNumberOfCores)).first);

        VectorOperationState set_mask(V_SET_VMASK, 0, 0, 44, select_mask.id);

        inst->SetOperationState(Instruction::VECTORONE, &set_mask);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }



      if(1) {

        inst = new Instruction();

        VectorOperationState select(V_SELECT_VMASK0 + select_mask.id, 0, temp_reg_arr[2*i + 1].id, temp_reg_arr[2*i].id, temp_reg_arr[2*i].id);

        inst->SetOperationState(Instruction::VECTORONE, &select);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }

    }



    for(uint32_t i = 0; i < output_row; i++) {

      // if(i == 0)

      //   continue;

      if(1) {

        inst = new Instruction();

        VectorOperationState add(V_F32_ADDITION, 0, temp_reg_arr[2*i].id, load_reg.id, load_reg.id);

        inst->SetOperationState(Instruction::VECTORTWO, &add);

        CompleteInstruction(inst);

        instruction_list.push_back(inst);

      }

    }



    if(1) {

      inst = new Instruction();

      VectorOperationState select(V_SELECT_VMASK0 + vmask.id, 0, zero_vreg.id, load_reg.id, load_reg.id);

      inst->SetOperationState(Instruction::VECTORONE, &select);

      CompleteInstruction(inst);

      instruction_list.push_back(inst);

    }

  }

}

//当间隔小于128时，不能将数据存在 vmem 最开始，最好放在1024及之后的位置
void Store8_128(Inst2& inst2, VReg& store_reg, uint32_t input_row, uint32_t input_col, uint32_t src_addr, uint32_t intervals) {
  Instruction* inst;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;

  assert(input_row <= kNumberOfSubcoresPerCore);
  assert(input_col <= kNumberOfCores);

  uint32_t stride = intervals / kNumberOfCores;
  uint32_t offset = intervals % kNumberOfCores;
  uint32_t addr_offset = src_addr % kNumberOfCores;
  uint32_t store_num = 8;
  uint32_t one_store_row;
  auto vmask = inst2.AllocVMask();
  std::vector<VReg> temp_reg_arr;

  assert(intervals >= kNumberOfCores || src_addr >= kNumberOfSubcores);

  if(stride % 2 == 1) {
    store_num = 1;
    one_store_row = std::min((uint32_t)kNumberOfSubcoresPerCore, input_row);
  }
  else{
    if(stride % kNumberOfSubcoresPerCore == 2 || stride % kNumberOfSubcoresPerCore == 6) {
      one_store_row = std::min((uint32_t)4, input_row);
      store_num = std::min((uint32_t)2, (input_row+3)/4);
    }
    else if(stride % kNumberOfSubcoresPerCore == 4) {
      one_store_row = std::min((uint32_t)2, input_row);
      store_num = std::min((uint32_t)4, (input_row+1)/2);
    }
    else{
      one_store_row = 1;
      store_num = std::min((uint32_t)8, input_row/1);
    }
  }

  if(offset == 0 && addr_offset == 0) {
    auto zero_vreg = inst2.AllocVReg("");

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(input_col).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(input_col).first);
      VectorOperationState set_vmask(V_SET_VMASK, 0, 0, 44, vmask.id);
      inst->SetOperationState(Instruction::VECTORONE, &set_vmask);
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_vreg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    uint32_t _one_store_row = one_store_row;
    for(uint32_t i = 0; i < input_row; i+=_one_store_row) {
      uint32_t id_mask = 0;
      auto sreg = inst2.AllocSReg();
      _one_store_row = std::min(one_store_row, input_row - i);
      for(uint32_t j = 0; j < _one_store_row; j++) {
        id_mask += 1 << (i+j);
      }

      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(src_addr / kVMemSeg).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(src_addr / kVMemSeg).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg.id);
        ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg.id);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, stride);
        inst->SetImmediateValue(Instruction::IMMEDIATE5, id_mask);
        VectorStoreOperationState vstore(V_STORE_WITH_VMASK0 + vmask.id, 0, store_reg.id, 1, 2, 4, 0, 7);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
  
  }
  else{
    auto left_mask = inst2.AllocVMask();
    auto right_mask = inst2.AllocVMask();
    auto zero_vreg = inst2.AllocVReg("");
    auto temp_reg = inst2.AllocVReg("");
    auto core_id_reg = inst2.AllocVReg("");

    if(1) {
      inst = new Instruction();
      VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, core_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, zero_vreg.id);
      inst->SetOperationState(Instruction::VECTORTWO, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }

    if(1) {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(kNumberOfCores - 1).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(kNumberOfCores - 1).first);
      VectorOperationState _and(V_U32_AND, 0, core_id_reg.id, 44, core_id_reg.id);
      inst->SetOperationState(Instruction::VECTORONE, &_and);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    } 

    for(uint32_t i = 0; i < input_row; i++ ) {
        temp_reg_arr.emplace_back(inst2.AllocVReg(""));
    }

    for(uint32_t i = 0; i < input_row; i++) {
      if(1) {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE3, i * offset + addr_offset);
        MTIOperationState rotate(MTI_ROTATE, 0, store_reg.id, 5, 0);
        inst->SetOperationState(Instruction::MTI, &rotate);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
    
    for(uint32_t i = 0; i < input_row; i++) {
      if(1) {
        inst = new Instruction();
        MTROperationState trf_pop(MTR_READ_TRANSPOSE_RESULT, 0, temp_reg_arr[i].id, 0);
        inst->SetOperationState(Instruction::MTR, &trf_pop);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }

    for(uint32_t i = 0; i < input_row; i++, src_addr+=intervals) {
      uint32_t src_addr_128 = AlignTo128Bytes(src_addr) - i * kNumberOfCores;
      uint32_t left_offset = (i * offset + addr_offset)%kNumberOfCores;

      auto sreg_1 = inst2.AllocSReg();
      auto sreg_2 = inst2.AllocSReg();      

      if(left_offset + input_col >= kNumberOfCores) {
        if(1) {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(left_offset).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(left_offset).first);
          VectorOperationState greater_eq(V_S32_GREATEREQUAL, 0, core_id_reg.id, 44, left_mask.id);
          inst->SetOperationState(Instruction::VECTORONE, &greater_eq);

          inst->SetImmediateValue(Instruction::IMMEDIATE2, HelperGetValue(left_offset + input_col - kNumberOfCores).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE3, HelperGetValue(left_offset + input_col - kNumberOfCores).first);
          VectorOperationState less(V_S32_LESSER, 0, core_id_reg.id, 45, right_mask.id);
          inst->SetOperationState(Instruction::VECTORTWO, &less);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        } 
      }
      else if(left_offset + input_col < kNumberOfCores){
        auto temp_reg = inst2.AllocVReg("");

        if(1) {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(left_offset + input_col).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(left_offset + input_col).first);
          VectorOperationState set_vmask(V_SET_VMASK, 0, core_id_reg.id, 44, left_mask.id);
          inst->SetOperationState(Instruction::VECTORONE, &set_vmask);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          VectorOperationState select_vmask(V_SELECT_VMASK0+left_mask.id, 0, zero_vreg.id, core_id_reg.id, temp_reg.id);
          inst->SetOperationState(Instruction::VECTORONE, &select_vmask);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(left_offset).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(left_offset).first);
          VectorOperationState greater_eq(V_S32_GREATEREQUAL, 0, temp_reg.id, 44, left_mask.id);
          inst->SetOperationState(Instruction::VECTORONE, &greater_eq);
          VectorOperationState set_vmask(V_SET_VMASK, 0, temp_reg.id, 46, right_mask.id);
          inst->SetOperationState(Instruction::VECTORTWO, &set_vmask);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }

      if(1) {
        if(1) {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((src_addr - i * kNumberOfCores) / kVMemSeg).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((src_addr - i * kNumberOfCores) / kVMemSeg).first);
          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg_1.id);
          ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg_1.id);
          inst->SetOperationState(Instruction::SCALARONE, &set_base);
          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
          inst->SetImmediateValue(Instruction::IMMEDIATE5, 1 << i);
          VectorStoreOperationState vstore(V_STORE_WITH_VMASK0+left_mask.id, 0, temp_reg_arr[i].id, 1, 2, 4, 0, 7);
          inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }

        if(1) {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(src_addr_128 / kVMemSeg).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(src_addr_128 / kVMemSeg).first);
          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, sreg_2.id);
          ScalarOperationState set_base(S_U32_MOVE, 0, 1, 44, sreg_2.id);
          inst->SetOperationState(Instruction::SCALARONE, &set_base);
          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
          inst->SetImmediateValue(Instruction::IMMEDIATE5, 1 << i);
          VectorStoreOperationState vstore(V_STORE_WITH_VMASK0+right_mask.id, 0, temp_reg_arr[i].id, 1, 2, 4, 0, 7);
          inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
    }
  }
}
