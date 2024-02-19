#pragma once
#ifndef __UTIILS__
  #define __UTIILS__

#    include "../GPT2Define.h"
#    include "../Data.h"
#    include <vector>
uint32_t AlignTo128Bytes(uint32_t _size);

void AddNoop(unsigned int number, std::vector<Instruction *> &bundle);

void CompleteInstruction(Instruction *instruction);

void HBM_TO_VMEM(std::vector<Instruction *> &instruction_list,
                            uint32_t src_addr,
                            uint32_t dest_addr,
                            uint32_t len);
void HBM_TO_VMEM_Stride(std::vector<Instruction *> &instruction_list,
                    uint32_t input_addr,
                    uint32_t dest_addr,
                    uint32_t common,
                    uint32_t hbmstride,
                    uint32_t vmemstride);
void VMEM_TO_HBM(std::vector<Instruction *> &instruction_list,
                            uint32_t src_addr,
                            uint32_t dest_addr,
                            uint32_t len);
data<4> AddVector(std::vector<Instruction *> &instruction_list,
                  data<4> input1,
                  data<4> input2,
                  uint32_t output);

data<1>
AddVector(INST_TYPE &inst2,
          data<1> input1,
          data<1> input2,
          float beta,
          float alpha,
          uint32_t output);

data<4> AddVector(std::vector<Instruction *> &instruction_list,
                  data<4> input1,
                  data<1> input2,
                  uint32_t output);

void Load8_128(Inst2& inst2, VReg& load_reg, uint32_t output_row, uint32_t output_col, uint32_t src_addr, uint32_t intervals);

void Store8_128(Inst2& inst2, VReg& store_reg, uint32_t input_row, uint32_t input_col, uint32_t src_addr, uint32_t intervals);

void Load8_128_V2(Inst2& inst2, std::vector<VReg>& load_reg_arr, std::vector<uint32_t> output_row_arr, std::vector<uint32_t> output_col_arr, std::vector<uint32_t> src_addr_arr, std::vector<uint32_t> intervals_arr);
#endif