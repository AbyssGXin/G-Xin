#include "xys/Xys.h"
#include "fxc/Fxc.h"
#include "hbm/Hbm.h"
#include <stdio.h>
#include  <iostream>
#include <bitset>
// #include "simple_test/test_helper.h"
#include "../device/Devices/Device_Simulator.h"
#include "dma/DmaEngine.h"
#include <fstream>
#include <string>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <map>
#include "BLOOM/bloom/bloom.h"

# define BLOOMMLP_test 0
# define BLOOMBlock_test 0
# define BLOOMModel_test 0
# define linear_test 0
# define build_alibi_test 0
# define _prepare_attn_mask_test 0
# define BLOOMMLPBackward_test 0
# define BLOOMBlockBackward_test 0
# define BLOOMModelBackward_test 0
# define BLOOMForCausalLMBackward_test 0
# define BLOOMAttentionBackward_test 0
# define matmul_test 0
# define MatMulDxBackward_test 0
# define Training_test 0
# define Scheduler_test 0
# define tt_test 0
# define tt7b_test 0
# define dropout_test 0
# define droptest 0
# define loss_test 1
#define PI 3.14159265358979323846

typedef std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>  vec6d_t_i;
typedef std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>  vec5d_t_i;
typedef std::vector<std::vector<std::vector<std::vector<int>>>>  vec4d_t_i;
typedef std::vector<std::vector<std::vector<int>>>  vec3d_t_i;
typedef std::vector<std::vector<int>>  vec2d_t_i;
typedef std::vector<int>  vec1d_t_i;

std::string Train_7B_Back = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_backward_f32/";
std::string Train_7B_For = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/";


data<3> dropouttest2(INST_TYPE inst2, data<3> input, float p, uint32_t output_addr) {

  std::vector<Instruction*> &instruction_list = inst2.inst.insts;

  Instruction* inst;



  data<3> output(output_addr, {input.dims[0], input.dims[1], input.dims[2]});



  auto p_reg = inst2.AllocVReg("");

  auto core_id_reg = inst2.AllocVReg("");

  inst2(VCoreId, core_id_reg.id);

  inst2(VMov, inst2.inst.ImmeF(p), p_reg.id);



  //丢弃概率为 0 时，只需要将输入搬运至输出地址

  if(p == 0 && input.addr != output_addr) {

    Memcopy(input.asVMem(inst2), output.asVMem(inst2));

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


void load_input(float *input, std::string path, int num) {
  std::ifstream in(path);
  std::cout << "load: " << path << std::endl;
  assert(in);
  std::string line;
  for (int i = 0; i < num; i++)
  {
    std::getline(in, line);
    assert(line.length() > 0);
    auto temp = atof(line.c_str());
    // uint32_t t = stoul(line, 0, 16);
    // float x = *(float *)(&t);
    // std::cout << i << ": " << temp << std::endl;
    input[i] = temp;
  }
  // std::getline(in, line);
  // assert(line.length() <= 0);
  in.close();

}

void load_input_mask(float *input, std::string path, int num) {
  std::ifstream in(path);
  std::cout << "load: " << path << std::endl;
  assert(in);
  std::string line;
  for (int i = 0; i < num; i++)
  {
    std::getline(in, line);
    assert(line.length() > 0);
    bool value = (line == "True");
    float temp;
    if (value) temp = 1.0;
    else temp = 0.0;
    // auto temp = atof(line.c_str());

    // uint32_t t = stoul(line, 0, 16);
    // float x = *(float *)(&t);
    // std::cout << i << ": " << temp << std::endl;
    input[i] = temp;
  }
  std::getline(in, line);
  // assert(line.length() <= 0);
  in.close();

}

uint32_t read_file(std::vector<uint32_t>& data, std::vector<std::string> name,  std::string base_path, std::map<std::string, uint32_t>& weights_addr, uint32_t addr_offset, std::string offset_path = "") {
  uint32_t addr = 0 + addr_offset;
  for(auto _name : name) {
    uint32_t cont = 0;
    std::ifstream in(base_path + _name + offset_path + ".txt");
    std::cout << "load: " << _name + offset_path + ".txt   weight_addr: " << addr ;
    assert(in);
    std::string line;
    weights_addr[_name + offset_path] = addr;
    while(!in.eof()) {
      std::getline(in, line);
      if(line.length() == 0) break;
      float temp = atof(line.c_str());
      uint32_t u32 = *(uint32_t*)(&(temp));
      // std::cout << "line: " << line << std::endl;
      // uint32_t t = stoul(line, NULL, 16);
      // float x = *(float *)(&t);

      if(line.length() != 0) {
        addr++;
        cont++;
        data.push_back(u32);
      }
    }
    
    std::cout << " size: " << cont << std::endl;

    if(addr % 128 != 0) {
      uint32_t padding = (addr/128 + 1) * 128 - addr;
      addr = (addr/128 + 1) * 128;
      for(uint32_t i = 0; i < padding; i++) {
        data.push_back(0);
      }
      std::cout << "erro_addr: " << addr << std::endl;
      // exit(0);
    }
  }

  return addr;
}

// void read(uint32_t* data, std::string str, int length, uint32_t& index)
// {
//   std::ifstream in(str);
//   std::cout << "load: " << str << std::endl;
//   assert(in);
//   std::string line;
//   for (int i = 0; i < length; i++)
//   {
//     std::getline(in, line);
//     if (line.length() == 0) break;
//     auto temp = atof(line.c_str());
//     data[index++] = (uint32_t)(temp);
//     if (i == 0) std::cout << "temp: " << temp << "  read: " << *(float*)(&data[index - 1]) << std::endl;
//   }
//   std::getline(in, line);
//   in.close();
//   return ;
// }

void _prepare_attn_maskTest()
{
  std::cout << "_prepare_attn_maskTest" << std::endl;
  Inst2 inst2;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 1 * 41);
  data<2> attention_mask(0, {1, 41});
  BLOOMConfig config;

  HBMAddr() = 0;
  std::tuple<uint32_t, uint32_t> input_shape(1, 41);
  uint32_t past_key_values_length = 0;
  _prepare_attn_mask(inst2, attention_mask, input_shape, past_key_values_length, 128);

  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[1 * 41];
  float* input = new float[1 * 41];
  uint32_t index = 0;

  load_input(input, "simple_test/BLOOM/test_data/bloom_560m/Model_attention_mask.txt", 1 * 41);
  for(uint32_t i = 0; i < 1 * 41; i++, index++) {
    data[index] = *(uint32_t*)(&(input[i]));
  }

  simulator.WriteToHBM(data, 1 * 41, 0);

  std::cout << "old instruct.size: " << instruction_list.size() << std::endl;

  // instruction_list = schedule(instruction_list);

  // std::cout << "new instruct.size: " << instruction_list.size() << std::endl;
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_index: " << i << " inst_vec_size: " << inst_vec.size() << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }
  
  std::cout << "Execute end\n";
  // simulator.PrintHBM(0, 13 * 50304);
  // simulator.DebugPrintVmem(0, 13*50304);
  // std::cout << "result.dims: " << result.dims.size() << std::endl;
  // for(uint32_t i = 0; i <  result.dims.size(); i++) {
  //   std::cout << "result.dim[" << i << "]: " << result.dims[i] << std::endl;
  // }
  // float* test = new float[41*4096*4];
  // load_input(test, "simple_test/BLOOM/test_data/bloom_7b/MLP_2_hidden_states.txt", 41 * 4096*4);

  // simulator.DebugPrintVmem(result.addr, result.addr+result.size());
  simulator.DebugPrintVmem(128, 6 * 1024 + 128);
  
  return;
}

void build_alibiTest()
{
  std::cout << "build_alibiTest\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 128);
  data<2> hidden_states(0, {1, 41});
  BLOOMConfig config;

  HBMAddr() = 0;
  build_alibi(inst2,  hidden_states, config.n_head, 128);

  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[1 * 41];
  float* input = new float[1 * 41];
  uint32_t index = 0;

  load_input(input, "simple_test/bloom_560m/transformer.attention_mask.txt", 1 * 41);
  for(uint32_t i = 0; i < 1 * 41; i++, index++) {
    data[index] = *(uint32_t*)(&(input[i]));
  }

  simulator.WriteToHBM(data, 1 * 41, 0);

  std::cout << "old instruct.size: " << instruction_list.size() << std::endl;

  // instruction_list = schedule(instruction_list);

  // std::cout << "new instruct.size: " << instruction_list.size() << std::endl;
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_index: " << i << " inst_vec_size: " << inst_vec.size() << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }
  
  std::cout << "Execute end\n";
  // simulator.PrintHBM(0, 13 * 50304);
  // simulator.DebugPrintVmem(0, 13*50304);
  // std::cout << "result.dims: " << result.dims.size() << std::endl;
  // for(uint32_t i = 0; i <  result.dims.size(); i++) {
  //   std::cout << "result.dim[" << i << "]: " << result.dims[i] << std::endl;
  // }
  // float* test = new float[41*4096*4];
  // load_input(test, "simple_test/BLOOM/test_data/bloom_7b/MLP_2_hidden_states.txt", 41 * 4096*4);

  // simulator.DebugPrintVmem(result.addr, result.addr+result.size());
  simulator.DebugPrintVmem(128, 2048 + 128);
  
  return;

}

void matmulTest()
{
  std::cout << "matmulTest\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 41 * 1024 * 3);
  data<3> hidden_states(0, {1, 41, 1024 * 3});
  data<2> weight_addr;   
  weight_addr.hbmaddr = 0 + hidden_states.size();
  weight_addr.dims = {3 * 1024, 1024};
  BLOOMConfig config;

  std::cout << "func 1\n";
  HBMAddr() = 0;
  data<3> result = matmul(inst2, hidden_states.as<4>(), weight_addr, hidden_states.addr + hidden_states.size())[0];
  std::cout << "func 2\n";

  instruction_list = schedule(instruction_list, true);
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[41 * 1024 * 3 + 3072 * 1024];
  float* input = new float[41 * 1024 * 3];
  float* weights = new float[1024 * 3072];
  uint32_t index = 0;

  load_input(input, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23.self_attention.query_key_value_in.txt", 41*1024 * 3);
  load_input(weights, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.query_key_value.weight.txt", 1024 * 1024 * 3);

  for(uint32_t i = 0; i < 41*1024*3; i++, index++) {
    data[index] = *(uint32_t*)(&(input[i]));
  }

  for(uint32_t i = 0; i < 1024 * 1024 * 3; i++) {
    data[index] = *(uint32_t*)(&(weights[i]));
    index++;
  }

  simulator.WriteToHBM(data, 41 * 1024 * 3 + 1024 * 1024 * 3, 0);

  std::cout << "old instruct.size: " << instruction_list.size() << std::endl;

  // instruction_list = schedule(instruction_list);

  // std::cout << "new instruct.size: " << instruction_list.size() << std::endl;
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_index: " << i << " inst_vec_size: " << inst_vec.size() << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }
  
  std::cout << "Execute end\n";
  // std::cout << "new cycle: " <<   cyclenum << std::endl;
  // simulator.PrintHBM(0, 13 * 50304);
  simulator.DebugPrintVmem(result.addr, result.addr+result.size());
  // std::cout << "result.dims: " << result.dims.size() << std::endl;
  // for(uint32_t i = 0; i <  result.dims.size(); i++) {
  //   std::cout << "result.dim[" << i << "]: " << result.dims[i] << std::endl;
  // }
  // float* test = new float[41*1024];
  // load_input(test, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23.self_attention.query_key_value_out.txt", 41 * 1024);

  // simulator.DebugPrintVmem_tensor(result.addr, result.addr+result.size(), test);
  
  return;
}

void linearTest() {
  std::cout << "linearTest\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 41 * 1024);
  data<3> hidden_states(0, {1, 41, 1024});
  data<2> weight_addr;   
  weight_addr.hbmaddr = hidden_states.size();
  weight_addr.dims = {4096, 1024};
  data<1> bias_addr;
  bias_addr.hbmaddr = weight_addr.hbmaddr + weight_addr.size();
  bias_addr.dims = {4096};
  GPT2Config config;
  int output_addr = hidden_states.addr + hidden_states.size();

  std::cout << "func 1\n";
  HBMAddr() = 0;

  uint32_t scalar_local_time_reg = 31;

  if (1) {

    inst = new Instruction();

    ScalarOperationState read_time(S_READ, 0, 0, 0, scalar_local_time_reg);

    inst->SetOperationState(Instruction::SCALARONE, &read_time);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }


  data<3> result = linear(inst2, hidden_states, weight_addr, bias_addr, output_addr);
  std::cout << "func 2\n";

  uint32_t scalar_local_time_reg1 = 30;

  if (1) {

    inst = new Instruction();

    ScalarOperationState read_time(S_READ, 0, 0, 0, scalar_local_time_reg1);

    inst->SetOperationState(Instruction::SCALARONE, &read_time);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  if(1) {

    inst = new Instruction();

    ScalarOperationState set_base1(S_U32_MOVE, 0, 0, 46, 28);

    inst->SetOperationState(Instruction::SCALARONE, &set_base1);

    ScalarOperationState store(S_SMEM_STORE, 0, 31, 28, 0);

    inst->SetOperationState(Instruction::SCALARTWO, &store);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  if(1) {

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE0, 4);

    ScalarOperationState set_base1(S_U32_MOVE, 0, 0, 32, 28);

    inst->SetOperationState(Instruction::SCALARONE, &set_base1);

    ScalarOperationState store(S_SMEM_STORE, 0, 30, 28, 0);

    inst->SetOperationState(Instruction::SCALARTWO, &store);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[41 * 1024 + 1024 * 4096 + 4096];
  float* input = new float[41 * 1024];
  float* weights = new float[1024 * 4096];
  float* bias = new float[4096];
  uint32_t index = 0;

  load_input(input, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.0.mlp_in.txt", 41*1024);
  load_input(weights, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.0.mlp.dense_h_to_4h.weight.txt", 1024 * 4096);
  load_input(bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.0.mlp.dense_h_to_4h.bias.txt", 4096);

  for(uint32_t i = 0; i < 41*1024; i++, index++) {
    data[index] = *(uint32_t*)(&(input[i]));
  }

  for(uint32_t i = 0; i < 1024 * 4096; i++) {
    data[index] = *(uint32_t*)(&(weights[i]));
    index++;
  }

  for(uint32_t i = 0; i < 4096; i++, index++) {
    data[index] = *(uint32_t*)(&(bias[i]));
  }

  simulator.WriteToHBM(data, 41*1024 + 1024 * 4096 + 4096, 0);

  std::cout << "old instruct.size: " << instruction_list.size() << std::endl;

  // instruction_list = schedule(instruction_list);

  // std::cout << "new instruct.size: " << instruction_list.size() << std::endl;
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_index: " << i << " inst_vec_size: " << inst_vec.size() << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }
  
  std::cout << "Execute end\n";

  // simulator.PrintHBM(0, 13 * 50304);
  // simulator.DebugPrintVmem(0, 13*50304);
  // std::cout << "result.dims: " << result.dims.size() << std::endl;
  // for(uint32_t i = 0; i <  result.dims.size(); i++) {
  //   std::cout << "result.dim[" << i << "]: " << result.dims[i] << std::endl;
  // }
  // float* test = new float[41*4096*4];
  // load_input(test, "simple_test/BLOOM/test_data/bloom_7b/MLP_2_hidden_states.txt", 41 * 4096*4);

  // simulator.DebugPrintVmem_tensor(result.addr, result.addr+result.size(), test);
  simulator.DebugPrintSmem(0, 1024);
  return;
}

void BLOOMMLPTest() {
  std::cout << "BLOOMMLP\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 115 * 1024 * 2);
  data<4> hidden_states(0, {1, 1, 115, 1024});
  data<4> residual(hidden_states.addr + hidden_states.size(), {1, 1, 115, 1024});
  data<2> _fc_weight(residual.addr + residual.size(), {1024*4, 1024});
  data<1> _fc_bias(_fc_weight.addr + _fc_weight.size(), {1024*4});
  data<2> _proj_weight(_fc_bias.addr + _fc_bias.size(), {1024, 1024*4});
  data<1> _proj_bias(_proj_weight.addr + _proj_weight.size(), {1024});
  BLOOMConfig config;

  uint32_t weightaddr = _proj_bias.addr + _proj_bias.size();
  HBMAddr() = 0;
  std::cout << "weights_addr: " << _proj_weight.addr << std::endl;
  std::cout << "bias_addr: " << _proj_bias.addr << std::endl;   
  std::map<std::string, uint32_t> weight_map{{"transformer.h.0.mlp.dense_h_to_4h.weight", _fc_weight.addr}, {"transformer.h.0.mlp.dense_h_to_4h.bias", _fc_bias.addr}, 
                                            {"transformer.h.0.mlp.dense_4h_to_h.weight", _proj_weight.addr}, {"transformer.h.0.mlp.dense_4h_to_h.bias", _proj_bias.addr}};
  std::map<std::string, uint32_t> forward_map{{"transformer.h.0.mlp.dense_h_to_4h_in", weightaddr}};
  weightaddr += 115 * 1024;
  forward_map.insert(std::make_pair("transformer.h.0.mlp.gelu_impl_in", weightaddr));
  weightaddr += 115 * 4096;
  forward_map.insert(std::make_pair("transformer.h.0.mlp.dense_4h_to_h_in", weightaddr));

  // config.training = true;
  data<4> result = BLOOMMLP(inst2, config, hidden_states, residual, residual.addr+residual.size(), residual.addr+residual.size(), "transformer.h.0.mlp", weight_map, forward_map);
  // data<4> result = Conv1D(instruction_list, hidden_states, _fc_weight.addr, {768, 3072}, _fc_bias.addr, {3072}, hidden_states.addr + hidden_states.size());
  
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[115 * 1024 * 2 + 1024*1024*4*2 + 1024*4 + 1024];
  float* input1 = new float[115*1024];
  float* input2 = new float[115*1024];
  float* fc_weight = new float[1024*1024*4];
  float* fc_bias = new float[1024*4];
  float* proj_weight = new float[1024*1024*4];
  float* proj_bias = new float[1024];  
  uint32_t index = 0;

  load_input(input1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.h.0.mlp_in.txt", 115*1024);
  load_input(input2, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.h.0.mlp_in.txt", 115*1024);
  load_input(fc_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.0.mlp.dense_h_to_4h.weight.txt", 1024*1024 * 4);
  load_input(fc_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.0.mlp.dense_h_to_4h.bias.txt", 1024 * 4);
  load_input(proj_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.0.mlp.dense_4h_to_h.weight.txt", 1024*1024 * 4);
  load_input(proj_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.0.mlp.dense_4h_to_h.bias.txt", 1024);

  for(uint32_t i = 0; i < 115*1024; i++,index++) {
    data[index] = *(uint32_t*)(&(input1[i]));
  }

  for(uint32_t i = 0; i < 115*1024; i++,index++) {
    data[index] = *(uint32_t*)(&(input2[i]));
  }

  for(uint32_t i = 0; i <1024*1024*4; i++,index++) {
    data[index] = *(uint32_t*)(&(fc_weight[i]));
  }

  for(uint32_t i = 0; i <1024*4; i++,index++) {
    data[index] = *(uint32_t*)(&(fc_bias[i]));
  }

  for(uint32_t i = 0; i <1024*1024*4; i++,index++) {
    data[index] = *(uint32_t*)(&(proj_weight[i]));
  }

  for(uint32_t i = 0; i <1024; i++,index++) {
    data[index] = *(uint32_t*)(&(proj_bias[i]));
  }
  simulator.WriteToHBM(data, 115 * 1024 * 2 + 1024*1024*4*2 + 1024 + 1024*4, 0);

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
    auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);
     

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
    // for (auto it = range.first; it != range.second; it++)
    // {
    //   float* test = new float[it->second.len];
    //   std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
    //   load_input(test, it->second.compare_file, *(int*)(&it->second.len));
    //   simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
    //   // if ((it->second.name == "transformer.h.0.mlp_in") || (it->second.name == "transformer.h.0.mlp_out")
    //   //  || (it->second.name == "transformer.h.0.self_attention_in") || (it->second.name == "transformer.h.0.self_attention_out")
    //   //  || (it->second.name == "transformer.h.0_out"))
    //   // {
    //   //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
    //   // }
    //   // if(it->second.name == "transformer.h.23.self_attention.bloombmm_out")
    //   // {
    //   //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
    //   // }
    // }
    //   // }
  }

  simulator.DebugPrintVmem_Write(result.addr, result.addr + result.size(), "mlp_nodrop");
  std::cout << "Execute end\n";
  // simulator.PrintHBM(addr_map["lm_head_weight"], addr_map["lm_head_weight"] + 32128*512);
  // simulator.DebugPrintVmem(0, 13*3072);

  // float* test = new float[41*1024 * 4];
  // load_input(test, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/测试组/bloom_560m_forward_f32/transformer.h.0.mlp.dense_h_to_4h_out.txt", 41 * 1024 * 4);
  // // simulator.DebugPrintVmem(result.addr, result.addr+result.size());
  // simulator.DebugPrintVmem_tensor(result.addr, result.addr+result.size(), test);
  // // simulator.DebugPrintVmem_Write(result.addr, result.addr+result.size(), "simu");
  return;
}

void BLOOMBlockTest()
{
  std::cout << "BLOOMBlock\n";
  Inst2 inst2;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 44544);
  std::string path = "simple_test/bloom_560m/";
  std::vector<std::string> name{"transformer.h.0.input_layernorm.bias", "transformer.h.0.input_layernorm.weight", \
                                "transformer.h.0.mlp.dense_4h_to_h.bias", "transformer.h.0.mlp.dense_4h_to_h.weight", \
                                "transformer.h.0.mlp.dense_h_to_4h.bias", "transformer.h.0.mlp.dense_h_to_4h.weight", \
                                "transformer.h.0.post_attention_layernorm.bias", "transformer.h.0.post_attention_layernorm.weight", \
                                "transformer.h.0.self_attention.dense.bias", "transformer.h.0.self_attention.dense.weight", \
                                "transformer.h.0.self_attention.query_key_value.bias", "transformer.h.0.self_attention.query_key_value.weight"};
                                // "transformer.h.0.attention_output"};
  std::map<std::string, uint32_t> weights_addr;
  std::vector<uint32_t> _data;

  read_file(_data, name, path, weights_addr, 44544);

  data<4> hidden_states(0, {1, 1, 41, 1024});
  data<4> alibi(hidden_states.size() + hidden_states.addr, {1, 16, 1, 41});
  uint32_t sim_addr = (alibi.addr + alibi.size() + 127) / 128 * 128;
  data<4> attention_mask(sim_addr, {1, 1, 41, 41});
  uint32_t sim_addr_2 = (attention_mask.addr + attention_mask.size() + 127) / 128 * 128; 
  std::cout << "sim_addr:  " << sim_addr_2 << std::endl;
  BLOOMConfig config;

  HBMAddr() = 0;
  ShowFuncCallInfo() = 1;

  // std::tuple<data<4>, std::vector<data<4>>> result = BLOOMBlock(inst2, config, hidden_states, attention_mask, "transformer.h.0", weights_addr, 0, hidden_states.size());
  // data<3> result = BLOOMBlock(inst2, config, hidden_states, alibi, attention_mask, "transformer.h.0", weights_addr, sim_addr_2);
  data<3> result;
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[44544];
  float* input1 = new float[41 * 1024];
  float* alibitest = new float[16 * 41];
  float*  attentionmask = new float[41 * 41];
  uint32_t index = 0;

  load_input(input1, "simple_test/bloom_560m_forward_f16/transformer.h.0_in.txt", 41 * 1024);
  load_input(alibitest, "simple_test/bloom_560m_forward_f16/transformer.alibi.txt", 16 * 41);
  load_input_mask(attentionmask, "simple_test/bloom_560m_forward_f16/transformer.attention_mask.txt", 41 * 41);
  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input1[i]));
  }
  for (uint32_t i = 0; i < 41 *  16; i++, index++)
  {
    data[index] = *(uint32_t*)(&(alibitest[i])); 
  }
  index  = sim_addr;
  for (uint32_t i = 0; i  < 41 * 41; i++, index++)
  {
    data[index] = *(uint32_t*)(&(attentionmask[i]));
  }
  simulator.WriteToHBM(data, index, 0);
  index = sim_addr_2;
  data = new uint32_t[_data.size()];
  // memcpy(data, &_data[0], _data.size()*sizeof(uint32_t));
  for (uint32_t i = 0; i < _data.size(); i++)
  {
    data[i] = _data[i];
  }
  std::cout << "_data.size: " << _data.size() << std::endl;
  simulator.WriteToHBM(data, _data.size(), index);

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);

  for (uint32_t i = 1; i <= inst_vec.size(); i++)
  {
    std::cout << "inst_vec: " << i << std::endl;
    int leng = int(((i * 1.0) / inst_vec.size()) * 50);
    std::cout << (i * 1.0 / inst_vec.size()) * 100 << "%\n";
    std::cout << std::string(50+2, '-') << std::endl;
    std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
    std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }
  
  std::cout << "Execute end\n";
  // simulator.PrintHBM(weights_addr.at("weights_transformer_h_0_ln_1_bias"), weights_addr.at("weights_transformer_h_0_ln_1_bias") + 768);
  // simulator.PrintHBM(weights_addr.at("weights_transformer_h_0_attn_c_proj_weight"), weights_addr.at("weights_transformer_h_0_attn_masked_bias"));
  // simulator.DebugPrintVmem(0, 13*3072);
  // std::cout << "result.dims: " << result.dims.size() << std::endl;
  // for(uint32_t i = 0; i <  result.dims.size(); i++) {
  //   std::cout << "result.dim[" << i << "]: " << result.dims[i] << std::endl;
  // }
  // simulator.DebugPrintVmem(std::get<0>(result).addr, std::get<0>(result).addr+std::get<0>(result).size());

  float* test = new float[41 * 1024 * 3];
  load_input(test, "simple_test/bloom_560m_forward_f16/transformer.h.0.self_attention.query_key_value_out.txt", 41 * 1024 * 3);
  // float *test = new float[1024];
  // load_input(test, "simple_test/bloom_560m/transformer.h.0.input_layernorm.bias.txt", 1024);
  // load_input(test, "simple_test/BLOOM/bloom_560m/attention_output.txt", 41 * 4096);

  simulator.DebugPrintVmem_tensor(result.addr, result.addr + result.size(), test);
  // simulator.DebugPrintVmem(382464, 382464 + 10240);
  // simulator.PrintHBM(41 * 1024, 41 * 1024 + 1024);
  return;
}

void BLOOMModelTest() 
{
  std::cout << "BLOOMModel\n";
  Inst2 inst2;
  std::vector<Instruction *> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 1024);
  // HBM_TO_VMEM(instruction_list, 0, 0, 1024 *41);
  // std::string path = "simple_test/BLOOM/bloom_7b/";
  data<2> input_ids(0, {1, 41});
  // data<2> input_ids(0, {41, 1024});
  data<2> attention_mask;
  // data<2> embeddings_weight(input_ids.addr+input_ids.size(), {250880, 1024});
  data<2> embeddings_weight(1024, {250880, 1024});
  data<1> word_embeddings_weight(embeddings_weight.addr + embeddings_weight.size(), {1024});
  data<1> word_embeddings_bias(word_embeddings_weight.addr + word_embeddings_weight.size(), {1024});
  data<1> ln_f_weight(word_embeddings_bias.size() + word_embeddings_bias.addr, {1024});
  data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {1024});
  


  // std::vector<std::string> name{"transformer.word_embeddings.weight"};
  std::map<std::string, uint32_t> weights_addr{{"transformer.word_embeddings.weight", embeddings_weight.addr},
                                              {"transformer.word_embeddings_layernorm.weight", word_embeddings_weight.addr},
                                              {"transformer.word_embeddings_layernorm.bias", word_embeddings_bias.addr},
                                              {"transformer.ln_f.weight", ln_f_weight.addr},
                                              {"transformer.ln_f.bias", ln_f_bias.addr}};
  
  uint32_t weightaddr = ln_f_bias.addr + ln_f_bias.size();
  
  for (int i = 0; i <= 23; i++)
  {
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.weight", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.bias", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight", weightaddr));
    weightaddr += 3 * 1024 * 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias", weightaddr));
    weightaddr += 3 * 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.weight", weightaddr));
    weightaddr += 1024 * 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.bias", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight", weightaddr));
    weightaddr += 1024 * 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias", weightaddr));
    weightaddr += 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight", weightaddr));
    weightaddr += 1024 * 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias", weightaddr));
    weightaddr += 1024;
  }
  

  BLOOMConfig config;

  HBMAddr() = 0;

  // data<41> result;
  data<3> result;
  // data<3> result = BLOOMModel(inst2, config, input_ids, attention_mask, "transformer", weights_addr, 1024);

  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[weightaddr];
  float* input1 = new float[1 * 41];
  // float* input1 = new float[41*1024];
  float* emb_weight = new float[250880 * 1024];
  float* wel_weight = new float[1024];
  float* wel_bias = new float[1024];
  float* lnf_weight = new float[1024];
  float* lnf_bias = new float[1024];
  uint32_t index = 0;

  load_input(input1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.word_embeddings_in.txt", 1 * 41);
  // load_input(input1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.0_in.txt", 41 * 1024);
  load_input(emb_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.word_embeddings.weight.txt", 1024 * 250880);
  load_input(wel_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.word_embeddings_layernorm.weight.txt", 1024);
  load_input(wel_bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.word_embeddings_layernorm.bias.txt", 1024);
  load_input(lnf_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.ln_f.weight.txt", 1024);
  load_input(lnf_bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.ln_f.bias.txt", 1024);

  for (uint32_t i = 0; i < 1 * 41; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input1[i]));
  }
  index = 1024;
  // for (uint32_t i = 0; i < 1024*41; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(input1[i]));
  // }
  for(uint32_t i = 0; i < 250880 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(emb_weight[i]));
  }
  for(uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(wel_weight[i]));
  }
  for(uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(wel_bias[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(lnf_weight[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(lnf_bias[i]));
  }
  for (int i = 0; i <= 23; i++)
  {
    float *matrix1 = new float[1024];
    load_input(matrix1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".input_layernorm.weight.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix1[j]));
    float *matrix2 = new float[1024];
    load_input(matrix2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".input_layernorm.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix2[j]));
    float *matrix3 = new float[3 * 1024 * 1024];
    load_input(matrix3, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight.txt", 3 * 1024 * 1024);
    for (uint32_t j = 0; j < 3 * 1024 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix3[j]));
    float *matrix4 = new float[3 * 1024];
    load_input(matrix4, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias.txt", 3 * 1024);
    for (uint32_t j = 0; j < 3 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix4[j]));
    float *matrix5  = new float[1024 * 1024];
    load_input(matrix5, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.weight.txt", 1024 * 1024);
    for (uint32_t j = 0; j < 1024 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix5[j]));
    float *matrix6 = new float[1024];
    load_input(matrix6, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix6[j]));
    float *matrix7 = new float[1024];
    load_input(matrix7, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix7[j]));
    float *matrix8 = new float[1024];
    load_input(matrix8, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix8[j]));
    float *matrix9 = new float[1024 * 4096];
    load_input(matrix9, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight.txt", 1024 * 4096);
    for (uint32_t j = 0; j < 1024 * 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix9[j]));
    float *matrix10 = new float[4096];
    load_input(matrix10, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix10[j]));
    float *matrix11 = new float[1024 * 4096];
    load_input(matrix11, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight.txt", 1024 * 4096);
    for (uint32_t j = 0; j < 1024 * 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix11[j]));
    float *matrix12 = new float[1024];
    load_input(matrix12, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix12[j]));
  } 

  simulator.WriteToHBM(data, index, 0);

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
    auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);
     

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);

    for (auto it = range.first; it != range.second; it++)
    {
      float* test = new float[it->second.len];
      std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
      load_input(test, it->second.compare_file, *(int*)(&it->second.len));
      simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
      // if ((it->second.name == "transformer.h.0.mlp_in") || (it->second.name == "transformer.h.0.mlp_out")
      //  || (it->second.name == "transformer.h.0.self_attention_in") || (it->second.name == "transformer.h.0.self_attention_out")
      //  || (it->second.name == "transformer.h.0_out"))
      // {
      //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
      // }
      // if(it->second.name == "transformer.h.23.self_attention.bloombmm_out")
      // {
      //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
      // }
    }
  }

  std::cout << "Execute end\n";
  // simulator.DebugPrintVmem_Write(result.addr, result.addr + result.size(), "output");
  // simulator.PrintHBM(weights_addr.at("weights_transformer_h_0_ln_1_bias"), weights_addr.at("weights_transformer_h_0_ln_1_bias") + 768);
  // simulator.PrintHBM(weights_addr.at("weights_transformer_h_0_attn_c_proj_weight"), weights_addr.at("weights_transformer_h_0_attn_masked_bias"));
  // simulator.DebugPrintVmem(0, 13*3072);
  // std::cout << "result.dims: " << result.dims.size() << std::endl;
  // for(uint32_t i = 0; i <  result.dims.size(); i++) {
  //   std::cout << "result.dim[" << i << "]: " << result.dims[i] << std::endl;
  // }
  // simulator.DebugPrintVmem(std::get<0>(result).addr, std::get<0>(result).addr+std::get<0>(result).size());

  // float* test1 = new float[41 * 1024];
  // load_input(test1, "/home/yinxun/workspace/dlc_simulator/src/simple_test/bloom_560m_forward_f16/transformer.h.0.self_attention_out.txt", 41 * 1024);
  // simulator.DebugPrintVmem_tensor(result.addr, result.addr + result.size(), test1);
  return ;
}

void BLOOMMLPBackwardTest()
{
  std::cout << "BLOOMMLPBackward\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  ShowFuncCallInfo() = true;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 115 * 1024);
  data<4> hidden_states(0, {1, 1, 115, 1024});
  data<4> forward_proj(hidden_states.addr + hidden_states.size(), {1, 1, 115, 4096});
  data<4> gelu_for(forward_proj.addr + forward_proj.size(), {1, 1, 115, 4096});
  data<4> forward_fc(gelu_for.addr + gelu_for.size(), {1, 1, 115, 1024});
  data<2> _proj_weight(forward_fc.addr + forward_fc.size(), {1024, 4096});
  data<1> _proj_bias(_proj_weight.addr + _proj_weight.size(), {1024});
  data<2> _fc_weight(_proj_bias.addr + _proj_bias.size(), {4096, 1024});
  data<1> _fc_bias(_fc_weight.addr + _fc_weight.size(), {4096});

  BLOOMConfig config;
  config.training = true;
  HBMAddr() = 0;
  std::cout << "_proj_weight" << _proj_weight.addr << std::endl;
  std::cout << "_fc_weight" << _fc_weight.addr << std::endl;
  std::map<std::string, uint32_t> weight_map{{"transformer.h.23.mlp.dense_4h_to_h.weight", _proj_weight.addr}, 
                                            {"transformer.h.23.mlp.dense_4h_to_h.bias", _proj_bias.addr},
                                            {"transformer.h.23.mlp.dense_h_to_4h.weight", _fc_weight.addr}, 
                                            {"transformer.h.23.mlp.dense_h_to_4h.bias", _fc_bias.addr}};

  std::map<std::string, uint32_t> forward_map{{"transformer.h.23.mlp.dense_4h_to_h_in", forward_proj.addr}, 
                                              {"transformer.h.23.mlp.gelu_impl_in", gelu_for.addr},
                                              {"transformer.h.23.mlp.dense_h_to_4h_in", forward_fc.addr}};

  data<4> result = BLOOMMLPBackward(inst2, config, hidden_states, hidden_states.addr + hidden_states.size(), hidden_states.addr + hidden_states.size(), "transformer.h.23.mlp", weight_map, forward_map);

  // HBM_TO_VMEM(instruction_list, result.addr, 0, 1024);
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t *data = new uint32_t[115 * 1024 + 115 * 4096 + 115 * 4096 + 115 * 1024 + 1024 * 4096 + 1024 + 4096 * 1024 + 4096];
  float *input1 = new float[115 * 1024];
  float *forward_1 = new float[115 * 4096];
  float *gelu = new float[115 * 4096];
  float *forward_2 = new float[115 * 1024];
  float *proj_weight = new float[1024 * 4096];
  float *proj_bias = new float[1024];
  float *fc_weight = new float[4096 * 1024];
  float *fc_bias = new float[4096];

  uint32_t index = 0;

  load_input(input1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32_drop/transformer.h.12.mlp_in.txt", 115 * 1024);
  load_input(forward_1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32/transformer.h.12.mlp.dense_4h_to_h_in.txt", 115 * 4096);
  load_input(gelu, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32/transformer.h.12.mlp.gelu_impl_in.txt", 115 * 4096);
  load_input(forward_2, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32/transformer.h.12.mlp.dense_h_to_4h_in.txt", 115 * 1024);
  load_input(proj_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.12.mlp.dense_4h_to_h.weight.txt", 1024 * 4096);
  load_input(proj_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.12.mlp.dense_4h_to_h.bias.txt", 1024);
  load_input(fc_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.12.mlp.dense_h_to_4h.weight.txt", 4096 * 1024);
  load_input(fc_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.12.mlp.dense_h_to_4h.bias.txt", 4096);

  for (uint32_t i = 0; i < 115 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input1[i]));
  }
  for (uint32_t i = 0; i < 115 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(forward_1[i]));
  }
  for (uint32_t i = 0; i < 115 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(gelu[i]));
  }
  for (uint32_t i = 0; i < 115 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(forward_2[i]));
  }
  for (uint32_t i = 0; i < 1024 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(proj_weight[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(proj_bias[i]));
  }
  for (uint32_t i = 0; i < 4096 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(fc_weight[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(fc_bias[i]));
  }
  simulator.WriteToHBM(data, 115 * 1024 + 115 * 4096 + 115 * 4096 + 115 * 1024 + 1024 * 4096 + 1024 + 4096 * 1024 + 4096, 0);

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
    auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);
     

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);

    // for (auto it = range.first; it != range.second; it++)
    // {
    //   float* test = new float[it->second.len];
    //   std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
    //   // std::cout << "new cycle: " << cyclenum << std::endl;
    //   load_input(test, it->second.compare_file, *(int*)(&it->second.len));
    //   simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
    // }
  }

  std::cout << "Execute end\n";
  simulator.DebugPrintVmem_Write(result.addr, result.addr + result.size(), "output");

  return ;
}

void BLOOMBlockBackwardTest()
{
  std::cout << "BLOOMBlockBackward\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 41 * 1024);
  data<4> hidden_states(0, {1, 1, 41, 1024});
  data<4> forward_proj(hidden_states.addr + hidden_states.size(), {1, 1, 41, 4096});
  data<4> gelu_for(forward_proj.addr + forward_proj.size(), {1, 1, 41, 4096});
  data<4> forward_fc(gelu_for.addr + gelu_for.size(), {1, 1, 41, 1024});
  data<2> _proj_weight(forward_fc.addr + forward_fc.size(), {1024, 4096});
  data<1> _proj_bias(_proj_weight.addr + _proj_weight.size(), {1024});
  data<2> _fc_weight(_proj_bias.addr + _proj_bias.size(), {4096, 1024});
  data<1> _fc_bias(_fc_weight.addr + _fc_weight.size(), {4096});
  data<4> forward_in(_fc_bias.addr + _fc_bias.size(), {1, 1, 41, 1024});
  data<1> forward_weight(forward_in.addr + forward_in.size(), {1024});
  data<1> forward_bias(forward_weight.addr + forward_weight.size(), {1024});
  data<2> dense_weight(forward_bias.addr + forward_bias.size(), {1024, 1024});
  data<1> dense_bias(dense_weight.addr + dense_weight.size(), {1024});
  data<2> dense_forward_in(dense_bias.addr + dense_bias.size(), {41, 1024});
  data<3> bmm_in_1(dense_forward_in.addr + dense_forward_in.size(), {16, 41, 41});
  int temp = (((bmm_in_1.addr + bmm_in_1.size()) + 128) / 128) * 128;
  data<3> bmm_in_2(temp, {16, 41, 64});
  data<4> softmax_forward_out(bmm_in_2.addr + bmm_in_2.size(), {1, 16, 41, 41});
  int temp1 = (((softmax_forward_out.addr + softmax_forward_out.size()) + 128) / 128) * 128;
  data<3> baddbmm_in_1(temp1, {16, 41, 64});
  data<3> baddbmm_in_2(baddbmm_in_1.addr + baddbmm_in_1.size(), {16, 64, 41});
  data<2> qkv_weight(baddbmm_in_2.addr + baddbmm_in_2.size(), {3072, 1024});
  data<1> qkv_bias(qkv_weight.addr + qkv_weight.size(), {3072});
  data<3> qkv_forward_in(qkv_bias.addr + qkv_bias.size(), {1, 41, 1024});

  BLOOMConfig config;

  HBMAddr() = 0;
  std::map<std::string, uint32_t> weight_map{{"transformer.h.23.mlp.dense_4h_to_h.weight", _proj_weight.addr},
                                            {"transformer.h.23.mlp.dense_4h_to_h.bias", _proj_bias.addr},
                                            {"transformer.h.23.mlp.dense_h_to_4h.weight", _fc_weight.addr},
                                            {"transformer.h.23.mlp.dense_h_to_4h.bias", _fc_bias.addr},
                                            {"transformer.h.23.post_attention_layernorm.weight", forward_weight.addr},
                                            {"transformer.h.23.post_attention_layernorm.bias", forward_bias.addr},
                                            {"transformer.h.23.self_attention.dense.weight", dense_weight.addr},
                                            {"transformer.h.23.self_attention.dense.bias", dense_bias.addr},
                                            {"transformer.h.23.self_attention.query_key_value.weight", qkv_weight.addr},
                                            {"transformer.h.23.self_attention.query_key_value.bias", qkv_bias.addr}};

  std::map<std::string, uint32_t> forward_map{{"transformer.h.23.mlp.dense_4h_to_h_in", forward_proj.addr},
                                              {"transformer.h.23.mlp.gelu_impl_in", gelu_for.addr},
                                              {"transformer.h.23.mlp.dense_h_to_4h_in", forward_fc.addr},
                                              {"transformer.h.23.post_attention_layernorm_in", forward_in.addr},
                                              {"transformer.h.23.self_attention.dense_in", dense_forward_in.addr},
                                              {"transformer.h.23.self_attention.attention_probs_reshaped", bmm_in_1.addr},
                                              {"transformer.h.23.self_attention.value_layer", bmm_in_2.addr},
                                              {"transformer.h.23.self_attention.softmax_out", softmax_forward_out.addr},
                                              {"transformer.h.23.self_attention.query_layer", baddbmm_in_1.addr},
                                              {"transformer.h.23.self_attention.key_layer", baddbmm_in_2.addr},
                                              {"transformer.h.23.self_attention.query_key_value_in", qkv_forward_in.addr}};

  data<4> result = BLOOMBlockBackward(inst2, config, hidden_states, hidden_states.addr + hidden_states.size(), hidden_states.addr + hidden_states.size(), "transformer.h.23", weight_map, forward_map);

  HBM_TO_VMEM(instruction_list, 8856576, 0, 1024);
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t *data = new uint32_t[41 * 1024 * 11 + 1024 * 4096 * 2 + 1024 * 6 + 1024 * 1024 + 1024 + 41 * 1024 + 16 * 41 * 41 + 16 * 41 * 64 + 16 * 41 * 41 + 16 * 41 * 64 + 16 * 64 * 41 + 3072 * 1024 + 3072 + 41 * 1024];
  float *input1 = new float[41 * 1024];
  float *forward_1 = new float[41 * 4096];
  float *gelu = new float[41 * 4096];
  float *forward_2 = new float[41 * 1024];
  float *dense1_weight = new float[1024 * 4096];
  float *dense1_bias = new float[1024];
  float *dense2_weight = new float[4096 * 1024];
  float *dense2_bias = new float[4096];
  float *post_in = new float[41 * 1024];
  float *post_weight = new float[1024];
  float *post_bias = new float[1024];
  float *denseweight = new float[1024 * 1024];
  float *densebias = new float[1024];
  float *denseforwardin = new float[41 * 1024];
  float *bmmin1 = new float[16 * 41 * 41];
  float *bmmin2 = new float[16 * 41 * 64];
  float *softmaxforwardout = new float[16 * 41 * 41];
  float *baddbmmin1 = new float[16 * 41 * 64];
  float *baddbmmin2 = new float[16 * 64 * 41];
  float *qkvweight = new float[3072 * 1024];
  float *qkvbias = new float[3072];
  float *qkvforwardin = new float[41 * 1024];
  uint32_t index = 0;

  load_input(input1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23.post_attention_layernorm_in.txt", 41 * 1024);
  load_input(forward_1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.mlp.dense_4h_to_h_in.txt", 41 * 4096);
  load_input(gelu, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.mlp.gelu_impl_in.txt", 41 * 4096);
  load_input(forward_2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.mlp.dense_h_to_4h_in.txt", 41 * 1024);
  load_input(dense1_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.mlp.dense_4h_to_h.weight.txt", 1024 * 4096);
  load_input(dense1_bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.mlp.dense_4h_to_h.bias.txt", 1024);
  load_input(dense2_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.mlp.dense_h_to_4h.weight.txt", 1024 * 4096);
  load_input(dense2_bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.mlp.dense_h_to_4h.bias.txt", 4096);
  load_input(post_in, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.post_attention_layernorm_in.txt", 41 * 1024);
  load_input(post_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.post_attention_layernorm.weight.txt", 1024);
  load_input(post_bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.post_attention_layernorm.bias.txt", 1024);
  load_input(denseweight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.dense.weight.txt", 1024 * 1024);
  load_input(densebias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.dense.bias.txt", 1024);
  load_input(denseforwardin, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.dense_in.txt", 41 * 1024);
  load_input(bmmin1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.attention_probs_reshaped.txt", 16 * 41 * 41);
  load_input(bmmin2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.value_layer.txt", 16 * 64 * 41);
  load_input(softmaxforwardout, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.softmax_out.txt", 16 * 41 * 41);
  load_input(baddbmmin1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.query_layer.txt", 16 * 41 * 64);
  load_input(baddbmmin2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.key_layer.txt", 16 * 64 * 41);
  load_input(qkvweight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.query_key_value.weight.txt", 3072 * 1024);
  load_input(qkvbias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.query_key_value.bias.txt", 3072);
  load_input(qkvforwardin, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.query_key_value_in.txt", 41 * 1024);
  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input1[i]));
  }
  for (uint32_t i = 0; i < 41 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(forward_1[i]));
  }
  for (uint32_t i = 0; i < 41 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(gelu[i]));
  }
  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(forward_2[i]));
  }
  for (uint32_t i = 0; i < 1024 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(dense1_weight[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(dense1_bias[i]));
  }
  for (uint32_t i = 0; i < 1024 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(dense2_weight[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(dense2_bias[i]));
  }
  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(post_in[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(post_weight[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(post_bias[i]));
  }
  for (uint32_t i = 0; i < 1024 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&denseweight[i]);
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&densebias[i]);
  }
  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&denseforwardin[i]);
  }
  for (uint32_t i = 0; i < 16 * 41 * 41; i++, index++)
  {
    data[index] = *(uint32_t*)(&bmmin1[i]);
  }
  index  = ((index + 128) / 128) * 128;
  for (uint32_t i = 0; i < 16 * 41 * 64; i++, index++)
  {
    data[index] = *(uint32_t*)(&bmmin2[i]);
  }
  for (uint32_t i = 0; i < 16 * 41 *41; i++, index++)
  {
    data[index] = *(uint32_t*)(&softmaxforwardout[i]);
  }
  index  = ((index + 128) / 128) * 128;
  for (uint32_t i = 0; i < 16 * 41 * 64; i++, index++)
  {
    data[index] = *(uint32_t*)(&baddbmmin1[i]);
  }
  for (uint32_t i = 0; i < 16 * 41 * 64; i++, index++)
  {
    data[index] = *(uint32_t*)(&baddbmmin2[i]);
  }
  for (uint32_t i = 0; i < 3072 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&qkvweight[i]);
  }
  for (uint32_t i = 0; i < 3072; i++, index++)
  {
    data[index] = *(uint32_t*)(&qkvbias[i]);
  }
  for (uint32_t i = 0; i< 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&qkvforwardin[i]);
  }
  
  
  simulator.WriteToHBM(data, 41 * 1024 * 11 + 1024 * 4096 * 2 + 1024 * 6 + 1024 * 1024 + 1024 + 41 * 1024 + 16 * 41 * 41 + 16 * 41 * 64 + 16 * 41 * 41 + 16 * 41 * 64 + 16 * 64 * 41 + 3072 * 1024 + 3072 + 41 * 1024 + 1024, 0);

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }

  std::cout << "Execute end\n";
  simulator.DebugPrintVmem(0, 1024);
  // simulator.DebugPrintVmem_Write(result.addr, result.addr + result.size(), "output");
  // simulator.DebugPrintVmem(result.addr, result.addr+result.size());
  // float* test1 = new float[41 * 1024];
  // load_input(test1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.ln_f_out.txt", 41 * 1024);
  // simulator.DebugPrintVmem_tensor(result.addr, result.addr + result.size(), test1);
  return ;
}

void BLOOMModelBackwardTest()
{
  std::cout << "BLOOMModelBackward\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  // HBM_TO_VMEM(instruction_list, 0, 0, 41 * 1024);
  // data<4> hidden_states(0, {1, 1, 41, 1024});
  // data<4> ln_in(hidden_states.addr + hidden_states.size(), {1, 1, 41, 1024});
  // data<1> ln_f_weight(ln_in.addr + ln_in.size(), {1024});
  // data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {1024});
  // data<4> embedding_in(ln_f_bias.addr + ln_f_bias.size(), {1, 1, 41, 1024});
  // data<1> embedding_weight(embedding_in.addr + embedding_in.size(), {1024});
  // data<1> embedding_bias(embedding_weight.addr + embedding_weight.size(), {1024});
  // BLOOMConfig config;

  // HBMAddr() = 0;
  // std::map<std::string, uint32_t> forward_map{{"transformer.ln_f_in", ln_in.addr}, {"transformer.word_embeddings_layernorm_in", embedding_in.addr}};
  // std::map<std::string, uint32_t> weights_map{{"transformer.ln_f.weight", ln_f_weight.addr},
  //                                             {"transformer.ln_f.bias", ln_f_bias.addr},
  //                                             {"transformer.word_embeddings_layernorm.weight", embedding_weight.addr},
  //                                             {"transformer.word_embeddings_layernorm.bias", embedding_bias.addr}};
  // uint32_t forwardaddr = embedding_bias.addr + embedding_bias.size();
  // for (int i = 23; i >= 23; i--)
  // {
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h_in", forwardaddr));
  //   forwardaddr += 41 * 4096;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.gelu_impl_in", forwardaddr));
  //   forwardaddr += 41 * 4096;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h_in", forwardaddr));
  //   forwardaddr += 41 * 1024;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm_in", forwardaddr));
  //   forwardaddr += 41 * 1024;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense_in", forwardaddr));
  //   forwardaddr += 41 * 1024;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.attention_probs_reshaped", forwardaddr));
  //   forwardaddr += 16 * 41 * 41;
  //   forwardaddr = ((forwardaddr + 128) / 128) * 128;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.value_layer", forwardaddr));
  //   forwardaddr += 41 * 1024; 
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.softmax_out", forwardaddr));
  //   forwardaddr += 16 * 41 * 41;
  //   forwardaddr = ((forwardaddr + 128) / 128) * 128;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_layer", forwardaddr));
  //   forwardaddr += 41 * 1024;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.key_layer", forwardaddr));
  //   forwardaddr += 41 * 1024;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value_in", forwardaddr));
  //   forwardaddr += 41 * 1024;
  //   forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm_in", forwardaddr));
  //   forwardaddr += 1024 * 41;
  // }
  // for(int i = 23; i >= 23; i--)
  // {
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight", forwardaddr));
  //   forwardaddr += 1024 * 4096;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias", forwardaddr));
  //   forwardaddr += 1024;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight", forwardaddr));
  //   forwardaddr += 1024 * 4096;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias", forwardaddr));
  //   forwardaddr += 4096;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight", forwardaddr));
  //   forwardaddr += 1024;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.weight", forwardaddr));
  //   forwardaddr += 1024 * 1024;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.bias", forwardaddr));
  //   forwardaddr += 1024;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight", forwardaddr));
  //   forwardaddr += 3072 * 1024;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias", forwardaddr));
  //   forwardaddr += 3072;
  //   weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.weight", forwardaddr));
  //   forwardaddr += 1024;
  // }

  HBM_TO_VMEM(instruction_list, 0, 0, 41 * 4096);
  data<4> hidden_states(0, {1, 1, 41, 4096});
  data<4> ln_in(hidden_states.addr + hidden_states.size(), {1, 1, 41, 4096});
  data<1> ln_f_weight(ln_in.addr + ln_in.size(), {4096});
  data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {4096});
  data<4> embedding_in(ln_f_bias.addr + ln_f_bias.size(), {1, 1, 41, 4096});
  data<1> embedding_weight(embedding_in.addr + embedding_in.size(), {4096});
  data<1> embedding_bias(embedding_weight.addr + embedding_weight.size(), {4096});
  BLOOMConfig config;

  HBMAddr() = 0;
  std::map<std::string, uint32_t> forward_map{{"transformer.ln_f_in", ln_in.addr}, {"transformer.word_embeddings_layernorm_in", embedding_in.addr}};
  std::map<std::string, uint32_t> weights_map{{"transformer.ln_f.weight", ln_f_weight.addr},
                                              {"transformer.ln_f.bias", ln_f_bias.addr},
                                              {"transformer.word_embeddings_layernorm.weight", embedding_weight.addr},
                                              {"transformer.word_embeddings_layernorm.bias", embedding_bias.addr}};
  uint32_t forwardaddr = embedding_bias.addr + embedding_bias.size();
  for (int i = 29; i >= 29; i--)
  {
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h_in", forwardaddr));
    forwardaddr += 41 * 4096 * 4;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.gelu_impl_in", forwardaddr));
    forwardaddr += 41 * 4096 * 4;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h_in", forwardaddr));
    forwardaddr += 41 * 4096;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm_in", forwardaddr));
    forwardaddr += 41 * 4096;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense_in", forwardaddr));
    forwardaddr += 41 * 4096;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.attention_probs_reshaped", forwardaddr));
    forwardaddr += 32 * 41 * 41;
    forwardaddr = ((forwardaddr + 128) / 128) * 128;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.value_layer", forwardaddr));
    forwardaddr += 41 * 4096; 
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.softmax_out", forwardaddr));
    forwardaddr += 32 * 41 * 41;
    forwardaddr = ((forwardaddr + 128) / 128) * 128;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_layer", forwardaddr));
    forwardaddr += 41 * 4096;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.key_layer", forwardaddr));
    forwardaddr += 41 * 4096;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value_in", forwardaddr));
    forwardaddr += 41 * 4096;
    forward_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm_in", forwardaddr));
    forwardaddr += 4096 * 41;
  }
  for(int i = 29; i >= 29; i--)
  {
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) +".mlp.dense_4h_to_h.weight", forwardaddr));
    forwardaddr += 4096 * 4096 * 4;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias", forwardaddr));
    forwardaddr += 4096;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight", forwardaddr));
    forwardaddr += 4096 * 4096 * 4;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias", forwardaddr));
    forwardaddr += 4096 * 4;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight", forwardaddr));
    forwardaddr += 4096;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias", forwardaddr));
    forwardaddr += 4096;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.weight", forwardaddr));
    forwardaddr += 4096 * 4096;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.bias", forwardaddr));
    forwardaddr += 4096;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight", forwardaddr));
    forwardaddr += 3072 * 4096 * 4;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias", forwardaddr));
    forwardaddr += 3072 * 4;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.weight", forwardaddr));
    forwardaddr += 4096;
    weights_map.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.bias", forwardaddr));
    forwardaddr += 4096;
  }

  uint32_t scalar_local_time_reg = 31;

  if (1) {

    inst = new Instruction();

    ScalarOperationState read_time(S_READ, 0, 0, 0, scalar_local_time_reg);

    inst->SetOperationState(Instruction::SCALARONE, &read_time);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  // data<4> result = BLOOMModelBackward(inst2, config, hidden_states, hidden_states.addr + hidden_states.size(), hidden_states.addr + hidden_states.size(), "transformer", weights_map, forward_map);
  // data<4> result = BLOOMBlockBackward(inst2, config, hidden_states, hidden_states.addr + hidden_states.size(), hidden_states.addr + hidden_states.size(), "transformer.h.29", weights_map, forward_map);
  // data<4> result = BLOOMMLPBackward(inst2, config, hidden_states, hidden_states.addr + hidden_states.size(), hidden_states.addr + hidden_states.size(), "transformer.h.29.mlp", weights_map, forward_map);
  data<3> result = BLOOMAttentionBackward(inst2, config, hidden_states[0], "transformer.h.29.self_attention", weights_map, forward_map, hidden_states.addr + hidden_states.size());

  uint32_t scalar_local_time_reg1 = 30;

  if (1) {

    inst = new Instruction();

    ScalarOperationState read_time(S_READ, 0, 0, 0, scalar_local_time_reg1);

    inst->SetOperationState(Instruction::SCALARONE, &read_time);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  if (1) {

    inst = new Instruction();

    ScalarOperationState sub(S_S32_SUBTRACTION, 0,

      30, 31, 29);

    inst->SetOperationState(Instruction::SCALARONE, &sub);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  if(1) {

    inst = new Instruction();

    ScalarOperationState set_base1(S_U32_MOVE, 0, 0, 46, 28);

    inst->SetOperationState(Instruction::SCALARONE, &set_base1);

    ScalarOperationState store(S_SMEM_STORE, 0, 31, 28, 0);

    inst->SetOperationState(Instruction::SCALARTWO, &store);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  if(1) {

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE0, 4);

    ScalarOperationState set_base1(S_U32_MOVE, 0, 0, 32, 28);

    inst->SetOperationState(Instruction::SCALARONE, &set_base1);

    ScalarOperationState store(S_SMEM_STORE, 0, 30, 28, 0);

    inst->SetOperationState(Instruction::SCALARTWO, &store);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  if(1) {

    inst = new Instruction();

    inst->SetImmediateValue(Instruction::IMMEDIATE0, 8);

    ScalarOperationState set_base1(S_U32_MOVE, 0, 0, 32, 28);

    inst->SetOperationState(Instruction::SCALARONE, &set_base1);

    ScalarOperationState store(S_SMEM_STORE, 0, 29, 28, 0);

    inst->SetOperationState(Instruction::SCALARTWO, &store);

    CompleteInstruction(inst);

    instruction_list.push_back(inst);

  }

  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends
  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  std::cout << "forwardaddr: " << forwardaddr << std::endl;
  // uint32_t *data = new uint32_t[forwardaddr];
  // float *input = new float[41 * 1024];
  // float *ln_f_in = new float[41 * 1024];
  // float *embedding = new float[41 * 1024];
  // float *ln_f_in_weight = new float[1024];
  // float *ln_f_in_bias = new float[1024];
  // float *embedding_in_weight = new float[1024];
  // float *embedding_in_bias = new float[1024];
  // uint32_t index = 0;

  // std::string forstr = "f32";
  // std::string backstr = "f32";
  // std::string weightstr = "f32";

  // load_input(input, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23_in.txt", 41 * 1024);
  // // load_input(input, "/home/yinxun/workspace/dlc_simulator/src/velocedata/transformer.h.23_in.txt", 41 * 1024);
  // load_input(ln_f_in, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.ln_f_in.txt", 41 * 1024);
  // load_input(ln_f_in_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.ln_f.weight.txt", 1024);
  // load_input(ln_f_in_bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.ln_f.bias.txt", 1024);
  // load_input(embedding, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.word_embeddings_layernorm_in.txt", 41 * 1024);
  // load_input(embedding_in_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.word_embeddings_layernorm.weight.txt", 1024);
  // load_input(embedding_in_bias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.word_embeddings_layernorm.bias.txt", 1024);

  // for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(input[i]));
  // }
  // for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(ln_f_in[i]));
  // }
  // for (uint32_t i = 0; i < 1024; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(ln_f_in_weight[i]));
  // }
  // for (uint32_t i = 0; i < 1024; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(ln_f_in_bias[i]));
  // }
  // for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(embedding[i]));
  // }
  // for (uint32_t i = 0; i < 1024; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(embedding_in_weight[i]));
  // }
  // for (uint32_t i = 0; i < 1024; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(embedding_in_bias[i]));
  // }
  // std::cout << "first index: " << index << std::endl;
  // for (int i = 23; i >= 23; i--)
  // {
  //   if (i >= 0)
  //   {
  //     std::cout << "forward i: " << i << std::endl;
  //     float *matrix1 = new float[41 * 4096];
  //     load_input(matrix1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h_in.txt", 41 * 4096);
  //     for (uint32_t j = 0; j < 41 * 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix1[j]));
  //     float *matrix2 = new float[41 * 4096];
  //     load_input(matrix2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".mlp.gelu_impl_in.txt", 41 * 4096);
  //     for (uint32_t j = 0; j < 41 * 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix2[j]));
  //     float *matrix3 = new float[41 * 1024];
  //     load_input(matrix3, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h_in.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix3[j]));
  //     float *matrix4 = new float[41 * 1024];
  //     load_input(matrix4, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm_in.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix4[j]));
  //     float *matrix5 = new float[41 * 1024];
  //     load_input(matrix5, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.dense_in.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix5[j]));
  //     float *matrix6 = new float[41 * 41 * 16];
  //     load_input(matrix6, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.attention_probs_reshaped.txt", 41 * 41 * 16);
  //     for (uint32_t j = 0; j < 41 * 41 * 16; j++, index++) data[index] = *(uint32_t*)(&(matrix6[j]));
  //     index = (index + 128) / 128 * 128;
  //     float *matrix7 = new float[41 * 1024];
  //     load_input(matrix7, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.value_layer.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix7[j]));
  //     float *matrix8 = new float[41 * 41 * 16];
  //     load_input(matrix8, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.softmax_out.txt", 41 * 41 * 16);
  //     for (uint32_t j = 0; j < 41 * 41 * 16; j++, index++) data[index] = *(uint32_t*)(&(matrix8[j]));
  //     index = (index + 128) / 128 * 128;
  //     float *matrix9 = new float[41 * 1024];
  //     load_input(matrix9, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.query_layer.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix9[j]));
  //     float *matrix10 = new float[41 * 1024];
  //     load_input(matrix10, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.key_layer.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix10[j]));
  //     float *matrix11 = new float[41 * 1024];
  //     load_input(matrix11, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value_in.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix11[j]));
  //     float *matrix12 = new float[41 * 1024];
  //     load_input(matrix12, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h." + std::to_string(i) + ".input_layernorm_in.txt", 41 * 1024);
  //     for (uint32_t j = 0; j < 41 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix12[j]));
  //   }
  // }
  // for (int i = 23; i >= 23; i--)
  // {
  //   if(1)
  //   {
  //     std::cout <<  "weight i: "  << i << std::endl;
  //     float *matrix1 = new float[4096 * 1024];
  //     load_input(matrix1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight.txt", 4096 * 1024);
  //     for (uint32_t j = 0; j < 4096 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix1[j]));
  //     float *matrix2 = new float[1024];
  //     load_input(matrix2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias.txt", 1024);
  //     for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix2[j]));
  //     float *matrix3 = new float[1024 * 4096];
  //     load_input(matrix3, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight.txt", 1024 * 4096);
  //     for (uint32_t j = 0; j < 1024 * 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix3[j]));
  //     float *matrix4 = new float[4096];
  //     load_input(matrix4, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias.txt", 4096);
  //     for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix4[j]));
  //     float *matrix5 = new float[1024];
  //     load_input(matrix5, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight.txt", 1024);
  //     for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix5[j]));
  //     float *matrix6 = new float[1024 * 1024];
  //     load_input(matrix6, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".self_attention.dense.weight.txt", 1024 * 1024);
  //     for (uint32_t j = 0; j < 1024 * 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix6[j]));
  //     float *matrix7 = new float[1024];
  //     load_input(matrix7, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".self_attention.dense.bias.txt", 1024);
  //     for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix7[j]));
  //     float *matrix8 = new float[1024 * 3072];
  //     load_input(matrix8, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight.txt", 1024 * 3072);
  //     for (uint32_t j = 0; j < 1024 * 3072; j++, index++) data[index] = *(uint32_t*)(&(matrix8[j]));
  //     float *matrix9 = new float[3072];
  //     load_input(matrix9, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias.txt", 3072);
  //     for (uint32_t j = 0; j < 3072; j++, index++) data[index] = *(uint32_t*)(&(matrix9[j]));
  //     float *matrix10 = new float[1024];
  //     load_input(matrix10, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_" + weightstr + "/transformer.h." + std::to_string(i) + ".input_layernorm.weight.txt", 1024);
  //     for (uint32_t j = 0; j < 1024; j++, index++) data[index] = *(uint32_t*)(&(matrix10[j]));
  //   }
  // }

  uint32_t *data = new uint32_t[forwardaddr];
  float *input = new float[41 * 4096];
  float *ln_f_in = new float[41 * 4096];
  float *embedding = new float[41 * 4096];
  float *ln_f_in_weight = new float[4096];
  float *ln_f_in_bias = new float[4096];
  float *embedding_in_weight = new float[4096];
  float *embedding_in_bias = new float[4096];
  uint32_t index = 0;

  load_input(input, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h.29.self_attention_in.txt", 41 * 4096);
  // load_input(input, "/home/yinxun/workspace/dlc_simulator/src/velocedata/transformer.h.23_in.txt", 41 * 1024);
  load_input(ln_f_in, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.ln_f_in.txt", 41 * 4096);
  load_input(ln_f_in_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.ln_f.weight.txt", 4096);
  load_input(ln_f_in_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.ln_f.bias.txt", 4096);
  load_input(embedding, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.word_embeddings_layernorm_in.txt", 41 * 4096);
  load_input(embedding_in_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.word_embeddings_layernorm.weight.txt", 4096);
  load_input(embedding_in_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.word_embeddings_layernorm.bias.txt", 4096);

  for (uint32_t i = 0; i < 41 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input[i]));
  }
  for (uint32_t i = 0; i < 41 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(ln_f_in[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(ln_f_in_weight[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(ln_f_in_bias[i]));
  }
  for (uint32_t i = 0; i < 41 * 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(embedding[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(embedding_in_weight[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    data[index] = *(uint32_t*)(&(embedding_in_bias[i]));
  }
  std::cout << "first index: " << index << std::endl;
  for (int i = 29; i >= 29; i--)
  {
    if (i >= 0)
    {
      std::cout << "forward i: " << i << std::endl;
      float *matrix1 = new float[41 * 4096 * 4];
      load_input(matrix1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h_in.txt", 41 * 4096 * 4);
      for (uint32_t j = 0; j < 41 * 4096 * 4; j++, index++) data[index] = *(uint32_t*)(&(matrix1[j]));
      float *matrix2 = new float[41 * 4096 * 4];
      load_input(matrix2, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".mlp.gelu_impl_in.txt", 41 * 4096 * 4);
      for (uint32_t j = 0; j < 41 * 4096 * 4; j++, index++) data[index] = *(uint32_t*)(&(matrix2[j]));
      float *matrix3 = new float[41 * 4096 ];
      load_input(matrix3, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h_in.txt", 41 * 4096 );
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix3[j]));
      float *matrix4 = new float[41 * 4096 ];
      load_input(matrix4, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm_in.txt", 41 * 4096);
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix4[j]));
      float *matrix5 = new float[41 * 4096 ];
      load_input(matrix5, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.dense_in.txt", 41 * 4096 );
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix5[j]));
      float *matrix6 = new float[41 * 41 * 32];
      load_input(matrix6, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.attention_probs_reshaped.txt", 41 * 41 * 32);
      for (uint32_t j = 0; j < 41 * 41 * 32; j++, index++) data[index] = *(uint32_t*)(&(matrix6[j]));
      index = (index + 128) / 128 * 128;
      float *matrix7 = new float[41 * 4096 ];
      load_input(matrix7, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.value_layer.txt", 41 * 4096 );
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix7[j]));
      float *matrix8 = new float[41 * 41 * 32];
      load_input(matrix8, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.softmax_out.txt", 41 * 41 * 32);
      for (uint32_t j = 0; j < 41 * 41 * 32; j++, index++) data[index] = *(uint32_t*)(&(matrix8[j]));
      index = (index + 128) / 128 * 128;
      float *matrix9 = new float[41 * 4096 ];
      load_input(matrix9, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.query_layer.txt", 41 * 4096 );
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix9[j]));
      float *matrix10 = new float[41 * 4096 ];
      load_input(matrix10, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.key_layer.txt", 41 * 4096 );
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix10[j]));
      float *matrix11 = new float[41 * 4096 ];
      load_input(matrix11, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value_in.txt", 41 * 4096 );
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix11[j]));
      float *matrix12 = new float[41 * 4096 ];
      load_input(matrix12, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_forward_f32/transformer.h." + std::to_string(i) + ".input_layernorm_in.txt", 41 * 4096 );
      for (uint32_t j = 0; j < 41 * 4096 ; j++, index++) data[index] = *(uint32_t*)(&(matrix12[j]));
    }
  }
  for (int i = 29; i >= 29; i--)
  {
    if(1)
    {
      std::cout <<  "weight i: "  << i << std::endl;
      float *matrix1 = new float[4096 * 4096 * 4];
      load_input(matrix1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight.txt", 4096 * 4096 * 4);
      for (uint32_t j = 0; j < 4096 * 4096 * 4; j++, index++) data[index] = *(uint32_t*)(&(matrix1[j]));
      float *matrix2 = new float[4096];
      load_input(matrix2, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias.txt", 4096);
      for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix2[j]));
      float *matrix3 = new float[4096 * 4096 * 4];
      load_input(matrix3, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight.txt", 4096 * 4096 * 4);
      for (uint32_t j = 0; j < 4096 * 4096 * 4; j++, index++) data[index] = *(uint32_t*)(&(matrix3[j]));
      float *matrix4 = new float[4096 * 4];
      load_input(matrix4, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias.txt", 4096 * 4);
      for (uint32_t j = 0; j < 4096 * 4; j++, index++) data[index] = *(uint32_t*)(&(matrix4[j]));
      float *matrix5 = new float[4096];
      load_input(matrix5, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight.txt", 4096);
      for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix5[j]));
      float *matrix12 = new float[4096];
      load_input(matrix12, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias.txt", 4096);
      for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix12[j]));
      float *matrix6 = new float[4096 * 4096];
      load_input(matrix6, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.weight.txt", 4096 * 4096);
      for (uint32_t j = 0; j < 4096 * 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix6[j]));
      float *matrix7 = new float[4096];
      load_input(matrix7, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.bias.txt", 4096);
      for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix7[j]));
      float *matrix8 = new float[4096 * 3072 * 4];
      load_input(matrix8, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight.txt", 4096 * 3072 * 4);
      for (uint32_t j = 0; j < 4096 * 3072 * 4; j++, index++) data[index] = *(uint32_t*)(&(matrix8[j]));
      float *matrix9 = new float[3072 * 4];
      load_input(matrix9, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias.txt", 3072 * 4);
      for (uint32_t j = 0; j < 3072 * 4; j++, index++) data[index] = *(uint32_t*)(&(matrix9[j]));
      float *matrix10 = new float[4096];
      load_input(matrix10, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".input_layernorm.weight.txt", 4096);
      for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix10[j]));
      float *matrix11 = new float[4096];
      load_input(matrix11, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".input_layernorm.bias.txt", 4096);
      for (uint32_t j = 0; j < 4096; j++, index++) data[index] = *(uint32_t*)(&(matrix11[j]));
    }
  }

  std::cout << "data: " << data[130049] << std::endl;
  std::cout << "index: " << index << std::endl;
  simulator.WriteToHBM(data, index, 0);

  std::cout << "instruction size: " << instruction_list.size() << std::endl;

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
    auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);
     

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);

    for (auto it = range.first; it != range.second; it++)
    {
      float* test = new float[it->second.len];
      std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
      // load_input(test, it->second.compare_file, *(int*)(&it->second.len));
      // simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
      // if ((it->second.name == "query_layer") || (it->second.name == "key_layer"))
      // {
      //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
      // }
      // if ((it->second.name == "transformer.h.0.mlp_in") || (it->second.name == "transformer.h.0.mlp_out")
      //  || (it->second.name == "transformer.h.0.self_attention_in") || (it->second.name == "transformer.h.0.self_attention_out")
      //  || (it->second.name == "transformer.h.0_out"))
      // {
      //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
      // }
      // if(it->second.name == "transformer.h.23.self_attention.bloombmm_out")
      // {
      //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
      // }
    }
  }
  std::cout << "Execute end\n";
  simulator.DebugPrintSmem(0, 1024);
  // simulator.DebugPrintVmem_Write(result.addr, result.addr + result.size(), "query_layer");
  // simulator.DebugPrintVmem_Hex(result.addr, result.addr + result.size());
  // simulator.DebugPrintVmem(result.addr, result.addr + result.size());
  // float *test1 = new float[41 * 41 * 16];
  // // load_input(test1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23.mlp.dense_4h_to_h_in.txt", 41 * 1024);
  // load_input(test1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23.self_attention.softmax_out.txt", 41 * 41 * 16);
  // // load_input(test1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward/transformer.word_embeddings_layernorm_out.txt", 41 * 1024);
  // simulator.DebugPrintVmem_tensor(result.addr, result.addr + result.size(), test1);
  // simulator.PrintHBM(130048, 140000);
  return ;
}

void BLOOMForCausalLMBackwardTest()
{
  std::cout << "BLOOMForCausalMBackward\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  // HBM_TO_VMEM(instruction_list, 0, 0, 41 * 250880);
  
  // HBM_TO_VMEM_Stride(instruction_list, 0, 0, 256, 250880 / 128);
  data<4> hidden_states(0, {1, 1, 41, 250880});
  data<2> lm_head(hidden_states.addr + hidden_states.size(), {250880, 1024});
  BLOOMConfig config;

  HBMAddr() = 0;
  std::cout << "lm_head: " << lm_head.addr << std::endl;
  std::cout << "lm_head_end: " << lm_head.addr + lm_head.size() << std::endl;
  std::map<std::string, uint32_t> weight_map{{"hidden_states", hidden_states.addr}, {"transformer.lm_head.weight", lm_head.addr}};

  data<4> result = BLOOMForCausalLMBackward(inst2, config, hidden_states, 0, 0, "transformer", weight_map);

  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  // uint32_t *data = new uint32_t[41 * 250880 + 250880 * 1024];
  // float *input = new float[41 * 250880];
  // float *lm_head_weight = new float[250880 * 1024];
  // uint32_t index = 0;

  // load_input(input, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward/lm_head_in.txt", 41 * 250880);
  // load_input(lm_head_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m/transformer.lm_head.weight.txt", 250880 * 1024);
  

  uint32_t *data = new uint32_t[41 * 250880 + 250880 * 1024];
  float *input = new float[41 * 250880];
  float *lm_head_weight = new float[250880 * 1024];
  uint32_t index = 0;

  load_input(input, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/lm_head_in.txt", 41 * 250880);
  load_input(lm_head_weight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.lm_head.weight.txt", 250880 * 1024);
  

  for (uint32_t i = 0; i < 41 * 250880; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input[i]));
  }
  for (uint32_t i = 0; i < 250880 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(lm_head_weight[i]));
  }
  // simulator.WriteToHBM(data, 41 * 1024, 0);
  simulator.WriteToHBM(data, 41 * 250880 + 250880 * 1024, 0);

  // std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);
  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);


    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }

  std::cout << "Execute end\n";
  // simulator.PrintHBM(0, 400);
  // simulator.DebugPrintVmem(0, 1024);
  float *test = new float[41 * 1024];
  load_input(test, "simple_test/bloom_560m_backward/lm_head_out.txt", 41 * 1024);
  simulator.DebugPrintVmem_tensor(result.addr, result.addr + result.size(), test);
  return ;
}

void BLOOMAttentionBackwardTest()
{
  std::cout << "BLOOMAttentionBakcward\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 41 * 1024);
  data<3> backward_input(0, {1, 41, 1024});
  data<2> dense_weight(backward_input.addr + backward_input.size(), {1024, 1024});
  data<1> dense_bias(dense_weight.addr + dense_weight.size(), {1024});
  data<2> dense_forward_in(dense_bias.addr + dense_bias.size(), {41, 1024});
  data<3> bmm_in_1(dense_forward_in.addr + dense_forward_in.size(), {16, 41, 41});
  int temp = (((bmm_in_1.addr + bmm_in_1.size()) + 128) / 128) * 128;
  data<3> bmm_in_2(temp, {16, 41, 64});
  data<4> softmax_forward_out(bmm_in_2.addr + bmm_in_2.size(), {1, 16, 41, 41});
  int temp1 = (((softmax_forward_out.addr + softmax_forward_out.size()) + 128) / 128) * 128;
  data<3> baddbmm_in_1(temp1, {16, 41, 64});
  data<3> baddbmm_in_2(baddbmm_in_1.addr + baddbmm_in_1.size(), {16, 64, 41});
  data<2> qkv_weight(baddbmm_in_2.addr + baddbmm_in_2.size(), {3072, 1024});
  data<1> qkv_bias(qkv_weight.addr + qkv_weight.size(), {3072});
  data<3> qkv_forward_in(qkv_bias.addr + qkv_bias.size(), {1, 41, 1024});
  BLOOMConfig config;


  HBMAddr() = 0;
  std::map<std::string, uint32_t> weight_map{{"transformer.h.23.self_attention.dense.weight", dense_weight.addr},
                                            {"transformer.h.23.self_attention.dense.bias", dense_bias.addr},
                                            {"transformer.h.23.self_attention.query_key_value.weight", qkv_weight.addr},
                                            {"transformer.h.23.self_attention.query_key_value.bias", qkv_bias.addr}};

  std::map<std::string, uint32_t> forward_map{{"transformer.h.23.self_attention.dense_in", dense_forward_in.addr},
                                              {"transformer.h.23.self_attention.attention_probs_reshaped", bmm_in_1.addr},
                                              {"transformer.h.23.self_attention.value_layer", bmm_in_2.addr},
                                              {"transformer.h.23.self_attention.softmax_out", softmax_forward_out.addr},
                                              {"transformer.h.23.self_attention.query_layer", baddbmm_in_1.addr},
                                              {"transformer.h.23.self_attention.key_layer", baddbmm_in_2.addr},
                                              {"transformer.h.23.self_attention.query_key_value_in", qkv_forward_in.addr}};


  data<3> result = BLOOMAttentionBackward(inst2, config, backward_input, "transformer.h.23.self_attention", weight_map, forward_map, backward_input.addr + backward_input.size());

  HBM_TO_VMEM(instruction_list, 1313536, 0, 1024);
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends
  AddNoop(10, instruction_list);

  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t *data = new uint32_t[41 * 1024 + 1024 * 1024 + 1024 + 41 * 1024 + 16 * 41 * 41 + 16 * 41 * 64 + 16 * 41 * 41 + 16 * 41 * 64 + 16 * 64 * 41 + 3072 * 1024 + 3072 + 41 * 1024];
  float *input = new float[41 * 1024];
  float *denseweight = new float[1024 * 1024];
  float *densebias = new float[1024];
  float *denseforwardin = new float[41 * 1024];
  float *bmmin1 = new float[16 * 41 * 41];
  float *bmmin2 = new float[16 * 41 * 64];
  float *softmaxforwardout = new float[16 * 41 * 41];
  float *baddbmmin1 = new float[16 * 41 * 64];
  float *baddbmmin2 = new float[16 * 64 * 41];
  float *qkvweight = new float[3072 * 1024];
  float *qkvbias = new float[3072];
  float *qkvforwardin = new float[41 * 1024];
  uint32_t index = 0;

  load_input(input, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23.self_attention_in.txt", 41 * 1024);
  load_input(denseweight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.dense.weight.txt", 1024 * 1024);
  load_input(densebias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.dense.bias.txt", 1024);
  load_input(denseforwardin, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.dense_in.txt", 41 * 1024);
  load_input(bmmin1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.attention_probs_reshaped.txt", 16 * 41 * 41);
  load_input(bmmin2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.value_layer.txt", 16 * 64 * 41);
  load_input(softmaxforwardout, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.softmax_out.txt", 16 * 41 * 41);
  load_input(baddbmmin1, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.query_layer.txt", 16 * 41 * 64);
  load_input(baddbmmin2, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.key_layer.txt", 16 * 64 * 41);
  load_input(qkvweight, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.query_key_value.weight.txt", 3072 * 1024);
  load_input(qkvbias, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h.23.self_attention.query_key_value.bias.txt", 3072);
  load_input(qkvforwardin, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f32/transformer.h.23.self_attention.query_key_value_in.txt", 41 * 1024);

  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&input[i]);
  }
  for (uint32_t i = 0; i < 1024 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&denseweight[i]);
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&densebias[i]);
  }
  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&denseforwardin[i]);
  }
  for (uint32_t i = 0; i < 16 * 41 * 41; i++, index++)
  {
    data[index] = *(uint32_t*)(&bmmin1[i]);
  }
  index  = ((index + 128) / 128) * 128;
  for (uint32_t i = 0; i < 16 * 41 * 64; i++, index++)
  {
    data[index] = *(uint32_t*)(&bmmin2[i]);
  }
  for (uint32_t i = 0; i < 16 * 41 *41; i++, index++)
  {
    data[index] = *(uint32_t*)(&softmaxforwardout[i]);
  }
  index  = ((index + 128) / 128) * 128;
  for (uint32_t i = 0; i < 16 * 41 * 64; i++, index++)
  {
    data[index] = *(uint32_t*)(&baddbmmin1[i]);
  }
  for (uint32_t i = 0; i < 16 * 41 * 64; i++, index++)
  {
    data[index] = *(uint32_t*)(&baddbmmin2[i]);
  }
  for (uint32_t i = 0; i < 3072 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&qkvweight[i]);
  }
  for (uint32_t i = 0; i < 3072; i++, index++)
  {
    data[index] = *(uint32_t*)(&qkvbias[i]);
  }
  for (uint32_t i = 0; i< 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&qkvforwardin[i]);
  }
  
  simulator.WriteToHBM(data, index  + 1, 0);


  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);
  std::cout << "Instruction size: " << instruction_list.size() << std::endl;
  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }

  simulator.DebugPrintVmem(0, 1024);
  // float* test = new float[41 * 1024];
  // load_input(test, "simple_test/bloom_560m_backward_f32/transformer.h.0.self_attention_out.txt", 41 * 1024);
  // simulator.DebugPrintVmem_tensor(result.addr, result.addr + result.size(), test);
  std::cout << "Execute end\n";
  return ;
} 

void MatMulDxBackwardTest()
{
  std::cout << "MatMulDxBackwardTest\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 41 * 1024 * 2);
  data<3> hidden_states(0, {16, 41, 64});
  data<3> weight_addr(hidden_states.addr + hidden_states.size(), {16, 41, 64});   
  BLOOMConfig config;

  std::cout << "func 1\n";
  HBMAddr() = 0;
  data<4> result = MatMulDxBackward(inst2, weight_addr.as<4>(), hidden_states.as<4>(), weight_addr.addr + weight_addr.size());
  std::cout << "func 2\n";

  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[41 * 1024 * 2];
  float* input = new float[41 * 1024];
  float* weights = new float[1024 * 41];
  uint32_t index = 0;

  load_input(input, "/home/yinxun/workspace/dlc_simulator/src/simu/transformer.h.23.self_attention.bloombmm_in.txt", 41*1024);
  load_input(weights, "/home/yinxun/workspace/dlc_simulator/src/simu/transformer.h.23.self_attention.value_layer.txt", 1024 * 41);

  for(uint32_t i = 0; i < 41*1024; i++, index++) {
    data[index] = *(uint32_t*)(&(input[i]));
  }

  for(uint32_t i = 0; i < 1024 * 41; i++) {
    data[index] = *(uint32_t*)(&(weights[i]));
    index++;
  }

  simulator.WriteToHBM(data, 41 * 1024 * 2, 0);

  std::cout << "old instruct.size: " << instruction_list.size() << std::endl;

  // instruction_list = schedule(instruction_list);

  // std::cout << "new instruct.size: " << instruction_list.size() << std::endl;
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_index: " << i << " inst_vec_size: " << inst_vec.size() << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }
  
  std::cout << "Execute end\n";
  // simulator.PrintHBM(0, 13 * 50304);
  // simulator.DebugPrintVmem(result.addr , result.addr + result.size());
  // std::cout << "result.dims: " << result.dims.size() << std::endl;
  // for(uint32_t i = 0; i <  result.dims.size(); i++) {
  //   std::cout << "result.dim[" << i << "]: " << result.dims[i] << std::endl;
  // }
  float* test = new float[41*16 * 41];
  load_input(test, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.23.self_attention.bloombmm_out.txt", 41 * 41 * 16);

  simulator.DebugPrintVmem_tensor(result.addr, result.addr+result.size(), test);
  
  return;
}

void TrainingTest()
{
  std::cout << "TrainingTest\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================
  HBM_TO_VMEM(instruction_list, 0, 0, 115 * 1024);
  data<4> input_ids(0, {1, 1, 115, 1024});
  data<2> lm_head_weight(input_ids.size() + input_ids.addr, {250880, 1024});

  std::map<std::string, uint32_t> weights_addr{{"transformer.lm_head.weight", lm_head_weight.addr}};
  BLOOMConfig config;

  HBMAddr() = 0;

  data<2> lm_logit_weights;
  lm_logit_weights.hbmaddr = weights_addr.at("transformer.lm_head.weight");
  lm_logit_weights.dims = {250880, 1024};

  int weightaddr = lm_head_weight.addr + lm_head_weight.size();
  std::cout << "lm_head: " << lm_logit_weights.hbmaddr << std::endl;
  int usehbm = weightaddr;
  int weights_use_vmem_size = (kVectorDataMemorySize - input_ids.addr - input_ids.size());
  // uint32_t weight_use_row = (weights_use_vmem_size / input_ids.dims[2]) / 128 * 128;
  int weight_use_row = (weights_use_vmem_size / (input_ids.dims[2] + lm_logit_weights.dims[1])) / 128 * 128;
  std::cout << "weight_use_row: " << weight_use_row << std::endl;

  int SplitNum = lm_logit_weights.dims[0] / weight_use_row;
  std::cout << "SplitNum: " << SplitNum << std::endl;
  for (int i = 0; i <= SplitNum; i++)
  {
    int weight_row_now = (lm_logit_weights.dims[0] - i * weight_use_row) >= weight_use_row ? weight_use_row : (lm_logit_weights.dims[0] - i * weight_use_row);
    std::cout << "weight_row_now: " << weight_row_now << std::endl;
    data<2> now_weight;
    now_weight.hbmaddr = lm_logit_weights.hbmaddr + i * weight_use_row * lm_logit_weights.dims[1];
    std::cout << "now_weight hbmaddr: " << now_weight.hbmaddr << std::endl;
    now_weight.dims = {(uint32_t)weight_row_now, lm_logit_weights.dims[1]};

    data<3> lm_head_output = linearNobias(inst2, input_ids[0], now_weight, input_ids.addr + input_ids.size());
    std::cout << "lm_head_output: " << lm_head_output.dims[0] << ' ' << lm_head_output.dims[1] << ' ' << lm_head_output.dims[2] << std::endl;

    int vmemaddr = lm_head_output.addr;
    int outputhbm = usehbm;
    for (int i = 0; i < lm_head_output.dims[1]; i++)
    {
      VMEM_TO_HBM(instruction_list, vmemaddr, outputhbm, lm_head_output.dims[2]);
      vmemaddr += lm_head_output.dims[2];
      outputhbm += lm_logit_weights.dims[0];
    }
    usehbm += lm_head_output.dims[2];
  }

  int num = lm_logit_weights.dims[0] / 1024;
  std::cout << "nums: " << num << std::endl;

  int vmemaddr = input_ids.addr + input_ids.size();
  usehbm = weightaddr + (input_ids.dims[1] - 1) * lm_logit_weights.dims[0];
  for (int i = 0; i < num; i++)
  {
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(vmemaddr / 32).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(vmemaddr / 32).first);
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    vmemaddr += 1024;
  }
  VMEM_TO_HBM(instruction_list, input_ids.addr + input_ids.size(), usehbm, lm_logit_weights.dims[0]);

  // std::vector<uint32_t> label(115, 0);
  vec1d_t_i labels{-100, -100, -100, -100, -100, -100, -100, -100, -100, 
                  -100, -100, -100, -100, -100, -100, -100, -100, -100, 
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, 1, 226041, 355, 8432, 19540, 
                  23124, 40651, 355, 842, 14194, 20451, 59280, 55675, 224909, 
                  60266, 420, 7436, 12142, 84793, 20451, 59280, 60266, 355, 
                  12703, 5136, 8401, 2079, 54682, 3616, 19651, 420, 12142, 
                  25477, 4990, 79267, 14554, 12142, 20451, 60266, 355, 58693, 
                  13344, 23107, 55675, 224909, 86689, 420, 2};
  int zero_num = 0;
  for (int i = 0; i < labels.size() - 1; i++)
  {
    labels[i] = labels[i + 1];
    if (labels[i] <= 0) zero_num++;
  }
  labels[labels.size() - 1] = 0; 
  std::cout << "zero_num: " << zero_num << std::endl;
  weights_use_vmem_size = (kVectorDataMemorySize - input_ids.addr - input_ids.size()) / 2;
  weight_use_row = weights_use_vmem_size / lm_logit_weights.dims[0];

  int weight_row = (kVectorDataMemorySize - input_ids.addr - input_ids.size()) / lm_logit_weights.dims[0];
  SplitNum = zero_num / weight_row;

  for (int i = 0; i <= SplitNum; i++)
  {
    int weight_row_now = (zero_num - i * weight_row) >= weight_row ? weight_row : (zero_num - i * weight_row);
    std::cout << "zero row_now: " << weight_row_now << std::endl;
    if (1)
    {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    int zerosplit = weight_row_now * lm_logit_weights.dims[0] / 1024;
    uint32_t zerohbmaddr = weightaddr + i * weight_row * lm_logit_weights.dims[0];
    for (int j = 0; j < zerosplit; j++)
    {
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((input_ids.addr + input_ids.size() + j * 1024) / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((input_ids.addr + input_ids.size() + j * 1024) / 32).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
    VMEM_TO_HBM(instruction_list, input_ids.addr + input_ids.size(), zerohbmaddr, weight_row_now * lm_logit_weights.dims[0]);
  }

  int testnum = input_ids.dims[2] - zero_num - 1;
  int testaddr = weightaddr + zero_num * lm_logit_weights.dims[0];
  SplitNum = testnum / weight_use_row;
  std::cout << "testaddr: " << testaddr << std::endl;
  std::cout << "testnum: " << testnum << std::endl;
  std::cout << "2 SplitNum: " << SplitNum << std::endl;
  int l = zero_num;
  for (int i = 0; i <= SplitNum; i++)
  {
    int weight_row_now = (testnum - i * weight_use_row) >= weight_use_row ? weight_use_row : (testnum - i * weight_use_row);
    if (weight_row_now == 0) break;
    data<3> soft(input_ids.addr + input_ids.size(), {1, weight_row_now, lm_logit_weights.dims[0]});
    HBM_TO_VMEM(instruction_list, testaddr + i * weight_use_row * lm_logit_weights.dims[0], soft.addr, lm_logit_weights.dims[0] * weight_row_now);
    data<3> leftsoft(soft.addr + soft.size(), {soft.dims[0], soft.dims[1], soft.dims[2]});
    Softmax(instruction_list, soft.addr, leftsoft.addr, soft.dims[0] * soft.dims[1], soft.dims[2]);
    std::cout << i << " soft: " << soft.addr << std::endl;
    std::cout << i << " leftsoft: " << leftsoft.addr << std::endl;
    for (int j = 0; j < weight_row_now; j++)
    {
      int batch1 = labels[l] / 1024;
      int batch2 = labels[l] % 1024;
      std::cout << "label " << l << ": " << labels[l] << std::endl;
      std::cout << "batch1: " << batch1 << std::endl;
      std::cout << "batch2: " << batch2 << std::endl;
      int softvmem = leftsoft.addr + j * lm_logit_weights.dims[0] + batch1 * 1024;
      std::cout << "softvmem: " << softvmem << std::endl;
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue((uint32_t)batch2).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue((uint32_t)batch2).first);
        VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
        VectorOperationState vmask0(V_S32_EQUAL, 0, 0, 44, 0);
        inst->SetOperationState(Instruction::VECTORTWO, &vmask0);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(softvmem / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(softvmem / 32).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      if (1)
      {
        inst = new Instruction();
        VectorOperationState sub(V_F32_SUBTRACTION, 0, 0, 49, 1);
        inst->SetOperationState(Instruction::VECTORTWO, &sub);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      if (1)
      {
        inst = new Instruction();
        VectorOperationState vmask0(V_SELECT_VMASK0, 0, 0, 1, 0);
        inst->SetOperationState(Instruction::VECTORONE, &vmask0);
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(softvmem / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(softvmem / 32).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      l++;
    }
    float div = 1.0 / (float(testnum));
    std::cout << "div: " << div << std::endl;
    Division(inst2, leftsoft, div);
    VMEM_TO_HBM(instruction_list, leftsoft.addr, testaddr + i * weight_use_row * lm_logit_weights.dims[0], lm_logit_weights.dims[0] * weight_row_now);
  }

  if (1)
  {
    inst = new Instruction();
    VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  int zerohbmaddr = weightaddr + (labels.size() - 1) * lm_logit_weights.dims[0];
  int zerosplit = lm_logit_weights.dims[0] / 1024;
  for (int j = 0; j < zerosplit; j++)
  {
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((input_ids.addr + input_ids.size() + j * 1024) / 32).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((input_ids.addr + input_ids.size() + j * 1024) / 32).first);
      inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      inst->SetOperationState(Instruction::SCALARONE, &set_base);
      inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
      inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  VMEM_TO_HBM(instruction_list, input_ids.addr + input_ids.size(), zerohbmaddr, lm_logit_weights.dims[0]);


  HBM_TO_VMEM(instruction_list, testaddr - 1024, 0, 251904);
  // data<4> softmax_input;
  // softmax_input.hbmaddr = input.addr;
  // softmax_input.dims = {1, input_ids.dims[1], input_ids.dims[2], lm_logit_weights.dims[0]};
  
  // data<4> backward_input = matmulIbWb(inst2, softmax_input, lm_logit_weights, 0);
  // inst2.Spy("transformer.lm_head_out", backward_input.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/lm_head_out.txt");
  // // data<4> backward_output = BLOOMModelBackward(inst2, config, backward_input, backward_input.addr + backward_input.size(), backward_input.addr + backward_input.size(), "transformer", weights_addr, forward_addr);

  // HBM_TO_VMEM(instruction_list, 65 * 1024, 0, 2048);
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);


  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* data = new uint32_t[weightaddr];
  float* input1 = new float[115*1024];
  float* lm_weight = new float[250880 * 1024];
  // float* input2 = new float[250880 * 115];

  uint32_t index = 0;
  load_input(input1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.ln_f_out.txt", 115 * 1024);
  load_input(lm_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.lm_head.weight.txt", 1024 * 250880);
  // load_input(input2, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/lm_head_in.txt", 115 * 250880);
  for (uint32_t i = 0; i < 1024 * 115; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input1[i]));
  }
  for(uint32_t i = 0; i < 250880 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(lm_weight[i]));
  }
  // for(uint32_t i = 0; i < 250880 * 115; i++, index++)
  // {
  //   data[index] = *(uint32_t*)(&(input2[i]));
  // }

  std::cout << "index: " << index << std::endl;
  simulator.WriteToHBM(data, index, 0);

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);

  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
    auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);
     

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(0x7fffffff);

    // for (auto it = range.first; it != range.second; it++)
    // {
    //   float* test = new float[it->second.len];
    //   std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
    //   load_input(test, it->second.compare_file, *(int*)(&it->second.len));
    //   simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
    //   // if ((it->second.name == "transformer.h.0.mlp_in") || (it->second.name == "transformer.h.0.mlp_out")
    //   //  || (it->second.name == "transformer.h.0.self_attention_in") || (it->second.name == "transformer.h.0.self_attention_out")
    //   //  || (it->second.name == "transformer.h.0_out"))
    //   // {
    //   //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
    //   // }
    //   // if(it->second.name == "transformer.h.23.self_attention.bloombmm_out")
    //   // {
    //   //   simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, it->second.name);
    //   // }
    // }
  }
  simulator.DebugPrintVmem(0,  251904);
  std::cout << "Execute end\n";
  return ;
}

void SchedulerTest()
{
  std::cout << "SchedulerTest\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction* inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================

  HBMAddr() = 0;
  int i;

  // for(i = 0; i < 7; i++) {
  //   inst = new Instruction();
  //   inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
  //   inst->SetImmediateValue(Instruction::IMMEDIATE0, (7 - i) * 128 * 8 / 32);
  //   inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
  //   ScalarOperationState move(S_U32_MOVE, 0, 0, 32, 0);
  //   inst->SetOperationState(Instruction::SCALARONE, &move);
  //   VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, (7 - i), 1, 2, 0, 0, 0);
  //   inst->SetOperationState(Instruction::VECTORLOAD, &vload);
  //   MTIOperationState mti(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, (7 - i), 0, 0);
  //   inst->SetOperationState(Instruction::MTI, &mti);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }
  
  // if(1) {
  //   inst = new Instruction();
  //   inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
  //   inst->SetImmediateValue(Instruction::IMMEDIATE0, 8 * 128 * 8 / 32);
  //   inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
  //   ScalarOperationState move(S_U32_MOVE, 0, 0, 32, 0);
  //   inst->SetOperationState(Instruction::SCALARONE, &move);
  //   VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 8, 1, 2, 0, 0, 0);
  //   inst->SetOperationState(Instruction::VECTORLOAD, &vload);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }

  // if(1) {
  //   inst = new Instruction();
  //   //放到GMR中时，我们需要先假乘一下，随便那个vector_register都可以
  //   MTIOperationState mti_gmr(MTI_MUL_GSTF_ROUNDED, 0, 9, 0, 0);
  //   inst->SetOperationState(Instruction::MTI, &mti_gmr);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }
  
  // if(1) {
  //   inst = new Instruction();
  //   MTROperationState mtr(MTR_READ_MATRIX_RESULT, 0, 10, 0);
  //   inst->SetOperationState(Instruction::MTR, &mtr);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }

  // //GMR中与16号vector_register(卷积核)矩阵相乘
  // if(1) {
  //   inst = new Instruction();
  //   MTIOperationState mti_mul(MTI_MUL_FLOAT_ROUNDED, 0, 8, 0, 0);
  //   inst->SetOperationState(Instruction::MTI, &mti_mul);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }

  // if(1) {
  //   inst = new Instruction();
  //   MTROperationState mtr(MTR_READ_MATRIX_RESULT, 0, 11, 0);
  //   inst->SetOperationState(Instruction::MTR, &mtr);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }




  // for(i = 0; i < 8; i++) {
  //   inst = new Instruction();
  //   inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
  //   inst->SetImmediateValue(Instruction::IMMEDIATE0, (32 - i) * 128 * 8 / 32);
  //   inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
  //   ScalarOperationState move(S_U32_MOVE, 0, 0, 32, 0);
  //   inst->SetOperationState(Instruction::SCALARONE, &move);
  //   VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, (31 - i), 1, 2, 0, 0, 0);
  //   inst->SetOperationState(Instruction::VECTORLOAD, &vload);
  //   MTIOperationState mti(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, (31 - i), 0, 0);
  //   inst->SetOperationState(Instruction::MTI, &mti);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }
  
  // if(1) {
  //   inst = new Instruction();
  //   inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
  //   inst->SetImmediateValue(Instruction::IMMEDIATE0, 33 * 128 * 8 / 32);
  //   inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
  //   ScalarOperationState move(S_U32_MOVE, 0, 0, 32, 0);
  //   inst->SetOperationState(Instruction::SCALARONE, &move);
  //   VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 20, 1, 2, 0, 0, 0);
  //   inst->SetOperationState(Instruction::VECTORLOAD, &vload);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }

  // if(1) {
  //   inst = new Instruction();
  //   //放到GMR中时，我们需要先假乘一下，随便那个vector_register都可以
  //   MTIOperationState mti_gmr(MTI_MUL_GSTF_ROUNDED, 0, 19, 0, 0);
  //   inst->SetOperationState(Instruction::MTI, &mti_gmr);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }
  
  // if(1) {
  //   inst = new Instruction();
  //   MTROperationState mtr(MTR_READ_MATRIX_RESULT, 0, 18, 0);
  //   inst->SetOperationState(Instruction::MTR, &mtr);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }

  // //GMR中与16号vector_register(卷积核)矩阵相乘
  // if(1) {
  //   inst = new Instruction();
  //   MTIOperationState mti_mul(MTI_MUL_FLOAT_ROUNDED, 0, 20, 0, 0);
  //   inst->SetOperationState(Instruction::MTI, &mti_mul);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }

  // if(1) {
  //   inst = new Instruction();
  //   MTROperationState mtr(MTR_READ_MATRIX_RESULT, 0, 17, 0);
  //   inst->SetOperationState(Instruction::MTR, &mtr);
  //   CompleteInstruction(inst);
  //   instruction_list.push_back(inst);
  // }

  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 1);
    VectorOperationState move(V_U32_MOVE, 0, 0, 32, 1);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    //transpose start with width of 8.
    MTIOperationState transpose_start(MTI_TRANSPOSE_START, 0, 1, 0, 0);
    inst->SetOperationState(Instruction::MTI, &transpose_start);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  for(int i = 2; i < 16; i++)
  {
    if(1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, i);
      VectorOperationState move(V_U32_MOVE, 0, 0, 32, i);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      //transpose start with width of 8.
      MTIOperationState transpose(MTI_TRANSPOSE, 0, i, 0, 0);
      inst->SetOperationState(Instruction::MTI, &transpose);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }


  if (1)
  {
    Instruction* inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 16);
    VectorOperationState move(V_U32_MOVE, 0, 0, 32, 16);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    MTIOperationState transpose_end(MTI_TRANSPOSE_END, 0, 16, 0, 0);
    inst->SetOperationState(Instruction::MTI, &transpose_end);
    CompleteInstruction(inst);
    instruction_list.push_back(inst); 
  }

  for(int i=0; i<16; i++){
    inst = new Instruction();
    MTROperationState read_trans(MTR_READ_TRANSPOSE_RESULT, 0, i, 0);
    inst->SetOperationState(Instruction::MTR, &read_trans);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 17);
    VectorOperationState move(V_U32_MOVE, 0, 0, 32, 17);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    //transpose start with width of 8.
    MTIOperationState transpose_start(MTI_TRANSPOSE_START, 0, 17, 0, 0);
    inst->SetOperationState(Instruction::MTI, &transpose_start);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  for(int i = 1; i < 6; i++)
  {
    if(1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, 17 + i);
      VectorOperationState move(V_U32_MOVE, 0, 0, 32, 17 + i);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      //transpose start with width of 8.
      MTIOperationState transpose(MTI_TRANSPOSE, 0, 17 + i, 0, 0);
      inst->SetOperationState(Instruction::MTI, &transpose);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }

  if (1)
  {
    Instruction* inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 23);
    VectorOperationState move(V_U32_MOVE, 0, 0, 32, 23);
    inst->SetOperationState(Instruction::VECTORONE, &move);
    MTIOperationState transpose_end(MTI_TRANSPOSE_END, 0, 23, 0, 0);
    inst->SetOperationState(Instruction::MTI, &transpose_end);
    CompleteInstruction(inst);
    instruction_list.push_back(inst); 
  }

  for(int i=0; i<6; i++){
    inst = new Instruction();
    MTROperationState read_trans(MTR_READ_TRANSPOSE_RESULT, 0, i, 0);
    inst->SetOperationState(Instruction::MTR, &read_trans);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  instruction_list = schedule(instruction_list, true);
  // Halt
  inst = new Instruction();
  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  inst->SetOperationState(Instruction::SCALARONE, &scalar);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);
  /////////////////////////////////////////////////////////////////////////////////////////
  // Test Ends

  AddNoop(10, instruction_list);

  // Creating simulator Object and start running.
  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t *data = new uint32_t[41 * 1024];
  float *input = new float[41 * 1024];
  uint32_t index = 0;

  load_input(input, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/transformer.h.0_in.txt", 41 * 1024);

  for (uint32_t i = 0; i < 41 * 1024; i++, index++)
  {
    data[index] = *(uint32_t*)(&(input[i]));
  }
  // simulator.WriteToHBM(data, index, 0);
  char* cn_test = (char*) data;
  simulator.WriteToVmemWithOffset(cn_test, 41 * 1024 / 128, 0);
  
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);
  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);


    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }

  std::cout << "Execute end\n";
  // simulator.DebugPrintVmem(0, 3072);
  // simulator.PrintHBM(0, 1024);
  return ;
}

void ttTest()
{
  std::cout << "TT Test" << std::endl;
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  HBMAddr() = 0;
  HBM_TO_VMEM(instruction_list, 0, 0, 115 * 1024);
  data<4> hidden_state(0, {1, 1, 115, 1024});
  data<2> attention_mask(hidden_state.addr + hidden_state.size(), {1, 115});
  data<2> word_embeddings_weight(hidden_state.addr + hidden_state.size(), {250880, 1024});
  data<1> word_embeddings_layernorm_weight(word_embeddings_weight.addr + word_embeddings_weight.size(), {1024});
  data<1> word_embeddings_layernorm_bias(word_embeddings_layernorm_weight.addr + word_embeddings_layernorm_weight.size(), {1024});
  data<1> ln_f_weight(word_embeddings_layernorm_bias.addr + word_embeddings_layernorm_bias.size(), {1024});
  data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {1024});
  data<2> lm_logit_weights(ln_f_bias.addr + ln_f_bias.size(), {250880, 1024});
  
  std::map<std::string, uint32_t> weights_addr{{"transformer.word_embeddings.weight", word_embeddings_weight.addr},
                                              {"transformer.word_embeddings_layernorm.weight", word_embeddings_layernorm_weight.addr},
                                              {"transformer.word_embeddings_layernorm.bias", word_embeddings_layernorm_bias.addr},
                                              {"transformer.ln_f.weight", ln_f_weight.addr},
                                              {"transformer.ln_f.bias", ln_f_bias.addr},
                                              {"transformer.lm_head.weight", lm_logit_weights.addr}};
  uint32_t weightaddr = lm_logit_weights.addr + lm_logit_weights.size();
  for (int i = 23; i < 24; i++)
  {
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.weight", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.bias", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight", weightaddr));
    weightaddr += 1024 * 3072;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias", weightaddr));
    weightaddr += 3072;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.weight", weightaddr));
    weightaddr += 1024 * 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.bias", weightaddr));
    weightaddr += 1024;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight", weightaddr));
    weightaddr += 1024 * 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias", weightaddr));
    weightaddr += 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight", weightaddr));
    weightaddr += 1024 * 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias", weightaddr));
    weightaddr += 1024;
  }

  std::map<std::string, uint32_t> forward_addr{{"transformer.ln_f_in", weightaddr}};
  weightaddr += 115 * 1024;
  for (int i = 23; i >= 23; i--)
  {
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h_in", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.gelu_impl_in", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h_in", weightaddr));
    weightaddr += 115 * 1024;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm_in", weightaddr));
    weightaddr += 115 * 1024;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense_in", weightaddr));
    weightaddr += 115 * 1024;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.attention_probs_reshaped", weightaddr));
    weightaddr += 16 * 115 * 115;
    weightaddr = ((weightaddr + 128) / 128) * 128;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.value_layer", weightaddr));
    weightaddr += 115 * 1024;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.softmax_out", weightaddr));
    weightaddr += 16 * 115 * 115;
    weightaddr = ((weightaddr + 128) / 128) * 128;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_layer", weightaddr));
    weightaddr += 115 * 1024;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.key_layer", weightaddr));
    weightaddr += 115 * 1024;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value_in", weightaddr));
    weightaddr += 115 * 1024;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm_in", weightaddr));
    weightaddr += 115 * 1024;
  }

  BLOOMConfig config;

  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* datas = new uint32_t[weightaddr + 115 * 250880];
  float* input1 = new float[115*1024]; 
  float* emb_weight = new float[250880 * 1024];
  float* wel_weight = new float[1024];
  float* wel_bias = new float[1024];
  float* lf_weight = new float[1024];
  float* lf_bias = new float[1024];
  float* lm_weight = new float[250880 * 1024];
  float* lm_input = new float[250880 * 115];
  uint32_t index = 0;
  load_input(input1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.h.23_in.txt", 115 * 1024);
  load_input(emb_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.word_embeddings.weight.txt", 1024 * 250880);
  load_input(wel_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.word_embeddings_layernorm.weight.txt", 1024);
  load_input(wel_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.word_embeddings_layernorm.bias.txt", 1024);
  load_input(lf_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.ln_f.weight.txt", 1024);
  load_input(lf_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.ln_f.bias.txt", 1024);
  load_input(lm_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.lm_head.weight.txt", 1024 * 250880);
  load_input(lm_input, "/home/yinxun/dlc_simulator_test/src/simu/output0.txt", 115 * 250880);
  for (uint32_t i = 0; i < 115 * 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(input1[i]));
  }
  for(uint32_t i = 0; i < 250880 * 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(emb_weight[i]));
  }
  for(uint32_t i = 0; i < 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(wel_weight[i]));
  }
  for(uint32_t i = 0; i < 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(wel_bias[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lf_weight[i]));
  }
  for (uint32_t i = 0; i < 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lf_bias[i]));
  }
  for(uint32_t i = 0; i < 250880 * 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lm_weight[i]));
  }

  for (int i = 23; i <= 23; i++)
  {
    float *matrix1 = new float[1024];
    load_input(matrix1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".input_layernorm.weight.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix1[j]));
    float *matrix2 = new float[1024];
    load_input(matrix2, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".input_layernorm.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix2[j]));
    float *matrix7 = new float[1024];
    load_input(matrix7, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix7[j]));
    float *matrix8 = new float[1024];
    load_input(matrix8, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix8[j]));
    float *matrix3 = new float[3 * 1024 * 1024];
    load_input(matrix3, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight.txt", 3 * 1024 * 1024);
    for (uint32_t j = 0; j < 3 * 1024 * 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix3[j]));
    float *matrix4 = new float[3 * 1024];
    load_input(matrix4, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias.txt", 3 * 1024);
    for (uint32_t j = 0; j < 3 * 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix4[j]));
    float *matrix5  = new float[1024 * 1024];
    load_input(matrix5, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.weight.txt", 1024 * 1024);
    for (uint32_t j = 0; j < 1024 * 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix5[j]));
    float *matrix6 = new float[1024];
    load_input(matrix6, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix6[j]));
    float *matrix9 = new float[1024 * 4096];
    load_input(matrix9, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight.txt", 1024 * 4096);
    for (uint32_t j = 0; j < 1024 * 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix9[j]));
    float *matrix10 = new float[4096];
    load_input(matrix10, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix10[j]));
    float *matrix11 = new float[1024 * 4096];
    load_input(matrix11, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight.txt", 1024 * 4096);
    for (uint32_t j = 0; j < 1024 * 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix11[j]));
    float *matrix12 = new float[1024];
    load_input(matrix12, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias.txt", 1024);
    for (uint32_t j = 0; j < 1024; j++, index++) datas[index] = *(uint32_t*)(&(matrix12[j]));
  } 
  for(uint32_t i = 0; i < 250880 * 115; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lm_input[i]));
  }

  std::cout << "index: " << index << std::endl;
  simulator.WriteToHBM(datas, index, 0);

  uint32_t* amask = new uint32_t[128];
  for(int i = 0; i < 115; i++) amask[i] = 1;
  simulator.WriteToVmemWithOffset((char*)amask, 128 / 128, attention_mask.addr);

  uint32_t *alibiu = new uint32_t[1920];
  float *alibif = new float[16 * 115];
  load_input(alibif, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/alibi.txt", 16 * 115);
  for(uint32_t i = 0; i < 16 * 115; i++) alibiu[i] = *(uint32_t*)(&(alibif[i]));
  std::cout << "=====" << std::endl;
  simulator.WriteToVmemWithOffset((char*)alibiu, 1920 / 128, AlignTo128Bytes(attention_mask.addr + attention_mask.size()));

  std::tuple<uint32_t, uint32_t> input_shape(attention_mask.dims[0], attention_mask.dims[1]);
  data<3> alibi(attention_mask.addr + AlignTo128Bytes(attention_mask.size()), {config.n_head * attention_mask.dims[0], 1, attention_mask.dims[1]});
  // build_alibi(inst2, attention_mask, config.n_head, alibi.addr);
  std::cout << "alibi.addr: " << alibi.addr << std::endl;
  std::cout << "alibi: " << alibi.dims[0] << " " << alibi.dims[1] << " " << alibi.dims[2] << std::endl;
  // inst2.Spy("baddbmm_in", alibi.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.h.23.self_attention.baddbmm_in.txt");
  data<4> causal_mask = _prepare_attn_mask(inst2, attention_mask, input_shape, 0, alibi.addr + AlignTo128Bytes(alibi.size()));
  std::cout << "causal_mask: " << causal_mask.dims[0] << " " << causal_mask.dims[1] << " " << causal_mask.dims[2] << " " << causal_mask.dims[3] << std::endl;
  // inst2.Spy("causal_mask", causal_mask.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.h.23.self_attention.baddbmm_in.txt");
  uint32_t Block_addr = causal_mask.addr + AlignTo128Bytes(causal_mask.size());
  std::cout << "block_addr: " << Block_addr << std::endl;
  
  data<3> result(Block_addr, {1, 115, 1024});

  // int numz = 15;

  // for (int z = 0; z < numz; z++)
  // {
  //   hidden_state.addr = 0;
  //   hidden_state.dims = {1, 1, 115, 1024};
    
  //   data<3> result = BLOOMBlock(inst2, config, hidden_state, alibi.as<4>(), causal_mask, "transformer.h.23", weights_addr, forward_addr, Block_addr);

  //   data<1> lnf_weight(result.addr + result.size(), {config.hidden_size});
  //   data<1> lnf_bias(lnf_weight.addr + lnf_weight.size(), {config.hidden_size});
  //   HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.weight"), lnf_weight.addr, lnf_weight.size());
  //   HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.bias"), lnf_bias.addr, lnf_bias.size());
    
  //   if (config.training)
  //   {
  //     std::string name = "transformer.ln_f_in";
  //     if (forward_addr.find(name) == forward_addr.end()) std::cout << "Not forwardMap: " << name << std::endl;
  //     VMEM_TO_HBM(instruction_list, result.addr, forward_addr.at(name), result.size());
  //   }
  //   inst2.Spy("transformer.ln_f_in", result.asVMem(inst2), "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.ln_f_in.txt");
  //   result = LayerNorm(inst2, result, lnf_weight, lnf_bias, lnf_bias.addr + lnf_bias.size(), config.layer_norm_epsilon);
  //   inst2.Spy("transformer.ln_f_out", result.asVMem(inst2), "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.ln_f_out.txt");

    uint32_t usehbm = weightaddr;
    uint32_t weights_use_vmem_size = (kVectorDataMemorySize - result.addr - result.size());
  //   // uint32_t weight_use_row = (weights_use_vmem_size / input_ids.dims[2]) / 128 * 128;
    int weight_use_row = (weights_use_vmem_size / (result.dims[1] + lm_logit_weights.dims[1])) / 128 * 128;
  //   std::cout << "weight_use_row: " << weight_use_row << std::endl;

    int SplitNum = lm_logit_weights.dims[0] / weight_use_row;
  //   std::cout << "SplitNum: " << SplitNum << std::endl;
  //   for (int i = 0; i <= SplitNum; i++)
  //   {
  //     int weight_row_now = (lm_logit_weights.dims[0] - i * weight_use_row) >= weight_use_row ? weight_use_row : (lm_logit_weights.dims[0] - i * weight_use_row);
  //     std::cout << "weight_row_now: " << weight_row_now << std::endl;
  //     data<2> now_weight;
  //     now_weight.hbmaddr = lm_logit_weights.addr + i * weight_use_row * lm_logit_weights.dims[1];
  //     std::cout << "now_weight hbmaddr: " << now_weight.hbmaddr << std::endl;
  //     now_weight.dims = {(uint32_t)weight_row_now, lm_logit_weights.dims[1]};

  //     data<3> lm_head_output = linearNobias(inst2, result, now_weight, result.addr + result.size());
  //     std::cout << "lm_head_output: " << lm_head_output.dims[0] << ' ' << lm_head_output.dims[1] << ' ' << lm_head_output.dims[2] << std::endl;

  //     uint32_t vmemaddr = lm_head_output.addr;
  //     uint32_t outputhbm = usehbm;
  //     for (int i = 0; i < lm_head_output.dims[1]; i++)
  //     {
  //       VMEM_TO_HBM(instruction_list, vmemaddr, outputhbm, lm_head_output.dims[2]);
  //       vmemaddr += lm_head_output.dims[2];
  //       outputhbm += lm_logit_weights.dims[0];
  //     }
  //     usehbm += lm_head_output.dims[2];
  //   }
  //   // Halt
  //   if (1)
  //   {
  //     inst = new Instruction();
  //     ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  //     inst->SetOperationState(Instruction::SCALARONE, &scalar);
  //     CompleteInstruction(inst);
  //     instruction_list.push_back(inst);
  //   }
  //   // Test Ends
  //   AddNoop(10, instruction_list);
  //   std::cout << "instruction size: " << instruction_list.size() << std::endl;
  //   std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);
  //   for (uint32_t i = 1; i <= inst_vec.size(); i++)
  //   {
  //     std::cout << "inst_vec: " << i << std::endl;
  //     int leng = int(((i * 1.0) / inst_vec.size()) * 50);
  //     auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
  //     std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
  //     std::cout << std::string(50+2, '-') << std::endl;
  //     std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
  //     std::cout << std::string(50+2, '-') << std::endl;
  //     // Halt
  //     inst = new Instruction();
  //     ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  //     inst->SetOperationState(Instruction::SCALARONE, &scalar);
  //     CompleteInstruction(inst);
  //     inst_vec[i-1].push_back(inst);
      

  //     AddNoop(10, inst_vec[i-1]);

  //     simulator.WriteToImem(inst_vec[i-1]);
  //     simulator.Execute(0x7fffffff);

  //     for (auto it = range.first; it != range.second; it++)
  //     {
  //       if (it->second.name != "output")
  //       {
  //         float* test = new float[it->second.len];
  //         std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
  //         load_input(test, it->second.compare_file, *(int*)(&it->second.len));
  //         simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
  //         simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, std::to_string(z) + "/" + it->second.name);
  //       }
  //     }
  //   }
  //   simulator.PrintHBM_Write(weightaddr, weightaddr + 115 * 250880, "output" + std::to_string(z));
  //   instruction_list.clear();
  //   std::cout << "clear size: " << instruction_list.size() << std::endl;

    
    int num = lm_logit_weights.dims[0] / 1024;
    std::cout << "nums: " << num << std::endl;

    int vmemaddr = result.addr + result.size();
    usehbm = weightaddr + (result.dims[1] - 1) * lm_logit_weights.dims[0];
    for (int i = 0; i < num; i++)
    {
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(vmemaddr / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(vmemaddr / 32).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      vmemaddr += 1024;
    }
    VMEM_TO_HBM(instruction_list, result.addr + result.size(), usehbm, lm_logit_weights.dims[0]);

    // std::vector<uint32_t> label(115, 0);
    vec1d_t_i labels{-100, -100, -100, -100, -100, -100, -100, -100, -100, 
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, 
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, 1, 226041, 355, 8432, 19540, 
                    23124, 40651, 355, 842, 14194, 20451, 59280, 55675, 224909, 
                    60266, 420, 7436, 12142, 84793, 20451, 59280, 60266, 355, 
                    12703, 5136, 8401, 2079, 54682, 3616, 19651, 420, 12142, 
                    25477, 4990, 79267, 14554, 12142, 20451, 60266, 355, 58693, 
                    13344, 23107, 55675, 224909, 86689, 420, 2};
    int zero_num = 0;
    for (int i = 0; i < labels.size() - 1; i++)
    {
      labels[i] = labels[i + 1];
      if (labels[i] <= 0) zero_num++;
    }
    labels[labels.size() - 1] = 0; 
    std::cout << "zero_num: " << zero_num << std::endl;
    weights_use_vmem_size = (kVectorDataMemorySize - result.addr - result.size()) / 2;
    weight_use_row = weights_use_vmem_size / lm_logit_weights.dims[0];

    int weight_row = (kVectorDataMemorySize - result.addr - result.size()) / lm_logit_weights.dims[0];
    SplitNum = zero_num / weight_row ;
    for (int i = 0; i <= SplitNum; i++)
    {
      int weight_row_now = (zero_num - i * weight_row) >= weight_row ? weight_row : (zero_num - i * weight_row);
      std::cout << "zero row_now: " << weight_row_now << std::endl;
      if (1)
      {
        inst = new Instruction();
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      int zerosplit = weight_row_now * lm_logit_weights.dims[0] / 1024;
      uint32_t zerohbmaddr = weightaddr + i * weight_row * lm_logit_weights.dims[0];
      for (int j = 0; j < zerosplit; j++)
      {
        if (1)
        {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).first);
          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
          ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
          inst->SetOperationState(Instruction::SCALARONE, &set_base);
          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
          VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
          inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      VMEM_TO_HBM(instruction_list, result.addr + result.size(), zerohbmaddr, weight_row_now * lm_logit_weights.dims[0]);
    }

    int testnum = result.dims[1] - zero_num - 1;
    int testaddr = weightaddr + zero_num * lm_logit_weights.dims[0];
    SplitNum = testnum / weight_use_row;
    std::cout << "testaddr: " << testaddr << std::endl;
    std::cout << "testnum: " << testnum << std::endl;
    std::cout << "2 SplitNum: " << SplitNum << std::endl;
    int l = zero_num;
    for (int i = 0; i <= SplitNum; i++)
    {
      int weight_row_now = (testnum - i * weight_use_row) >= weight_use_row ? weight_use_row : (testnum - i * weight_use_row);
      if (weight_row_now == 0) break;
      data<3> soft(result.addr + result.size(), {1, weight_row_now, lm_logit_weights.dims[0]});
      std::cout << i << " soft: " << soft.addr << std::endl;
      HBM_TO_VMEM(instruction_list, testaddr + i * weight_use_row * lm_logit_weights.dims[0], soft.addr, lm_logit_weights.dims[0] * weight_row_now);
      data<3> leftsoft(soft.addr + soft.size(), {soft.dims[0], soft.dims[1], soft.dims[2]});
      Softmax(instruction_list, soft.addr, leftsoft.addr, soft.dims[0] * soft.dims[1], soft.dims[2]);
      std::cout << i << " leftsoft: " << leftsoft.addr << std::endl;
      // for (int j = 0; j < weight_row_now; j++)
      // {
      //   int batch1 = labels[l] / 1024;
      //   int batch2 = labels[l] % 1024;
      //   std::cout << "label " << l << ": " << labels[l] << std::endl;
      //   std::cout << "batch1: " << batch1 << std::endl;
      //   std::cout << "batch2: " << batch2 << std::endl;
      //   uint32_t softvmem = leftsoft.addr + j * lm_logit_weights.dims[0] + batch1 * 1024;
      //   std::cout << "softvmem: " << softvmem << std::endl;
      //   if (1)
      //   {
      //     inst = new Instruction();
      //     inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue((uint32_t)batch2).second);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue((uint32_t)batch2).first);
      //     VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, 0);
      //     inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
      //     VectorOperationState vmask0(V_S32_EQUAL, 0, 0, 44, 0);
      //     inst->SetOperationState(Instruction::VECTORTWO, &vmask0);
      //     CompleteInstruction(inst);
      //     instruction_list.push_back(inst);
      //   }
      //   if (1)
      //   {
      //     inst = new Instruction();
      //     inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(softvmem / 32).second);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(softvmem / 32).first);
      //     inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      //     ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      //     inst->SetOperationState(Instruction::SCALARONE, &set_base);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      //     VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
      //     inst->SetOperationState(Instruction::VECTORLOAD, &vload);
      //     CompleteInstruction(inst);
      //     instruction_list.push_back(inst);
      //   }
      //   if (1)
      //   {
      //     inst = new Instruction();
      //     VectorOperationState sub(V_F32_SUBTRACTION, 0, 0, 49, 1);
      //     inst->SetOperationState(Instruction::VECTORTWO, &sub);
      //     CompleteInstruction(inst);
      //     instruction_list.push_back(inst);
      //   }
      //   if (1)
      //   {
      //     inst = new Instruction();
      //     VectorOperationState vmask0(V_SELECT_VMASK0, 0, 0, 1, 0);
      //     inst->SetOperationState(Instruction::VECTORONE, &vmask0);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(softvmem / 32).second);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(softvmem / 32).first);
      //     inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
      //     ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
      //     inst->SetOperationState(Instruction::SCALARONE, &set_base);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
      //     inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
      //     VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
      //     inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
      //     CompleteInstruction(inst);
      //     instruction_list.push_back(inst);
      //   }
      //   l++;
      // }
      float div = 1.0 / (float(testnum));
      std::cout << "div: " << div << std::endl;
      Division(inst2, leftsoft, div);
      VMEM_TO_HBM(instruction_list, leftsoft.addr, testaddr + i * weight_use_row * lm_logit_weights.dims[0], lm_logit_weights.dims[0] * weight_row_now);
    }

    if (1)
    {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    int zerohbmaddr = weightaddr + (labels.size() - 1) * lm_logit_weights.dims[0];
    int zerosplit = lm_logit_weights.dims[0] / 1024;
    // for (int j = 0; j < zerosplit; j++)
    // {
    //   if (1)
    //   {
    //     inst = new Instruction();
    //     inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).second);
    //     inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).first);
    //     inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
    //     ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
    //     inst->SetOperationState(Instruction::SCALARONE, &set_base);
    //     inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
    //     inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
    //     VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
    //     inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
    //     CompleteInstruction(inst);
    //     instruction_list.push_back(inst);
    //   }
    // }
    // VMEM_TO_HBM(instruction_list, result.addr + result.size(), zerohbmaddr, lm_logit_weights.dims[0]);


    // // HBM_TO_VMEM(instruction_list, testaddr, 0, 2048);
    // data<4> softmax_input;
    // softmax_input.hbmaddr = weightaddr;
    // softmax_input.dims = {1, result.dims[0], result.dims[1], lm_logit_weights.dims[0]};
    
    // lm_logit_weights.hbmaddr = lm_logit_weights.addr;
    // // hidden_state = matmulIbWb(inst2, softmax_input, lm_logit_weights, Block_addr);

    // Halt
    if (1)
    {
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    // Test Ends
    AddNoop(10, instruction_list);
    std::cout << "instruction size: " << instruction_list.size() << std::endl;
    std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);
    for (uint32_t i = 1; i <= inst_vec.size(); i++)
    {
      std::cout << "inst_vec: " << i << std::endl;
      int leng = int(((i * 1.0) / inst_vec.size()) * 50);
      auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
      std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
      std::cout << std::string(50+2, '-') << std::endl;
      std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
      std::cout << std::string(50+2, '-') << std::endl;
      // Halt
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      inst_vec[i-1].push_back(inst);
      

      AddNoop(10, inst_vec[i-1]);

      simulator.WriteToImem(inst_vec[i-1]);
      simulator.Execute(0x7fffffff);

      // for (auto it = range.first; it != range.second; it++)
      // {
      //   if (it->second.name != "output")
      //   {
      //     float* test = new float[it->second.len];
      //     std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
      //     load_input(test, it->second.compare_file, *(int*)(&it->second.len));
      //     simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
      //     simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, std::to_string(z) + "/" + it->second.name);
      //   }
      // }
    }
    // simulator.DebugPrintVmem_Write(hidden_state.addr, hidden_state.addr + hidden_state.size(), "output");
    simulator.PrintHBM(index - 250880, index);
    instruction_list.clear();
    std::cout << "clear size: " << instruction_list.size() << std::endl;

  //   data<4> ln_f_in(hidden_state.addr + hidden_state.size(), {hidden_state.dims[0], hidden_state.dims[1], hidden_state.dims[2], hidden_state.dims[3]});
  //   HBM_TO_VMEM(instruction_list, forward_addr.at("transformer.ln_f_in"), ln_f_in.addr, ln_f_in.size());
  //   data<1> ln_f_weight(ln_f_in.addr + ln_f_in.size(), {1024});
  //   HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.weight"), ln_f_weight.addr, ln_f_weight.size());
  //   data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {1024});
  //   HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.bias"), ln_f_bias.addr, ln_f_bias.size());
  //   inst2.Spy("transformer.ln_f_in", hidden_state.asVMem(inst2), "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/transformer.ln_f_in.txt");
  //   data<4> ln_f_dx = LayerNormDxBackward(inst2, ln_f_in, hidden_state, ln_f_weight, ln_f_bias.addr + ln_f_bias.size(), config.layer_norm_epsilon);
  //   inst2.Spy("transformer.ln_f_out", ln_f_dx.asVMem(inst2), "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/transformer.ln_f_out.txt");
  //   data<1> ln_f_dw = LayerNormDwBackward(inst2, ln_f_in, hidden_state, ln_f_dx.addr + ln_f_dx.size(), config.layer_norm_epsilon);
  //   if (config.training)
  //   {
  //     data<2> ln_f_weight_new;
  //     ln_f_weight_new.hbmaddr = weights_addr.at("transformer.ln_f.weight");
  //     ln_f_weight_new.dims = {1, ln_f_weight.dims[0]};
  //     UpdateWeight(inst2, config.update_lr, ln_f_weight_new, ln_f_dw.as<2>(), ln_f_dw.addr + ln_f_dw.size());
  //   }
    
  //   data<1> ln_f_db = LayerNormDbBackward(inst2, hidden_state, ln_f_dw.addr + ln_f_dw.size());
  //   if (config.training)
  //   {
  //     data<1> ln_f_bias_new;
  //     ln_f_bias_new.hbmaddr = weights_addr.at("transformer.ln_f.bias");
  //     ln_f_bias_new.dims = {ln_f_bias.dims[0]};
  //     UpdateBias(inst2, config.update_lr, ln_f_bias_new, ln_f_db, ln_f_db.addr + ln_f_db.size());
  //   }


  //   data<4> block = BLOOMBlockBackward(inst2, config, ln_f_dx, ln_f_dx.addr + ln_f_dx.size(), ln_f_dx.addr + ln_f_dx.size(), "transformer.h.23", weights_addr, forward_addr);
    
  //   // Halt
  //   if (1)
  //   {
  //     inst = new Instruction();
  //     ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  //     inst->SetOperationState(Instruction::SCALARONE, &scalar);
  //     CompleteInstruction(inst);
  //     instruction_list.push_back(inst);
  //   }
  //   // Test Ends
  //   AddNoop(10, instruction_list);
  //   std::cout << "instruction size: " << instruction_list.size() << std::endl;
  //   inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);
  //   for (uint32_t i = 1; i <= inst_vec.size(); i++)
  //   {
  //     std::cout << "inst_vec: " << i << std::endl;
  //     int leng = int(((i * 1.0) / inst_vec.size()) * 50);
  //     auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
  //     std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
  //     std::cout << std::string(50+2, '-') << std::endl;
  //     std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
  //     std::cout << std::string(50+2, '-') << std::endl;
  //     // Halt
  //     inst = new Instruction();
  //     ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
  //     inst->SetOperationState(Instruction::SCALARONE, &scalar);
  //     CompleteInstruction(inst);
  //     inst_vec[i-1].push_back(inst);
      

  //     AddNoop(10, inst_vec[i-1]);

  //     simulator.WriteToImem(inst_vec[i-1]);
  //     simulator.Execute(0x7fffffff);

  //     for (auto it = range.first; it != range.second; it++)
  //     {
  //       float* test = new float[it->second.len];
  //       std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
  //       load_input(test, it->second.compare_file, *(int*)(&it->second.len));
  //       simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
  //       simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, std::to_string(z) + "b/" + it->second.name);
  //     }
  //   }
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.ln_f.weight"), weights_addr.at("transformer.ln_f.weight") + 1024, std::to_string(z) + "t/transformer.ln_f.weight");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.ln_f.bias"), weights_addr.at("transformer.ln_f.bias") + 1024, std::to_string(z) + "t/transformer.ln_f.bias");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.input_layernorm.weight"), weights_addr.at("transformer.h.23.input_layernorm.weight") + 1024, std::to_string(z) + "t/transformer.h.23.input_layernorm.weight");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.input_layernorm.bias"), weights_addr.at("transformer.h.23.input_layernorm.bias") + 1024, std::to_string(z) + "t/transformer.h.23.input_layernorm.bias");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.post_attention_layernorm.weight"), weights_addr.at("transformer.h.23.post_attention_layernorm.weight") + 1024, std::to_string(z) + "t/transformer.h.23.post_attention_layernorm.weight");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.post_attention_layernorm.bias"), weights_addr.at("transformer.h.23.post_attention_layernorm.bias") + 1024, std::to_string(z) + "t/transformer.h.23.post_attention_layernorm.bias");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.self_attention.query_key_value.weight"), weights_addr.at("transformer.h.23.self_attention.query_key_value.weight") + 1024 * 3072, std::to_string(z) + "t/transformer.h.23.self_attention.query_key_value.weight");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.self_attention.query_key_value.bias"), weights_addr.at("transformer.h.23.self_attention.query_key_value.bias") + 3072, std::to_string(z) + "t/transformer.h.23.self_attention.query_key_value.bias");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.self_attention.dense.weight"), weights_addr.at("transformer.h.23.self_attention.dense.weight") + 1024 * 1024, std::to_string(z) + "t/transformer.h.23.self_attention.dense.weight");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.self_attention.dense.bias"), weights_addr.at("transformer.h.23.self_attention.dense.bias") + 1024, std::to_string(z) + "t/transformer.h.23.self_attention.dense.bias");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.mlp.dense_h_to_4h.weight"), weights_addr.at("transformer.h.23.mlp.dense_h_to_4h.weight") + 1024 * 4096, std::to_string(z) + "t/transformer.h.23.mlp.dense_h_to_4h.weight");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.mlp.dense_h_to_4h.bias"), weights_addr.at("transformer.h.23.mlp.dense_h_to_4h.bias") + 4096, std::to_string(z) + "t/transformer.h.23.mlp.dense_h_to_4h.bias");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.mlp.dense_4h_to_h.weight"), weights_addr.at("transformer.h.23.mlp.dense_4h_to_h.weight") + 1024 * 4096, std::to_string(z) + "t/transformer.h.23.mlp.dense_4h_to_h.weight");
  //   simulator.PrintHBM_Write(weights_addr.at("transformer.h.23.mlp.dense_4h_to_h.bias"), weights_addr.at("transformer.h.23.mlp.dense_4h_to_h.bias") + 1024, std::to_string(z) + "t/transformer.h.23.mlp.dense_4h_to_h.bias");
  //   instruction_list.clear();
  //   std::cout << "clear size: " << instruction_list.size() << std::endl;
  //   float a = (float)(z);
  //   float b = (float)(numz - 1);
  //   config.update_lr = 0.0 + (config.update_lr_max + 0.0) / 2 * (1.0 + cos(a / b * PI));
  //   std::cout << "lr:" << config.update_lr << std::endl;
  // }
  return ;
}

void DropoutTest()
{
  std::cout << "DropoutTest\n";
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  // ============================ Insert your code above here =====================================

  if (1)
  {
    //move_ls1 ls_function_x for branch1_control
    //vmem_padding for vmemstore_address_update
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 0 );
    inst->SetImmediateValue(Instruction::IMMEDIATE1, 32 ); 
    ScalarOperationState move_ls1(S_U32_MOVE, 0, 0, 32, 7);
    inst->SetOperationState(Instruction::SCALARONE, &move_ls1);
    ScalarOperationState vmem_padding(S_U32_MOVE, 0, 0, 33, 8);
    inst->SetOperationState(Instruction::SCALARTWO, & vmem_padding);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);

    // set base address   
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 0 );
    inst->SetImmediateValue(Instruction::IMMEDIATE1, 1 );
    ScalarOperationState base_address( S_U32_MOVE, 0, 0, 32, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_address);
    ScalarOperationState move_one(S_U32_MOVE, 0, 0, 33, 1);
    inst->SetOperationState(Instruction::SCALARTWO, &move_one);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  
  inst = new Instruction();
  VectorOperationState Get_core(V_GET_V_CORE_ID, 0, 0, 0, 1);
  inst->SetOperationState(Instruction::VECTORONE, &Get_core);
  VectorOperationState Set_seed(V_RNG_RESEED, 0, 1, 0, 0);
  inst->SetOperationState(Instruction::VECTORTWO, &Set_seed);
  CompleteInstruction(inst);
  instruction_list.push_back(inst);

  AddNoop(10, instruction_list);

  if (1)
  {
    //reg4 control num of branch1_loop
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, 1024);    //1024*8*4=32m
    ScalarOperationState move_branch1( S_U32_MOVE, 0, 0, 32, 4);
    inst->SetOperationState(Instruction::SCALARONE, & move_branch1);
    ScalarOperationState base_update(S_S32_SUBTRACTION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARTWO, & base_update);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);

    inst = new Instruction();
    ScalarOperationState base_update1(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update1);
    VectorOperationState RNG1(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 1);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG1);
    VectorStoreOperationState store1(V_STORE, 0, 1, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store1);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);   

    inst = new Instruction();
    ScalarOperationState base_update2(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update2);
    VectorOperationState RNG2(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 2);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG2);
    VectorStoreOperationState store2(V_STORE, 0, 2, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store2);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);   

    inst = new Instruction();
    ScalarOperationState base_update3(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update3);
    VectorOperationState RNG3(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 3);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG3);
    VectorStoreOperationState store3(V_STORE, 0, 3, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store3);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);   

    inst = new Instruction();
    ScalarOperationState base_update4(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update4);
    VectorOperationState RNG4(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 4);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG4);
    VectorStoreOperationState store4(V_STORE, 0, 4, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store4);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);   

    inst = new Instruction();
    ScalarOperationState base_update5(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update5);
    VectorOperationState RNG5(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 5);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG5);
    VectorStoreOperationState store5(V_STORE, 0, 5, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store5);
    CompleteInstruction(inst);
    instruction_list.push_back(inst); 

    inst = new Instruction();
    ScalarOperationState base_update6(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update6);
    VectorOperationState RNG6(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 6);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG6);
    VectorStoreOperationState store6(V_STORE, 0, 6, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store6);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);   
    
    inst = new Instruction();
    ScalarOperationState base_update7(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update7);
    VectorOperationState RNG7(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 7);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG7);
    VectorStoreOperationState store7(V_STORE, 0, 7, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store7);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);   
    
    inst = new Instruction();
    ScalarOperationState base_update8(S_S32_ADDITION, 0, 0, 8, 0);
    inst->SetOperationState(Instruction::SCALARONE, & base_update8);
    VectorOperationState RNG8(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, 8);
    inst->SetOperationState(Instruction::VECTORTWO, &RNG8);
    VectorStoreOperationState store8(V_STORE, 0, 8, 1, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &store8);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);   

    // branch to control store2vmem loop 
    inst = new Instruction();
    ScalarOperationState update_ls1(S_S32_ADDITION, 0, 1, 7, 7);
    inst->SetOperationState(Instruction::SCALARONE, & update_ls1);
    ScalarOperationState ls_than(S_S32_LESSER, 0, 7, 4, 1);
    inst->SetOperationState(Instruction::SCALARTWO, & ls_than);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);

    // inst = new Instruction();
    // inst->SetImmediateValue(Instruction::IMMEDIATE0, -10);
    // ScalarOperationState branch(S_BRANCH, 1, 0, 0, 1);
    // inst->SetOperationState(Instruction::SCALARONE, &branch);
    // CompleteInstruction(inst);
    // instruction_list.push_back(inst); 
  }


  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(0 / 32).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(0 / 32).first);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
    ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
    inst->SetOperationState(Instruction::SCALARONE, &set_base);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
    VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 2, 1, 2, 4, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  // Halt
  if (1)
  {
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  AddNoop(10, instruction_list);

  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();


  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);
  std::cout << "Instruction size: " << instruction_list.size() << std::endl;
  for(uint32_t i = 1; i <= inst_vec.size(); i++) {
    std::cout << "inst_vec: " << i << std::endl;
		int leng = int(((i * 1.0) / inst_vec.size()) * 50);
		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
		std::cout << std::string(50+2, '-') << std::endl;
		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
		std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(30000000);
  }

  simulator.DebugPrintVmem(0, 1024);
}

void dropoutTest() {

   std::cout << "RNGTest\n";

  Inst2 inst2;

  std::vector<Instruction*> &instruction_list = inst2.inst.insts;

  Instruction* inst;

  std::vector<Instruction> bundle;

  // ============================ Insert your code above here =====================================



  ShowFuncCallInfo() = true;

  HBM_TO_VMEM(instruction_list, 0, 0, 1024);

  data<3> input(0, {1, 1, 1024});



  auto seed_reg = inst2.AllocVReg("");

  inst2(VCoreId, seed_reg.id);

  inst2(VShlU, seed_reg.id, inst2.inst.ImmeU(7), seed_reg.id);

  inst2(VReSeed, seed_reg.id);



  auto res = dropouttest2(inst2, input, 0.6, 1024);



  // Halt

  inst = new Instruction();

  ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);

  inst->SetOperationState(Instruction::SCALARONE, &scalar);

  CompleteInstruction(inst);

  instruction_list.push_back(inst);

  /////////////////////////////////////////////////////////////////////////////////////////

  // Test Ends



  AddNoop(10, instruction_list);



  // Creating simulator Object and start running.

  Device_Simulator simulator(Device::DEVICE_SIMULATOR);

  simulator.OpenDeviceWithHBM();



  uint32_t* input_data = new uint32_t[1024];

  for(int i = 0; i < 1024; i++) {

    float temp = i * 1.0;

    input_data[i] = *(uint32_t *)(&temp);

  }



  simulator.WriteToHBM(input_data, 1024, 0);



  std::cout << "old instruct.size: " << instruction_list.size() << std::endl;



  // instruction_list = schedule(instruction_list);



  // std::cout << "new instruct.size: " << instruction_list.size() << std::endl;

  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000);



  for(uint32_t i = 1; i <= inst_vec.size(); i++) {

    std::cout << "inst_index: " << i << " inst_vec_size: " << inst_vec.size() << std::endl;

		int leng = int(((i * 1.0) / inst_vec.size()) * 50);

		std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";

		std::cout << std::string(50+2, '-') << std::endl;

		std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;

		std::cout << std::string(50+2, '-') << std::endl;

    // Halt

    inst = new Instruction();

    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);

    inst->SetOperationState(Instruction::SCALARONE, &scalar);

    CompleteInstruction(inst);

    inst_vec[i-1].push_back(inst);



    AddNoop(10, inst_vec[i-1]);



    simulator.WriteToImem(inst_vec[i-1]);

    simulator.Execute(3000000000);

  }

  

  std::cout << "Execute end\n";



  // simulator.DebugPrintVmem(0, 1024);

  simulator.DebugPrintVmem(res.addr, res.addr + res.size());

  return;
}

void lossTest()
{
  std::cout << "loss Test" << std::endl;
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  HBMAddr() = 0;
  BLOOMConfig config;

  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  data<4> hidden_state(0, {1, 1, 115, 250880});
  data<2> lm_logit_weights(hidden_state.addr + hidden_state.size(), {250880, 1024});
  uint32_t weightaddr = lm_logit_weights.addr + lm_logit_weights.size();


  vec1d_t_i labels{-100, -100, -100, -100, -100, -100, -100, -100, -100, 
                  -100, -100, -100, -100, -100, -100, -100, -100, -100, 
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                  -100, -100, -100, -100, 1, 226041, 355, 8432, 19540, 
                  23124, 40651, 355, 842, 14194, 20451, 59280, 55675, 224909, 
                  60266, 420, 7436, 12142, 84793, 20451, 59280, 60266, 355, 
                  12703, 5136, 8401, 2079, 54682, 3616, 19651, 420, 12142, 
                  25477, 4990, 79267, 14554, 12142, 20451, 60266, 355, 58693, 
                  13344, 23107, 55675, 224909, 86689, 420, 2};

  int zero_num = 0;
  for (int i = 0; i < labels.size() - 1; i++)
  {
    labels[i] = labels[i + 1];
    if (labels[i] <= 0) zero_num++;
  }
  labels[labels.size() - 1] = 0;
  std::cout << "zero_num: " << zero_num << std::endl;

  int testnum = hidden_state.dims[2] - zero_num - 1;
  std::cout << "testnum: " << testnum << std::endl;
  int testaddr = hidden_state.addr + zero_num * hidden_state.dims[3];
  std::cout << "testaddr: " << testaddr << std::endl;
  int l = zero_num;
  
  uint32_t test_weight = kVectorDataMemorySize - testnum * 1024;
  std::cout << "test_weight: " << test_weight << std::endl;
  int weights_use_vmem_size = test_weight / 2;
  int weight_use_row = weights_use_vmem_size / lm_logit_weights.dims[0];
  int SplitNum = testnum / weight_use_row;
  std::cout << "SplitNum: " << SplitNum << std::endl;


  for (int i = 0; i <= SplitNum; i++)
  {
    int weight_row_now = (testnum - i * weight_use_row) >= weight_use_row ? weight_use_row : (testnum - i * weight_use_row);
    if (weight_row_now == 0) break;
    data<3> soft(0, {1, weight_row_now, lm_logit_weights.dims[0]});
    HBM_TO_VMEM(instruction_list, testaddr + i * weight_use_row * lm_logit_weights.dims[0], soft.addr, lm_logit_weights.dims[0] * weight_row_now);
    data<3> leftsoft(soft.addr + soft.size(), {soft.dims[0], soft.dims[1], soft.dims[2]});
    std::cout << i << " leftsoft: " << leftsoft.addr << std::endl;
    Softmax(instruction_list, soft.addr, leftsoft.addr, soft.dims[0] * soft.dims[1], soft.dims[2]);

    if (1)
    {
      inst = new Instruction();
      VectorOperationState mov(V_U32_MOVE, 0, 0, 53, 31);
      inst->SetOperationState(Instruction::VECTORONE, &mov);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      VectorOperationState log2(V_F32_LOG2, 0, 31, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &log2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      MTROperationState read_urf_2(MTR_READ_UNARY_EXECUTION_RESULT, 0, 31, 0);
      inst->SetOperationState(Instruction::MTR, &read_urf_2);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      VectorOperationState rcp(V_F32_RECIPROCAL, 0, 31, 0, 0);
      inst->SetOperationState(Instruction::VECTORONE, &rcp);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 31, 0);
      inst->SetOperationState(Instruction::MTR, &urf_pop);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);  
    }
    // if (1)
    // {
    //   inst = new Instruction();
    //   MTROperationState read_urf_2(MTR_READ_UNARY_EXECUTION_RESULT, 0, 31, 0);
    //   inst->SetOperationState(Instruction::MTR, &read_urf_2);
    //   CompleteInstruction(inst);
    //   instruction_list.push_back(inst);
    //   }

    for (int j = 0; j < weight_row_now; j++)
    {
      int batch1 = labels[l] / 1024;
      int batch2 = labels[l] % 1024;
      std::cout << "label " << l << ": " << labels[l] << std::endl;
      std::cout << "batch1: " << batch1 << std::endl;
      std::cout << "batch2: " << batch2 << std::endl;
      uint32_t softvmem = leftsoft.addr + j * lm_logit_weights.dims[0] + batch1 * 1024;
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(softvmem / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(softvmem / 32).first);
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
        VectorOperationState log2(V_F32_LOG2, 0, 1, 0, 0);
        inst->SetOperationState(Instruction::VECTORONE, &log2);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      if (1)
      {
        inst = new Instruction();
        MTROperationState read_urf_1(MTR_READ_UNARY_EXECUTION_RESULT, 0, 1, 0);
        inst->SetOperationState(Instruction::MTR, &read_urf_1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      if (1)
      {
        inst = new Instruction();
        VectorOperationState mul(V_F32_MULTIPLICATION, 0, 1, 31, 1);
        inst->SetOperationState(Instruction::VECTORONE, &mul);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      int label_num = i * weight_use_row + j;
      std::cout << "num: " << label_num << std::endl;
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((test_weight + label_num * 1024) / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((test_weight + label_num * 1024) / 32).first);
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
      l++;
    }
  }
  VMEM_TO_HBM(instruction_list, test_weight, weightaddr, testnum * 1024);
  HBM_TO_SMEM(instruction_list, weightaddr, 0, testnum * 1024);

  // // if (1)
  // // {
  // //   inst = new Instruction();
  // //   ScalarOperationState smov(S_U32_MOVE, 0, 0, 46, 31);
  // //   inst->SetOperationState(Instruction::SCALARONE, &smov);
  // //   CompleteInstruction(inst);
  // //   instruction_list.push_back(inst);
  // // }
  if (1) {
    inst = new Instruction();
    ScalarOperationState s_mov(S_U32_MOVE, 0 /* perm_value */, 0 /* s_x */, 46 /* s_y */,
                               31 /* s_dest */);
    inst->SetOperationState(Instruction::SCALARONE, &s_mov);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  for (int i = 0; i < testnum; i++)
  {
    int batch2 = labels[zero_num + i] % 1024;
    std::cout << "batch2: " << batch2 << std::endl;
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(4 * (i * 1024 + batch2)).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(4 * (i * 1024 + batch2)).first);
      ScalarOperationState sload(S_SMEM_LOAD, 0, 0, 44, 28);
      inst->SetOperationState(Instruction::SCALARTWO, &sload);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(4 * i).second);
      inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(4 * i).first);
      ScalarOperationState sstore(S_SMEM_STORE, 0, 28, 44, 0);
      inst->SetOperationState(Instruction::SCALARTWO, &sstore);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    if (1)
    {
      inst = new Instruction();
      ScalarOperationState sadd(S_F32_ADDITION, 0, 28, 31, 31);
      inst->SetOperationState(Instruction::SCALARTWO, &sadd);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
  }
  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(4 * testnum).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(4 * testnum).first);
    ScalarOperationState sstore(S_SMEM_STORE, 0, 31, 44, 0);
    inst->SetOperationState(Instruction::SCALARTWO, &sstore);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 31);
    VectorOperationState vmov(V_U32_MOVE, 0, 0, 71, 1);
    inst->SetOperationState(Instruction::VECTORONE, &vmov);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  float test_num = (float)(testnum);
  std::cout << "test_num: " << test_num << std::endl;
  uint32_t unum = *(uint32_t*)(&test_num);
  std::cout << "unum: " << unum << std::endl;

  if (1)
  {
    inst = new Instruction();
    inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue(unum).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue(unum).first);
    VectorOperationState vmov(V_U32_MOVE, 0, 0, 44, 2);
    inst->SetOperationState(Instruction::VECTORONE, &vmov);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  if (1)
  {
    inst = new Instruction();
    VectorOperationState rcp(V_F32_RECIPROCAL, 0, 2, 0, 0);
    inst->SetOperationState(Instruction::VECTORONE, &rcp);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  if (1)
  {
    inst = new Instruction();
    MTROperationState urf_pop(MTR_READ_UNARY_EXECUTION_RESULT, 0, 2, 0);
    inst->SetOperationState(Instruction::MTR, &urf_pop);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);  
  }
  if (1)
  {
    inst = new Instruction();
    VectorOperationState mul(V_F32_MULTIPLICATION, 0, 2, 1, 1);
    inst->SetOperationState(Instruction::VECTORONE, &mul);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  if (1)
  {
    inst = new Instruction();
    // inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(0 / 32).second);
    // inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(0 / 32).first);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
    ScalarOperationState set_base(S_U32_MOVE, 0, 0, 46, 0);
    inst->SetOperationState(Instruction::SCALARONE, &set_base);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
    VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 1, 1, 2, 4, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }
  uint32_t* datas = new uint32_t[weightaddr];
  float* input1 = new float[115 * 250880];
  float* lm_weight = new float[250880 * 1024];
  uint32_t index = 0;
  load_input(input1, "/home/yinxun/dlc_simulator_test/src/simu/block_lnf_15_nodrop/output12.txt", 115 * 250880);
  load_input(lm_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.lm_head.weight.txt", 250880 * 1024);
  for (uint32_t i = 0; i < 115 * 250880; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(input1[i]));
  }
  for (uint32_t i = 0; i < 250880 * 1024; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lm_weight[i]));
  }

  std::cout << "index: " << index << std::endl;
  simulator.WriteToHBM(datas, index, 0);

  // Halt
  if (1)
  {
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    instruction_list.push_back(inst);
  }

  // Test Ends
  AddNoop(10, instruction_list);
  std::cout << "Instruction size: " << instruction_list.size() << std::endl;
  std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);
  for (uint32_t i = 1; i <= inst_vec.size(); i++)
  {
    std::cout << "inst_vec: " << i << std::endl;
    int leng = int(((i * 1.0) / inst_vec.size()) * 50);
    auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
    std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
    std::cout << std::string(50+2, '-') << std::endl;
    std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
    std::cout << std::string(50+2, '-') << std::endl;
    // Halt
    inst = new Instruction();
    ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    CompleteInstruction(inst);
    inst_vec[i-1].push_back(inst);
    

    AddNoop(10, inst_vec[i-1]);

    simulator.WriteToImem(inst_vec[i-1]);
    simulator.Execute(0x7fffffff);
  }
  // simulator.PrintHBM(weightaddr, weightaddr + testnum * 1024);
  simulator.DebugPrintSmem(0, testnum+1);
  simulator.DebugPrintVmem(0, 1024);

  return ;
}

void ttTest7B()
{
  std::cout << "TT Test" << std::endl;
  Inst2 inst2;
  std::vector<Instruction*> &instruction_list = inst2.inst.insts;
  Instruction *inst;
  std::vector<Instruction> bundle;
  HBMAddr() = 0;
  HBM_TO_VMEM(instruction_list, 0, 0, 115 * 4096);
  data<4> hidden_state(0, {1, 1, 115, 4096});
  data<2> attention_mask(hidden_state.addr + hidden_state.size(), {1, 115});
  data<2> word_embeddings_weight(hidden_state.addr + hidden_state.size(), {250880, 4096});
  data<1> word_embeddings_layernorm_weight(word_embeddings_weight.addr + word_embeddings_weight.size(), {4096});
  data<1> word_embeddings_layernorm_bias(word_embeddings_layernorm_weight.addr + word_embeddings_layernorm_weight.size(), {4096});
  data<1> ln_f_weight(word_embeddings_layernorm_bias.addr + word_embeddings_layernorm_bias.size(), {4096});
  data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {4096});
  data<2> lm_logit_weights(ln_f_bias.addr + ln_f_bias.size(), {250880, 4096});
  
  std::map<std::string, uint32_t> weights_addr{{"transformer.word_embeddings.weight", word_embeddings_weight.addr},
                                              {"transformer.word_embeddings_layernorm.weight", word_embeddings_layernorm_weight.addr},
                                              {"transformer.word_embeddings_layernorm.bias", word_embeddings_layernorm_bias.addr},
                                              {"transformer.ln_f.weight", ln_f_weight.addr},
                                              {"transformer.ln_f.bias", ln_f_bias.addr},
                                              {"transformer.lm_head.weight", lm_logit_weights.addr}};
  uint32_t weightaddr = lm_logit_weights.addr + lm_logit_weights.size();
  for (int i = 29; i < 30; i++)
  {
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.weight", weightaddr));
    weightaddr += 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm.bias", weightaddr));
    weightaddr += 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight", weightaddr));
    weightaddr += 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias", weightaddr));
    weightaddr += 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight", weightaddr));
    weightaddr += 4096 * 4096 * 3;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias", weightaddr));
    weightaddr += 4096 * 3;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.weight", weightaddr));
    weightaddr += 4096 * 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense.bias", weightaddr));
    weightaddr += 4096;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight", weightaddr));
    weightaddr += 4096 * 4096 * 4;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias", weightaddr));
    weightaddr += 4096 * 4;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight", weightaddr));
    weightaddr += 4096 * 4096 * 4;
    weights_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias", weightaddr));
    weightaddr += 4096;
  }

  std::map<std::string, uint32_t> forward_addr{{"transformer.ln_f_in", weightaddr}};
  weightaddr += 115 * 4096;
  for (int i = 29; i >= 29; i--)
  {
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h_in", weightaddr));
    weightaddr += 115 * 4096 * 4;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.gelu_impl_in", weightaddr));
    weightaddr += 115 * 4096 * 4;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h_in", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".post_attention_layernorm_in", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.dense_in", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.attention_probs_reshaped", weightaddr));
    weightaddr += 32 * 115 * 115;
    weightaddr = ((weightaddr + 128) / 128) * 128;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.value_layer", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.softmax_out", weightaddr));
    weightaddr += 32 * 115 * 115;
    weightaddr = ((weightaddr + 128) / 128) * 128;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_layer", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.key_layer", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".self_attention.query_key_value_in", weightaddr));
    weightaddr += 115 * 4096;
    forward_addr.insert(std::make_pair("transformer.h." + std::to_string(i) + ".input_layernorm_in", weightaddr));
    weightaddr += 115 * 4096;
  }
  std::cout << "weightaddr: " << weightaddr << std::endl;

  BLOOMConfig config;

  Device_Simulator simulator(Device::DEVICE_SIMULATOR);
  simulator.OpenDeviceWithHBM();

  uint32_t* datas = new uint32_t[weightaddr];
  float* input1 = new float[115*4096]; 
  float* emb_weight = new float[250880 * 4096];
  float* wel_weight = new float[4096];
  float* wel_bias = new float[4096];
  float* lf_weight = new float[4096];
  float* lf_bias = new float[4096];
  float* lm_weight = new float[250880 * 4096];
  uint32_t index = 0;
  load_input(input1, Train_7B_For + "transformer.h.29_in.txt", 115 * 4096);
  load_input(emb_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.word_embeddings.weight.txt", 4096 * 250880);
  load_input(wel_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.word_embeddings_layernorm.weight.txt", 4096);
  load_input(wel_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.word_embeddings_layernorm.bias.txt", 4096);
  load_input(lf_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.ln_f.weight.txt", 4096);
  load_input(lf_bias, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.ln_f.bias.txt", 4096);
  load_input(lm_weight, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.lm_head.weight.txt", 4096 * 250880);
  for (uint32_t i = 0; i < 115 * 4096; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(input1[i]));
  }
  for(uint32_t i = 0; i < 250880 * 4096; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(emb_weight[i]));
  }
  for(uint32_t i = 0; i < 4096; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(wel_weight[i]));
  }
  for(uint32_t i = 0; i < 4096; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(wel_bias[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lf_weight[i]));
  }
  for (uint32_t i = 0; i < 4096; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lf_bias[i]));
  }
  for(uint32_t i = 0; i < 250880 * 4096; i++, index++)
  {
    datas[index] = *(uint32_t*)(&(lm_weight[i]));
  }

  for (int i = 29; i <= 29; i++)
  {
    float *matrix1 = new float[4096];
    load_input(matrix1, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".input_layernorm.weight.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix1[j]));
    float *matrix2 = new float[4096];
    load_input(matrix2, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".input_layernorm.bias.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix2[j]));
    float *matrix7 = new float[4096];
    load_input(matrix7, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix7[j]));
    float *matrix8 = new float[4096];
    load_input(matrix8, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix8[j]));
    float *matrix3 = new float[3 * 4096 * 4096];
    load_input(matrix3, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight.txt", 3 * 4096 * 4096);
    for (uint32_t j = 0; j < 3 * 4096 * 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix3[j]));
    float *matrix4 = new float[3 * 4096];
    load_input(matrix4, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias.txt", 3 * 4096);
    for (uint32_t j = 0; j < 3 * 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix4[j]));
    float *matrix5  = new float[4096 * 4096];
    load_input(matrix5, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.weight.txt", 4096 * 4096);
    for (uint32_t j = 0; j < 4096 * 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix5[j]));
    float *matrix6 = new float[4096];
    load_input(matrix6, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".self_attention.dense.bias.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix6[j]));
    float *matrix9 = new float[4096 * 4096 * 4];
    load_input(matrix9, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight.txt", 4096 * 4096 * 4);
    for (uint32_t j = 0; j < 4096 * 4096 * 4; j++, index++) datas[index] = *(uint32_t*)(&(matrix9[j]));
    float *matrix10 = new float[4096 * 4];
    load_input(matrix10, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias.txt", 4096 * 4);
    for (uint32_t j = 0; j < 4096 * 4; j++, index++) datas[index] = *(uint32_t*)(&(matrix10[j]));
    float *matrix11 = new float[4096 * 4096 * 4];
    load_input(matrix11, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight.txt", 4096 * 4096 * 4);
    for (uint32_t j = 0; j < 4096 * 4096 * 4; j++, index++) datas[index] = *(uint32_t*)(&(matrix11[j]));
    float *matrix12 = new float[4096];
    load_input(matrix12, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_7b_f32/transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias.txt", 4096);
    for (uint32_t j = 0; j < 4096; j++, index++) datas[index] = *(uint32_t*)(&(matrix12[j]));
  } 
  std::cout << "index: " << index << std::endl;
  simulator.WriteToHBM(datas, index, 0);

  uint32_t* amask = new uint32_t[128];
  for(int i = 0; i < 115; i++) amask[i] = 1;
  simulator.WriteToVmemWithOffset((char*)amask, 128 / 128, attention_mask.addr);

  uint32_t *alibiu = new uint32_t[3712];
  float *alibif = new float[32 * 115];
  load_input(alibif, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/alibi_7b.txt", 32 * 115);
  for(uint32_t i = 0; i < 32 * 115; i++) alibiu[i] = *(uint32_t*)(&(alibif[i]));
  std::cout << "=====" << std::endl;
  simulator.WriteToVmemWithOffset((char*)alibiu, 3712 / 128, AlignTo128Bytes(attention_mask.addr + attention_mask.size()));

  std::tuple<uint32_t, uint32_t> input_shape(attention_mask.dims[0], attention_mask.dims[1]);
  data<3> alibi(attention_mask.addr + AlignTo128Bytes(attention_mask.size()), {config.n_head * attention_mask.dims[0], 1, attention_mask.dims[1]});
  // build_alibi(inst2, attention_mask, config.n_head, alibi.addr);
  std::cout << "alibi.addr: " << alibi.addr << std::endl;
  std::cout << "alibi: " << alibi.dims[0] << " " << alibi.dims[1] << " " << alibi.dims[2] << std::endl;
  // inst2.Spy("baddbmm_in", alibi.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.h.23.self_attention.baddbmm_in.txt");
  data<4> causal_mask = _prepare_attn_mask(inst2, attention_mask, input_shape, 0, alibi.addr + AlignTo128Bytes(alibi.size()));
  std::cout << "causal_mask: " << causal_mask.dims[0] << " " << causal_mask.dims[1] << " " << causal_mask.dims[2] << " " << causal_mask.dims[3] << std::endl;
  // inst2.Spy("causal_mask", causal_mask.asVMem(inst2), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/transformer.h.23.self_attention.baddbmm_in.txt");
  uint32_t Block_addr = causal_mask.addr + AlignTo128Bytes(causal_mask.size());
  std::cout << "block_addr: " << Block_addr << std::endl;
  
  int numz = 15;
  std::string weight_name = "transformer";

  for (int z = 0; z < numz; z++)
  {
    hidden_state.addr = 0;
    hidden_state.dims = {1, 1, 115, 4096};

    data<3> result(Block_addr, {1, 115, 4096});
    for (int p = 29; p < 30; p++)
    {
      inst2.Spy(weight_name + ".h." + std::to_string(p) + "_in", hidden_state.asVMem(inst2), Train_7B_For + weight_name + ".h." + std::to_string(p) + "_in.txt");
      result = BLOOMBlock(inst2, config, hidden_state, alibi.as<4>(), causal_mask, weight_name + ".h." + std::to_string(p), weights_addr, forward_addr, Block_addr);
      inst2.Spy(weight_name + ".h." + std::to_string(p) + "_out", result.asVMem(inst2), Train_7B_For + weight_name + ".h." + std::to_string(p) + "_out.txt");
      hidden_state.addr = result.addr;
    }
    data<1> lnf_weight(result.addr + result.size(), {config.hidden_size});
    data<1> lnf_bias(lnf_weight.addr + lnf_weight.size(), {config.hidden_size});
    HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.weight"), lnf_weight.addr, lnf_weight.size());
    HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.bias"), lnf_bias.addr, lnf_bias.size());
    
    if (config.training)
    {
      std::string name = "transformer.ln_f_in";
      if (forward_addr.find(name) == forward_addr.end()) std::cout << "Not forwardMap: " << name << std::endl;
      VMEM_TO_HBM(instruction_list, result.addr, forward_addr.at(name), result.size());
    }
    inst2.Spy("transformer.ln_f_in", result.asVMem(inst2), Train_7B_For + "transformer.ln_f_in.txt");
    result = LayerNorm(inst2, result, lnf_weight, lnf_bias, lnf_bias.addr + lnf_bias.size(), config.layer_norm_epsilon);
    inst2.Spy("transformer.ln_f_out", result.asVMem(inst2), Train_7B_For + "transformer.ln_f_out.txt");

    uint32_t usehbm = weightaddr;
    uint32_t weights_use_vmem_size = (kVectorDataMemorySize - result.addr - result.size());
    // uint32_t weight_use_row = (weights_use_vmem_size / input_ids.dims[2]) / 128 * 128;
    int weight_use_row = (weights_use_vmem_size / (result.dims[1] + lm_logit_weights.dims[1])) / 128 * 128;
    std::cout << "weight_use_row: " << weight_use_row << std::endl;

    int SplitNum = lm_logit_weights.dims[0] / weight_use_row;
    std::cout << "SplitNum: " << SplitNum << std::endl;
    for (int i = 0; i <= SplitNum; i++)
    {
      int weight_row_now = (lm_logit_weights.dims[0] - i * weight_use_row) >= weight_use_row ? weight_use_row : (lm_logit_weights.dims[0] - i * weight_use_row);
      std::cout << "weight_row_now: " << weight_row_now << std::endl;
      data<2> now_weight;
      now_weight.hbmaddr = lm_logit_weights.addr + i * weight_use_row * lm_logit_weights.dims[1];
      std::cout << "now_weight hbmaddr: " << now_weight.hbmaddr << std::endl;
      now_weight.dims = {(uint32_t)weight_row_now, lm_logit_weights.dims[1]};

      data<3> lm_head_output = linearNobias(inst2, result, now_weight, result.addr + result.size());
      std::cout << "lm_head_output: " << lm_head_output.dims[0] << ' ' << lm_head_output.dims[1] << ' ' << lm_head_output.dims[2] << std::endl;

      uint32_t vmemaddr = lm_head_output.addr;
      uint32_t outputhbm = usehbm;
      for (int i = 0; i < lm_head_output.dims[1]; i++)
      {
        VMEM_TO_HBM(instruction_list, vmemaddr, outputhbm, lm_head_output.dims[2]);
        vmemaddr += lm_head_output.dims[2];
        outputhbm += lm_logit_weights.dims[0];
      }
      usehbm += lm_head_output.dims[2];
    }
    // Halt
    if (1)
    {
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    // Test Ends
    AddNoop(10, instruction_list);
    std::cout << "instruction size: " << instruction_list.size() << std::endl;
    std::vector<std::vector<Instruction *>> inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);
    for (uint32_t i = 1; i <= inst_vec.size(); i++)
    {
      std::cout << "inst_vec: " << i << std::endl;
      int leng = int(((i * 1.0) / inst_vec.size()) * 50);
      auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
      std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
      std::cout << std::string(50+2, '-') << std::endl;
      std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
      std::cout << std::string(50+2, '-') << std::endl;
      // Halt
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      inst_vec[i-1].push_back(inst);
      

      AddNoop(10, inst_vec[i-1]);

      simulator.WriteToImem(inst_vec[i-1]);
      simulator.Execute(0x7fffffff);

      for (auto it = range.first; it != range.second; it++)
      {
        if (it->second.name != "output")
        {
          float* test = new float[it->second.len];
          std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
          load_input(test, it->second.compare_file, *(int*)(&it->second.len));
          simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
          simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, std::to_string(z) + "/" + it->second.name);
        }
      }
    }
    simulator.PrintHBM_Write(weightaddr, weightaddr + 115 * 250880, "output" + std::to_string(z));
    instruction_list.clear();
    std::cout << "clear size: " << instruction_list.size() << std::endl;

    
    int num = lm_logit_weights.dims[0] / 1024;
    std::cout << "nums: " << num << std::endl;

    int vmemaddr = result.addr + result.size();
    usehbm = weightaddr + (result.dims[1] - 1) * lm_logit_weights.dims[0];
    for (int i = 0; i < num; i++)
    {
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(vmemaddr / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(vmemaddr / 32).first);
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      vmemaddr += 1024;
    }
    VMEM_TO_HBM(instruction_list, result.addr + result.size(), usehbm, lm_logit_weights.dims[0]);

    // std::vector<uint32_t> label(115, 0);
    vec1d_t_i labels{-100, -100, -100, -100, -100, -100, -100, -100, -100, 
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, 
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, -100, -100, -100, -100, -100,  
                    -100, -100, -100, -100, 1, 226041, 355, 8432, 19540, 
                    23124, 40651, 355, 842, 14194, 20451, 59280, 55675, 224909, 
                    60266, 420, 7436, 12142, 84793, 20451, 59280, 60266, 355, 
                    12703, 5136, 8401, 2079, 54682, 3616, 19651, 420, 12142, 
                    25477, 4990, 79267, 14554, 12142, 20451, 60266, 355, 58693, 
                    13344, 23107, 55675, 224909, 86689, 420, 2};
    int zero_num = 0;
    for (int i = 0; i < labels.size() - 1; i++)
    {
      labels[i] = labels[i + 1];
      if (labels[i] <= 0) zero_num++;
    }
    labels[labels.size() - 1] = 0; 
    std::cout << "zero_num: " << zero_num << std::endl;
    weights_use_vmem_size = (kVectorDataMemorySize - result.addr - result.size()) / 2;
    weight_use_row = weights_use_vmem_size / lm_logit_weights.dims[0];

    int weight_row = (kVectorDataMemorySize - result.addr - result.size()) / lm_logit_weights.dims[0];
    SplitNum = zero_num / weight_row ;
    for (int i = 0; i <= SplitNum; i++)
    {
      int weight_row_now = (zero_num - i * weight_row) >= weight_row ? weight_row : (zero_num - i * weight_row);
      std::cout << "zero row_now: " << weight_row_now << std::endl;
      if (1)
      {
        inst = new Instruction();
        VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
        inst->SetOperationState(Instruction::VECTORONE, &move);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
      int zerosplit = weight_row_now * lm_logit_weights.dims[0] / 1024;
      uint32_t zerohbmaddr = weightaddr + i * weight_row * lm_logit_weights.dims[0];
      for (int j = 0; j < zerosplit; j++)
      {
        if (1)
        {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).first);
          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
          ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
          inst->SetOperationState(Instruction::SCALARONE, &set_base);
          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
          VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
          inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
      }
      VMEM_TO_HBM(instruction_list, result.addr + result.size(), zerohbmaddr, weight_row_now * lm_logit_weights.dims[0]);
    }

    int testnum = result.dims[1] - zero_num - 1;
    int testaddr = weightaddr + zero_num * lm_logit_weights.dims[0];
    SplitNum = testnum / weight_use_row;
    std::cout << "testaddr: " << testaddr << std::endl;
    std::cout << "testnum: " << testnum << std::endl;
    std::cout << "2 SplitNum: " << SplitNum << std::endl;
    int l = zero_num;
    for (int i = 0; i <= SplitNum; i++)
    {
      int weight_row_now = (testnum - i * weight_use_row) >= weight_use_row ? weight_use_row : (testnum - i * weight_use_row);
      if (weight_row_now == 0) break;
      data<3> soft(result.addr + result.size(), {1, weight_row_now, lm_logit_weights.dims[0]});
      HBM_TO_VMEM(instruction_list, testaddr + i * weight_use_row * lm_logit_weights.dims[0], soft.addr, lm_logit_weights.dims[0] * weight_row_now);
      data<3> leftsoft(soft.addr + soft.size(), {soft.dims[0], soft.dims[1], soft.dims[2]});
      Softmax(instruction_list, soft.addr, leftsoft.addr, soft.dims[0] * soft.dims[1], soft.dims[2]);
      std::cout << i << " soft: " << soft.addr << std::endl;
      std::cout << i << " leftsoft: " << leftsoft.addr << std::endl;
      for (int j = 0; j < weight_row_now; j++)
      {
        int batch1 = labels[l] / 1024;
        int batch2 = labels[l] % 1024;
        std::cout << "label " << l << ": " << labels[l] << std::endl;
        std::cout << "batch1: " << batch1 << std::endl;
        std::cout << "batch2: " << batch2 << std::endl;
        int softvmem = leftsoft.addr + j * lm_logit_weights.dims[0] + batch1 * 1024;
        std::cout << "softvmem: " << softvmem << std::endl;
        if (1)
        {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetValue((uint32_t)batch2).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetValue((uint32_t)batch2).first);
          VectorOperationState get_core_id(V_GET_V_CORE_ID, 0, 0, 0, 0);
          inst->SetOperationState(Instruction::VECTORONE, &get_core_id);
          VectorOperationState vmask0(V_S32_EQUAL, 0, 0, 44, 0);
          inst->SetOperationState(Instruction::VECTORTWO, &vmask0);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        if (1)
        {
          inst = new Instruction();
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(softvmem / 32).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(softvmem / 32).first);
          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
          ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
          inst->SetOperationState(Instruction::SCALARONE, &set_base);
          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
          VectorLoadOperationState vload(V_LOAD_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
          inst->SetOperationState(Instruction::VECTORLOAD, &vload);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        if (1)
        {
          inst = new Instruction();
          VectorOperationState sub(V_F32_SUBTRACTION, 0, 0, 49, 1);
          inst->SetOperationState(Instruction::VECTORTWO, &sub);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        if (1)
        {
          inst = new Instruction();
          VectorOperationState vmask0(V_SELECT_VMASK0, 0, 0, 1, 0);
          inst->SetOperationState(Instruction::VECTORONE, &vmask0);
          inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress(softvmem / 32).second);
          inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress(softvmem / 32).first);
          inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
          ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
          inst->SetOperationState(Instruction::SCALARONE, &set_base);
          inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
          inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
          VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
          inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
          CompleteInstruction(inst);
          instruction_list.push_back(inst);
        }
        l++;
      }
      float div = 1.0 / (float(testnum));
      std::cout << "div: " << div << std::endl;
      Division(inst2, leftsoft, div);
      VMEM_TO_HBM(instruction_list, leftsoft.addr, testaddr + i * weight_use_row * lm_logit_weights.dims[0], lm_logit_weights.dims[0] * weight_row_now);
    }

    if (1)
    {
      inst = new Instruction();
      VectorOperationState move(V_U32_MOVE, 0, 0, 46, 0);
      inst->SetOperationState(Instruction::VECTORONE, &move);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    int zerohbmaddr = weightaddr + (labels.size() - 1) * lm_logit_weights.dims[0];
    int zerosplit = lm_logit_weights.dims[0] / 1024;
    for (int j = 0; j < zerosplit; j++)
    {
      if (1)
      {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1, HelperGetAddress((result.addr + result.size() + j * 1024) / 32).first);
        inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, 0);
        ScalarOperationState set_base(S_U32_MOVE, 0, 0, 44, 0);
        inst->SetOperationState(Instruction::SCALARONE, &set_base);
        inst->SetImmediateValue(Instruction::IMMEDIATE4, 0);
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1);
        VectorStoreOperationState vstore(V_STORE_WITH_OFFSET, 0, 0, 1, 2, 4, 0, 0);
        inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
      }
    }
    VMEM_TO_HBM(instruction_list, result.addr + result.size(), zerohbmaddr, lm_logit_weights.dims[0]);


    // HBM_TO_VMEM(instruction_list, testaddr, 0, 2048);
    data<4> softmax_input;
    softmax_input.hbmaddr = weightaddr;
    softmax_input.dims = {1, result.dims[0], result.dims[1], lm_logit_weights.dims[0]};
    
    lm_logit_weights.hbmaddr = lm_logit_weights.addr;
    hidden_state = matmulIbWb(inst2, softmax_input, lm_logit_weights, Block_addr);

    // Halt
    if (1)
    {
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    // Test Ends
    AddNoop(10, instruction_list);
    std::cout << "instruction size: " << instruction_list.size() << std::endl;
    inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);
    for (uint32_t i = 1; i <= inst_vec.size(); i++)
    {
      std::cout << "inst_vec: " << i << std::endl;
      int leng = int(((i * 1.0) / inst_vec.size()) * 50);
      auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
      std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
      std::cout << std::string(50+2, '-') << std::endl;
      std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
      std::cout << std::string(50+2, '-') << std::endl;
      // Halt
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      inst_vec[i-1].push_back(inst);
      

      AddNoop(10, inst_vec[i-1]);

      simulator.WriteToImem(inst_vec[i-1]);
      simulator.Execute(0x7fffffff);

      for (auto it = range.first; it != range.second; it++)
      {
        if (it->second.name != "output")
        {
          float* test = new float[it->second.len];
          std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
          // load_input(test, it->second.compare_file, *(int*)(&it->second.len));
          // simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
          simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, std::to_string(z) + "/" + it->second.name);
        }
      }
    }
    instruction_list.clear();
    std::cout << "clear size: " << instruction_list.size() << std::endl;

    data<4> ln_f_in(hidden_state.addr + hidden_state.size(), {hidden_state.dims[0], hidden_state.dims[1], hidden_state.dims[2], hidden_state.dims[3]});
    HBM_TO_VMEM(instruction_list, forward_addr.at("transformer.ln_f_in"), ln_f_in.addr, ln_f_in.size());
    data<1> ln_f_weight(ln_f_in.addr + ln_f_in.size(), {4096});
    HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.weight"), ln_f_weight.addr, ln_f_weight.size());
    data<1> ln_f_bias(ln_f_weight.addr + ln_f_weight.size(), {4096});
    HBM_TO_VMEM(instruction_list, weights_addr.at("transformer.ln_f.bias"), ln_f_bias.addr, ln_f_bias.size());
    inst2.Spy("transformer.ln_f_in", hidden_state.asVMem(inst2), Train_7B_Back + "transformer.ln_f_in.txt");
    data<4> ln_f_dx = LayerNormDxBackward(inst2, ln_f_in, hidden_state, ln_f_weight, ln_f_bias.addr + ln_f_bias.size(), config.layer_norm_epsilon);
    inst2.Spy("transformer.ln_f_out", ln_f_dx.asVMem(inst2), Train_7B_Back + "transformer.ln_f_out.txt");
    data<1> ln_f_dw = LayerNormDwBackward(inst2, ln_f_in, hidden_state, ln_f_dx.addr + ln_f_dx.size(), config.layer_norm_epsilon);
    if (config.training)
    {
      data<2> ln_f_weight_new;
      ln_f_weight_new.hbmaddr = weights_addr.at("transformer.ln_f.weight");
      ln_f_weight_new.dims = {1, ln_f_weight.dims[0]};
      UpdateWeight(inst2, config.update_lr, ln_f_weight_new, ln_f_dw.as<2>(), ln_f_dw.addr + ln_f_dw.size());
    }
    
    data<1> ln_f_db = LayerNormDbBackward(inst2, hidden_state, ln_f_dw.addr + ln_f_dw.size());
    if (config.training)
    {
      data<1> ln_f_bias_new;
      ln_f_bias_new.hbmaddr = weights_addr.at("transformer.ln_f.bias");
      ln_f_bias_new.dims = {ln_f_bias.dims[0]};
      UpdateBias(inst2, config.update_lr, ln_f_bias_new, ln_f_db, ln_f_db.addr + ln_f_db.size());
    }

    data<4> block(ln_f_dx.addr + ln_f_dx.size(), {1, 1, 115, 4096});
    for (int p = 29; p >= 29; p--)
    {
      inst2.Spy(weight_name + ".h." + std::to_string(p) + "_in", ln_f_dx.asVMem(inst2), Train_7B_Back + weight_name + ".h." + std::to_string(p) + "_in.txt");
      block = BLOOMBlockBackward(inst2, config, ln_f_dx, ln_f_dx.addr + ln_f_dx.size(), ln_f_dx.addr + ln_f_dx.size(), "transformer.h." + std::to_string(p), weights_addr, forward_addr);
      inst2.Spy(weight_name + ".h." + std::to_string(p) + "_out", block.asVMem(inst2), Train_7B_Back + weight_name + ".h." + std::to_string(p) + "_out.txt");
      ln_f_dx.addr = block.addr;
    }
    // Halt
    if (1)
    {
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      instruction_list.push_back(inst);
    }
    // Test Ends
    AddNoop(10, instruction_list);
    std::cout << "instruction size: " << instruction_list.size() << std::endl;
    inst_vec = InstructionsSpilt(instruction_list, 14000, inst2.spies);
    for (uint32_t i = 1; i <= inst_vec.size(); i++)
    {
      std::cout << "inst_vec: " << i << std::endl;
      int leng = int(((i * 1.0) / inst_vec.size()) * 50);
      auto range = inst2.spies.equal_range(inst_vec[i-1][inst_vec[i-1].size() - 1]);
      std::cout << (i * 1.0 / inst_vec.size())*100 << "%\n";
      std::cout << std::string(50+2, '-') << std::endl;
      std::cout << std::string(1, '|') << std::string(leng, '=') << std::string(50-leng, ' ') << std::string(1, '|') << std::endl;
      std::cout << std::string(50+2, '-') << std::endl;
      // Halt
      inst = new Instruction();
      ScalarOperationState scalar(S_HALT, 0 /*perm*/, 0 /*s_x*/, 0 /*s_y*/, 0 /*s_dest*/);
      inst->SetOperationState(Instruction::SCALARONE, &scalar);
      CompleteInstruction(inst);
      inst_vec[i-1].push_back(inst);
      

      AddNoop(10, inst_vec[i-1]);

      simulator.WriteToImem(inst_vec[i-1]);
      simulator.Execute(0x7fffffff);

      for (auto it = range.first; it != range.second; it++)
      {
        float* test = new float[it->second.len];
        std::cout << "spies vmem " << i - 1 << ": " << it->second.addr << ", len: " << it->second.len << std::endl;
        // load_input(test, it->second.compare_file, *(int*)(&it->second.len));
        // simulator.DebugPrintVmem_dlc(it->second.addr, it->second.addr + it->second.len, test, it->second.name);
        simulator.DebugPrintVmem_Write(it->second.addr, it->second.addr + it->second.len, std::to_string(z) + "b/" + it->second.name);
      }
    }
    for (int p = 29; p >= 29; p--)
    {
      simulator.PrintHBM_Write(weights_addr.at("transformer.ln_f.weight"), weights_addr.at("transformer.ln_f.weight") + 4096, std::to_string(z) + "t/transformer.ln_f.weight");
      simulator.PrintHBM_Write(weights_addr.at("transformer.ln_f.bias"), weights_addr.at("transformer.ln_f.bias") + 4096, std::to_string(z) + "t/transformer.ln_f.bias");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".input_layernorm.weight"), weights_addr.at("transformer.h." + std::to_string(p) + ".input_layernorm.weight") + 4096, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".input_layernorm.weight");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".input_layernorm.bias"), weights_addr.at("transformer.h." + std::to_string(p) + ".input_layernorm.bias") + 4096, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".input_layernorm.bias");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".post_attention_layernorm.weight"), weights_addr.at("transformer.h." + std::to_string(p) + ".post_attention_layernorm.weight") + 4096, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".post_attention_layernorm.weight");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".post_attention_layernorm.bias"), weights_addr.at("transformer.h." + std::to_string(p) + ".post_attention_layernorm.bias") + 4096, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".post_attention_layernorm.bias");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.query_key_value.weight"), weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.query_key_value.weight") + 4096 * 4096 * 3, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".self_attention.query_key_value.weight");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.query_key_value.bias"), weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.query_key_value.bias") + 4096 * 3, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".self_attention.query_key_value.bias");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.dense.weight"), weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.dense.weight") + 4096 * 4096, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".self_attention.dense.weight");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.dense.bias"), weights_addr.at("transformer.h." + std::to_string(p) + ".self_attention.dense.bias") + 4096, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".self_attention.dense.bias");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_h_to_4h.weight"), weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_h_to_4h.weight") + 4096 * 4096 * 4, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".mlp.dense_h_to_4h.weight");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_h_to_4h.bias"), weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_h_to_4h.bias") + 4096 * 4, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".mlp.dense_h_to_4h.bias");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_4h_to_h.weight"), weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_4h_to_h.weight") + 4096 * 4096 * 4, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".mlp.dense_4h_to_h.weight");
      simulator.PrintHBM_Write(weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_4h_to_h.bias"), weights_addr.at("transformer.h." + std::to_string(p) + ".mlp.dense_4h_to_h.bias") + 4096, std::to_string(z) + "t/transformer.h." + std::to_string(p) + ".mlp.dense_4h_to_h.bias");
    }
    instruction_list.clear();
    std::cout << "clear size: " << instruction_list.size() << std::endl;
    float a = (float)(z);
    float b = (float)(numz - 1);
    config.update_lr = 0.0 + (config.update_lr_max + 0.0) / 2 * (1.0 + cos(a / b * PI));
    std::cout << "lr:" << config.update_lr << std::endl;
  }
  return ;
}


int main() 
{
#if BLOOMMLP_test
  BLOOMMLPTest();
#endif
#if BLOOMBlock_test
  BLOOMBlockTest();
#endif
#if BLOOMModel_test
  BLOOMModelTest();
#endif
#if linear_test
  linearTest();
#endif
#if build_alibi_test
  build_alibiTest();
#endif
#if _prepare_attn_mask_test
  _prepare_attn_maskTest();
#endif
#if BLOOMMLPBackward_test
  BLOOMMLPBackwardTest();
#endif
#if BLOOMBlockBackward_test
  BLOOMBlockBackwardTest();
#endif
#if BLOOMModelBackward_test
  BLOOMModelBackwardTest();
#endif
#if BLOOMForCausalLMBackward_test
  BLOOMForCausalLMBackwardTest();
#endif
#if BLOOMAttentionBackward_test
  BLOOMAttentionBackwardTest();
#endif
#if matmul_test
  matmulTest();
#endif
#if MatMulDxBackward_test
  MatMulDxBackwardTest();
#endif
#if Training_test
  TrainingTest();
#endif
#if Scheduler_test
  SchedulerTest();
#endif
#if tt_test
  ttTest();
#endif
#if dropout_test
  DropoutTest();
#endif
#if droptest 
  dropoutTest();
#endif
#if loss_test
  lossTest();
#endif
# if tt7b_test
  ttTest7B();
#endif
}