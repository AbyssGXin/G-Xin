#include "../Data.h"
#include "../GPT2Define.h"
#include "../utils/utils.h"
typedef Inst2 INST_TYPE;

data<4> Tanh(INST_TYPE &inst2, data<4> input, uint32_t output_addr);

data<4> Cosh(INST_TYPE &inst2, data<4> input, uint32_t output_addr);

data<4> NewGELUActivation(INST_TYPE &inst2, data<4> input, uint32_t output_addr);

// data<4> Addmm(std::vector<Instruction *> &instruction_list, data<4> mat_1, data<4> mat_2, data<1> bias, uint32_t output_addr, uint32_t beta = 1, uint32_t alpha = 1);

data<4> matmul(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulT(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulIvWv(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);
data<4> matmulIvWv(Inst2& inst2, data<4> input, data<4> weights, uint32_t output_addr);

data<4> matmul_v2(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulT_v2(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulIvWv_v2(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmul_v3(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulT_v3(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulIvWv_v3(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulIvWv_v3_(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> matmulIbWb(Inst2& inst2, data<4> input, data<2> weights, uint32_t output_addr);

data<4> Addmm(INST_TYPE &inst2,
                          const data<4> &hidden_states,
                          uint32_t hbmWeightAddr,
                    const std::array<uint32_t, 2> &weightSize,
                          uint32_t hbmBiasAddr,
                    uint32_t biasSize,
                    uint32_t output_addr,
                    uint32_t beta = 1,
                    uint32_t alpha = 1);

data<4> Conv1D(INST_TYPE &inst2,
                             const data<3> &hidden_states,
               const data<2> weight,
               const data<1> bias,
                 uint32_t output_addr);

data<3> LayerNorm_spare(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps = 0.000001);

data<3> LayerNorm_V1(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps = 0.000001);

data<3> LayerNorm_V2(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps = 0.000001);

data<3> LayerNorm_V3(INST_TYPE &inst2, data<3> input, data<1> weights, data<1> bias, uint32_t output_addr,float eps = 0.000001);

data<3> LayerNorm(INST_TYPE &inst2,
               data<3> hidden_states,
               data<1> weights,
               data<1> bias,
               uint32_t output_addr,
               float eps = 0.000001);

data<4> NewGELUActivationBackward(INST_TYPE &inst2, data<4> forward, data<4> backward, uint32_t output_addr);

data<4> LayerNormDxBackward(INST_TYPE &inst2, data<4> forward_input, data<4> backward_input, data<1> weights, uint32_t output_addr, float eps = 0.000001);

data<1> LayerNormDwBackward(INST_TYPE &inst2, data<4> forward_input, data<4> backward_input, uint32_t output_addr, float eps = 0.000001);

data<1> LayerNormDbBackward(INST_TYPE &inst2, data<4> backward_input, uint32_t output_addr);

data<4> Conv1DDxBackward(INST_TYPE &inst2, data<2> weight, data<4> backward_input, uint32_t backward_output_addr);

data<4>
Conv1DBackDx(INST_TYPE &inst2,
             data<2> weight,
             data<4> backward_input,
             uint32_t backward_output_addr);

data<4>
Conv1DBackDw(INST_TYPE &inst2,
             data<4> forward_input,
             data<4> backward_input,
             uint32_t backward_output_addr);

data<1>
Conv1DBackDb(INST_TYPE &inst2,
             data<4> dx,
             uint32_t output_addr);

data<4>
MatMulDxBackward(INST_TYPE &inst2,
                 data<4> forward_input,
                 data<4> backward_input,
                 uint32_t backward_output_addr);

data<4>
MatMulDwBackward(INST_TYPE &inst2,
                 data<4> forward_input,
                 data<4> backward_input,
                 uint32_t backward_output_addr);
                                
data<4>
SoftmaxBack(INST_TYPE &inst2,
            const data<4> &forward_output,
            const data<4> &backword_input,
            uint32_t output_addr);

void UpdateWeight(INST_TYPE &inst2,
                  float lr,
                  data<2> weight, // weight.addr为hbm地址
                  data<2> weight_grad,
                  uint32_t addr); // addr后的地址为可操作空间

void UpdateBias(INST_TYPE &inst2,
                float lr,
                data<1> bias,
                data<1> bias_grad,
                uint32_t addr);

data<3> linearAddVector(INST_TYPE &inst2, data<3> input1, data<1> input2, uint32_t output);

data<3> linear(INST_TYPE &inst2, data<3> input, data<2> weights, data<1> bias, int output_addr);
data<3> linearNobias(INST_TYPE &inst2, data<3> input, data<2> weights, int output_addr);

data<3> MatmulBackDx(INST_TYPE &inst2, data<3> input, data<2> weights, int output_addr);

data<3> linearBackDx(INST_TYPE &inst2, data<3> input, data<2> weights, int output_addr);

// data<3> linearBackDw(INST_TYPE &inst2, data<3> input, data<2> wegihts, int output_addr);

void build_alibi(INST_TYPE &inst2, data<2> input, uint32_t n_head, uint32_t output_addr);

data<4> _prepare_attn_mask(INST_TYPE &inst2, data<2> attention_mask, std::tuple<uint32_t, uint32_t> input_shape, uint32_t past_key_values_length, uint32_t output_addr);

data<3> baddbmm(INST_TYPE &inst2, data<3> input, data<3> batch1, data<3> batch2, float beta, float alpha, int output_addr);

data<4> maskedFill(INST_TYPE &inst2, data<4> scores, data<4> mask, float value, int output_addr);

void MatMulUpdateWeight(Inst2& inst2, data<3> forward_input, data<3> backward_input, data<2> weight, float update_lr, uint32_t output_addr);
void Division(Inst2& inst2, data<3> input, float num);

data<3> dropout(INST_TYPE inst2, data<3> input, float p, uint32_t output_addr);

data<3> dropout_add(INST_TYPE inst2, BLOOMConfig config, data<3> input1, data<3> input2, uint32_t output_addr);

void HBM_TO_SMEM(std::vector<Instruction*> &instruction_list, uint32_t input_addr, uint32_t dest_addr, uint32_t length, bool is_sync = true);