#ifndef _BLOOM_
#define _BLOOM_
#include "../torch/torch.h"
#include "../utils/utils.h"
#include <string>
#include <map>
#include <tuple>
#include <variant>
#include <nlohmann/json.hpp>
#include <filesystem>
using json = nlohmann::json;

namespace COLOR
{
const auto RED = "\033[1;31m";
const auto GREEN = "\033[1;32m";
const auto YELLOW = "\033[1;33m";
const auto BLUE = "\033[1;34m";
const auto WHITE = "\033[0;37m";
const auto CYAN = "\033[0;96m";
const auto ORANGE = "\033[38;2;255;165;0m";
const auto MIKU = "\033[38;2;147;214;214m";
const auto AYUMU = "\033[38;2;237;125;149m";
const auto KANON = "\033[38;2;255;127;39m";
const auto SETSUNA = "\033[38;2;216;28;47m";
const auto SHIORI = "\033[38;2;55;180;132m";
const auto CYARON = "\033[38;2;255;164;52m";
} // namespace COLOR

struct vec {
  vec2d_t input;
  vec1d_t_i label;
  vec2d_t_i attention_mask;
};

template<uint32_t N>
struct data {
  std::string vector_type;
  uint32_t dim;

  vec5d_t v5d;
  vec4d_t v4d;
  vec3d_t v3d;
  vec2d_t v2d;
  vec1d_t v1d;

  vec5d_t_i v5d_i;
  vec4d_t_i v4d_i;
  vec3d_t_i v3d_i;
  vec2d_t_i v2d_i;
  vec1d_t_i v1d_i;

  data(uint32_t n, std::vector<int> &dims, std::string data_type = "f") {
    vector_type = "vec" + std::to_string(n) + data_type;
    if(data_type == "f") {
      switch (n)
      {
      case 1:
        v1d = vec1d_t(dims[0], 0);
        break;
      case 2:
        v2d = vec2d_t(dims[0], vec1d_t(dims[1], 0));
        break;
      case 3:
        v3d = vec3d_t(dims[0], vec2d_t(dims[1], vec1d_t(dims[2], 0)));
        break;
      case 4:
        v4d = vec4d_t(dims[0], vec3d_t(dims[1], vec2d_t(dims[2], vec1d_t(dims[3], 0))));
        break;
      case 5:
        v5d = vec5d_t(dims[0], vec4d_t(dims[1], vec3d_t(dims[2], vec2d_t(dims[3], vec1d_t(dims[4], 0)))));
        break;
      default:
        break;
      }
    }

    if(data_type == "i") {
      switch (n)
      {
      case 1:
        v1d_i = vec1d_t_i(dims[0], 0);
        break;
      case 2:
        v2d_i = vec2d_t_i(dims[0], vec1d_t_i(dims[1], 0));
        break;
      case 3:
        v3d_i = vec3d_t_i(dims[0], vec2d_t_i(dims[1], vec1d_t_i(dims[2], 0)));
        break;
      case 4:
        v4d_i = vec4d_t_i(dims[0], vec3d_t_i(dims[1], vec2d_t_i(dims[2], vec1d_t_i(dims[3], 0))));
        break;
      case 5:
        v5d_i = vec5d_t_i(dims[0], vec4d_t_i(dims[1], vec3d_t_i(dims[2], vec2d_t_i(dims[3], vec1d_t_i(dims[4], 0)))));
        break;
      default:
        break;
      }
    }
  }

  data(){
    vector_type = "none";
    dim = 0;
  }
};

extern std::map<std::string, data<3>> Map;
extern std::map<std::string, data<2>> WeightMap;
extern std::map<std::string, data<1>> ForwardMap;

extern std::string TRAINER_STATE_NAME;

struct BLOOMConfig
{
    uint32_t num_head;
    uint32_t vocab_size = 250880;
    uint32_t hidden_size=4096;
    uint32_t n_layer=30;
    uint32_t n_head=32;
    float layer_norm_epsilon=1e-5;
    uint32_t initializer_range=0.02;
    bool use_cache = true;
    bool training = true;
    bool apply_residual_connection_post_layernorm = false;
    float hidden_dropout=0.0;
    float attention_dropout=0.0;
    std::string activation_function = "gelu_new";  
    float learning_rate = 0.00002;
    float learning_rate_max = 0.00002;
    uint32_t pretraining_tp = 1;
    bool slow_but_exact = false;
};

struct BLOOMTraining 
{
  int model_max_length;
  std::string data_path;
  std::string  output_dir;
  bool bf16;
  int num_train_epochs;
  int per_device_train_batch_size;
  int per_device_eval_batch_size;
  int gradient_accumulation_steps;
  std::string save_strategy;
  int save_steps;
  std::string evaluation_strategy;
  int save_total_limit;
  float learning_rate;
  float weight_decay;
  float warmup_ratio;
  std::string lr_scheduler_type;
  float logging_steps;
  std::string fsdp;
  std::string fsdp_transformer_layer_cls_to_wrap;
  bool tf32;
  bool gradient_checkpointing; 
  bool resume_from_checkpoint = false;
  int max_steps = -1;
  int train_dataloader_size = 1;
  int eval_steps = 0;
  int local_process_index = 0;
  int process_index = 0;
  bool ignore_data_skip = false;
  bool logging_nan_inf_filter = false;
};

class TrainerState
{
  public:
    TrainerState()
      : epoch(0.0), global_step(0), max_steps(0), logging_steps(500), 
        eval_steps(500), save_steps(500), num_train_epochs(0), total_flos(0.0),
        is_local_process_zero(true), is_world_process_zero(true), is_hyper_param_search(false),
        trial_name("") 
      {}
    
    void save_to_json(const std::string& json_path) 
    {
      std::ofstream output(json_path);
      nlohmann::json j;
      j = nlohmann::json{
        { "epoch", this->epoch },
        { "global_step", this->global_step },
        { "max_steps", this->max_steps },
        { "logging_steps", this->logging_steps },
        { "eval_steps", this->eval_steps },
        { "save_steps", this->save_steps },
        { "num_train_epochs", this->num_train_epochs },
        { "total_flos", this->total_flos },
        // { "log_history", this->log_history },
        { "best_model_checkpoint", this->best_model_checkpoint },
        { "is_local_process_zero", this->is_local_process_zero },
        { "is_world_process_zero", this->is_world_process_zero },
        { "is_hyper_param_search", this->is_hyper_param_search },
        { "trial_name", this->trial_name },
        // { "trial_params", this->trial_params }
      };
      output << j;
    }

    static TrainerState load_from_json(const std::string& json_path)
    {
      std::ifstream input(json_path);
      nlohmann::json j;
      input >> j;
      TrainerState ts;
      j.at("epoch").get_to(ts.epoch);
      j.at("global_step").get_to(ts.global_step);
      j.at("max_steps").get_to(ts.max_steps);
      j.at("logging_steps").get_to(ts.logging_steps);
      j.at("eval_steps").get_to(ts.eval_steps);
      j.at("save_steps").get_to(ts.save_steps);
      j.at("num_train_epochs").get_to(ts.num_train_epochs);
      j.at("total_flos").get_to(ts.total_flos);
      // j.at("log_history").get_to(ts.log_history);
      j.at("best_model_checkpoint").get_to(ts.best_model_checkpoint);
      j.at("is_local_process_zero").get_to(ts.is_local_process_zero);
      j.at("is_world_process_zero").get_to(ts.is_world_process_zero);
      j.at("is_hyper_param_search").get_to(ts.is_hyper_param_search);
      j.at("trial_name").get_to(ts.trial_name);
      // j.at("trial_params").get_to(ts.trial_params);
      return ts;
    }

    int epoch;
    int global_step;
    int max_steps;
    int logging_steps;
    int eval_steps;
    int save_steps;
    int num_train_epochs;
    float total_flos;
    std::map<std::string, float> log_history;
    std::string best_model_checkpoint;
    bool is_local_process_zero;
    bool is_world_process_zero;
    bool is_hyper_param_search;
    std::string trial_name;
    std::map<std::string, std::variant<std::string, float, int, bool>> trial_params;
};

void fillWightMapFromFile1(std::string name, int length);
void fillWightMapFromFile2(std::string name, int hight, int width);

void load_input(vec4d_t &input, std::string path, int num);
void load_input(vec3d_t &input, std::string path, int num);
void load_input(vec2d_t &input, std::string path, int num);
void load_input(vec1d_t &input, std::string path, int num);

void Error(vec4d_t input, std::string str);
void Error_FOR(vec4d_t input, std::string str);
void Error_index(vec4d_t input, std::string str);

void WriteTensor(vec4d_t input, std::string str);

void UpdateWeight(BLOOMConfig config, vec2d_t& weight, vec2d_t grad);

void UpdateWeight(BLOOMConfig config, vec1d_t& weight, vec1d_t grad);

void UpdateBias(BLOOMConfig config, vec1d_t& bias, vec1d_t grad);

vec3d_t BLOOMMLP(BLOOMConfig config, vec3d_t hidden_states, vec3d_t residual, std::string weights_path);

std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>, vec4d_t> BLOOMAttention(BLOOMConfig config, vec3d_t hiddenstates, vec3d_t residual, vec3d_t alibi, 
                                                                vec4d_t_i attention_mask, std::tuple<vec3d_t, vec3d_t> layer_past,
                                                                vec4d_t_i head_mask, bool use_cache, bool output_attentions, std::string path); 

std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>, vec4d_t> BLOOMBlock(BLOOMConfig config, 
                                                vec3d_t hidden_states, vec3d_t alibi, vec4d_t_i attention_mask, 
                                                std::tuple<vec3d_t, vec3d_t> layer_past, vec4d_t_i head_mask, 
                                                bool use_cache, bool output_attentions, std::string weights_path);

std::tuple<vec3d_t, std::vector<std::tuple<vec3d_t, vec3d_t>>, std::vector<vec3d_t>, std::vector<vec4d_t>> 
BLOOMModel(BLOOMConfig config, vec2d_t input_ids, std::vector<std::tuple<vec3d_t, vec3d_t>> past_key_values, 
          vec2d_t_i attention_mask, vec5d_t_i head_mask, vec3d_t inputs_embeds, 
          bool use_cache, bool output_attentions, bool output_hidden_states, bool return_dict,
          std::string weights_path); 

std::tuple<vec1d_t, vec3d_t, std::vector<std::tuple<vec3d_t, vec3d_t>>, std::vector<vec3d_t>, std::vector<vec4d_t>>
BloomForCausalLM(BLOOMConfig config, vec2d_t input_ids, std::vector<std::tuple<vec3d_t, vec3d_t>> past_key_values, vec2d_t_i attention_mask, 
                vec5d_t_i head_mask, vec3d_t inputs_embeds, vec2d_t_i labels, bool use_cache, bool output_attentions, bool output_hidden_states, bool return_dict,
                std::string weights_path);

vec3d_t BLOOMAttentionBackward(BLOOMConfig config, vec3d_t backward_input, bool use_cache, bool output_attentions, std::string weight_bias_name);

vec3d_t BLOOMMLPBackward(BLOOMConfig config, vec3d_t backward_input, std::string weights_path);

vec4d_t BLOOMBlockBackward(BLOOMConfig config, vec4d_t dx, std::string weights_path);

vec4d_t BLOOMModelBackward(BLOOMConfig config, vec4d_t lm_logits, std::string weights_path);

vec4d_t BLOOMForCausalLMBackward(BLOOMConfig config, vec4d_t hidden_states, std::string weights_path);

float BLOOMTrainingStep(BLOOMConfig config, BLOOMTraining training, vec2d_t inputs, vec1d_t_i labels, vec2d_t_i attention_mask);

float BLOOMTrain(BLOOMConfig config, BLOOMTraining training, std::vector<vec> train_dataloader);
#endif