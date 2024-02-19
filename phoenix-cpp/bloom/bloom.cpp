#include "bloom.h"
#include <fstream>
#include <iomanip>
// using namespace std;

std::string input_start = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\nHuman: <s>";
std::string input_end = "</s>Assistant: <s>";


std::string PATH_BACK = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/";
std::string PATH_FOR = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/";
std::string PATH = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/training/";
std::string PATH_TENSOR = "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test_data_f32/";
std::string PATH_TEST = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32_nodrop/";
std::string PATHBACK = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_backward_f32/";
std::string PATHUPDATE = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/training/grad/";
std::string PATHWeight = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/training/";
std::string PATHGRAD = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_grad/";


std::map<std::string, data<3>> Map;
std::map<std::string, data<2>> WeightMap;
std::map<std::string, data<1>> ForwardMap;

std::string TRAINER_STATE_NAME = "trainer_state.json";

bool berr = false;
bool tran = false;
bool write = false;
bool update = false;
bool biasupdate = false;
bool zero_grad = true;

int bloomsplit = 0;

void fillWightMapFromFile1(std::string name, int length)
{
  std::string path = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/" + name + ".txt";
  std::cout << "load: " << path << std::endl;
  std::ifstream in(path);
  assert(in);
  std::string line;
  vec1d_t input(length, 0.0);
  for (int i = 0; i < length; i++)
  {
    std::getline(in, line);
    assert(line.length() > 0);
    auto temp = atof(line.c_str());
    input[i] = temp;
  }
  in.close();
  data<2> d;
  d.v1d = input;
  WeightMap[name] = d;
  return ;
}


void fillWightMapFromFile2(std::string name, int hight, int width)
{
  std::string path = "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/" + name + ".txt";
  std::cout << "load: " << path << std::endl;
  std::ifstream in(path);
  assert(in);
  std::string line;
  vec2d_t input(hight, vec1d_t(width, 0.0));
  for (int i = 0; i < hight; i++)
  {
    for (int j = 0; j < width; j++)
    {
      std::getline(in, line);
      assert(line.length() > 0);
      auto temp = atof(line.c_str());
      input[i][j] = temp;
    }
  }
  in.close();
  data<2> d;
  d.v2d = input;
  WeightMap[name] = d;
  return ;
}


void load_input(vec4d_t &input, std::string path, int num)
{
    std::ifstream in(path);
    std::cout << "load: " << path << std::endl;
    assert(in);
    std::string line;
    // int index = 0;
    for (int i = 0; i < input.size(); i++)
    {
        for (int j = 0; j < input[0].size(); j++)
        {
            for (int n = 0; n < input[0][0].size(); n++)
            {
                for (int m = 0; m < input[0][0][0].size(); m++)
                {
                    std::getline(in, line);
                    assert(line.length() > 0);
                    auto temp = atof(line.c_str());
                    // float x = *(float *)(&temp);
                    // std::cout << index++ << ": " << temp << std::endl;
                    input[i][j][n][m] = temp;
                }
            }
        }
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}

void load_input(vec2d_t &input, std::string path, int num)
{
    std::ifstream in(path);
    std::cout << "load: " << path << std::endl;
    assert(in);
    std::string line;
    // int index = 0;
    for (int i = 0; i < input.size(); i++)
    {
        for (int j = 0; j < input[0].size(); j++)
        {
            std::getline(in, line);
            assert(line.length() > 0);
            auto temp = atof(line.c_str());
            // float x = *(float *)(&temp);
            // std::cout << index++ << ": " << temp << std::endl;
            input[i][j] = temp;
        }
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}

void load_input(vec1d_t &input, std::string path, int num)
{
    std::ifstream in(path);
    std::cout << "load: " << path << std::endl;
    assert(in);
    std::string line;
    // int index = 0;
    for (int i = 0; i < input.size(); i++)
    {
        std::getline(in, line);
        // std::cout << "line: " << line << std::endl;
        assert(line.length() > 0);
        auto temp = atof(line.c_str());
        // float x = *(float *)(&temp);
        // std::cout << index++ << ": " << temp << std::endl;
        input[i] = temp;
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}

void load_input(vec3d_t &input, std::string path, int num)
{
    std::ifstream in(path);
    std::cout << "load: " << path << std::endl;
    assert(in);
    std::string line;
    // int index = 0;
    for (int i = 0; i < input.size(); i++)
    {
        for (int j = 0; j < input[0].size(); j++)
        {
            for (int n = 0; n < input[0][0].size(); n++)
            {
                std::getline(in, line);
                assert(line.length() > 0);
                auto temp = atof(line.c_str());
                // float x = *(float *)(&temp);
                // std::cout << index++ << ": " << temp << std::endl;
                input[i][j][n] = temp;
            }
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}




void UpdateWeight(BLOOMConfig config, vec2d_t& weight, vec2d_t grad)
{
    float eps = 0.00000001;
    if (!config.training) return;
    std::cout << "Update weight " << weight.size() << ' ' << weight[0].size() <<std::endl;
    for (int i = 0; i < weight.size(); i++) {
        for (int j = 0; j < weight[0].size(); j++) {
          float lr = config.learning_rate * std::sqrt(1 - 0.999) / (1 - 0.9);
          float m = (1 - 0.9) * grad[i][j];
          float v = (1 - 0.999) * grad[i][j] * grad[i][j];
          weight[i][j] -= ((lr * m) / (std::sqrt(v) + eps));

        }
    }
}

void UpdateWeight(BLOOMConfig config, vec1d_t& weight, vec1d_t grad)
{
    float eps = 0.00000001;
    if (!config.training) return;
    std::cout << "Update weight " << weight.size() << std::endl;
    for (int i = 0; i < weight.size(); i++) {
      float lr = config.learning_rate * std::sqrt(1 - 0.999) / (1 - 0.9);
      float m = (1 - 0.9) * grad[i];
      float v = (1 - 0.999) * grad[i] * grad[i];
      weight[i] -= ((lr * m) / (std::sqrt(v) + eps));
    }
}

void UpdateBias(BLOOMConfig config, vec1d_t& bias, vec1d_t grad)
{
    float eps = 0.00000001;
    if (!config.training) return;
    std::cout << "Update bias " << bias.size() << std::endl;
    for (int i = 0; i < bias.size(); i++){
      float lr = config.learning_rate * std::sqrt(1 - 0.999) / (1 - 0.9);
      float m = (1 - 0.9) * grad[i];
      float v = (1 - 0.999) * grad[i] * grad[i];
      bias[i] -= ((lr * m) / (std::sqrt(v) + eps));
    }
}

void Error(vec4d_t input, std::string str)
{
    float res = 0.0;
    float cnt = 0.0;
    float maxv = 0.0;
    float max_coor = 0;
    std::cout << "load: " << str << std::endl;
    std::ifstream in(str);
    assert(in);
    std::ofstream write("Err.txt", std::ios::app);
    std::string line;
    for (auto x_1 : input)
    {
        for (auto x_2 : x_1)
        {
            for (auto x_3 : x_2)
            {
                for (auto x_4 : x_3)
                {
                    std::getline(in, line);
                    auto temp = atof(line.c_str());
                    if (temp != 0.0)
                    {
                        float kk = std::fabs((temp - x_4) / temp);
                        res += kk;
                        if (kk > maxv)
                        {
                            maxv = kk;
                        }
                        cnt++;
                    }
                }
            }
        }
    }
    std::string s1 = "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/";
    std::cout << str.erase(0, s1.length()) << ": " << std::endl;
    std::cout << "          平均误差: " <<  res / cnt * 100  << "%" << std::endl;
    std::cout << "          最大误差: " <<  maxv * 100 << "%" << std::endl;
    write << str << "   " << res / cnt * 100 << "%" << "   " << maxv * 100 << "%" << std::endl;
    write.close();
    // std::cout << '-' * 50 << std::endl;
    return ;
}

void Error_FOR(vec4d_t input, std::string str)
{
    float res = 0.0;
    float cnt = 0.0;
    float maxv = 0.0;
    float max_coor = 0;
    std::cout << "load: " << str << std::endl;
    std::ifstream in(str);
    assert(in);
    std::ofstream write("Err.txt", std::ios::app);
    std::string line;
    for (auto x_1 : input)
    {
        for (auto x_2 : x_1)
        {
            for (auto x_3 : x_2)
            {
                for (auto x_4 : x_3)
                {
                    std::getline(in, line);
                    auto temp = atof(line.c_str());
                    if (temp != 0.0)
                    {
                        float kk = std::fabs((temp - x_4) / temp);
                        // std::cout << "误差: " << kk * 100 << "%" << std::endl;
                        res += kk;
                        if (kk > maxv)
                        {
                            maxv = kk;
                        }
                        cnt++;
                    }
                }
            }
        }
    }
    std::string s1 = "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32/";
    std::cout << str.erase(0, s1.length()) << ": " << std::endl;
    std::cout << "          平均误差: " <<  res / cnt * 100  << "%" << std::endl;
    std::cout << "          最大误差: " <<  maxv * 100 << "%" << std::endl;
    write << str << "   " << res / cnt * 100 << "%" << "   " << maxv * 100 << "%" << std::endl;
    write.close();
    // std::cout << '-' * 50 << std::endl;
    return ;
}

void Error_grad(vec2d_t input, std::string str)
{
    float res = 0.0;
    float cnt = 0.0;
    float maxv = 0.0;
    float max_coor = 0;
    std::cout << "load: " << str << std::endl;
    std::ifstream in(str);
    assert(in);
    std::ofstream write("Err.txt", std::ios::app);
    std::string line;
    for (auto x_1 : input)
    {
        for (auto x_2 : x_1)
        {
          std::getline(in, line);
          auto temp = atof(line.c_str());
          if (temp != 0.0)
          {
              float kk = std::fabs((temp - x_2) / temp);
              // std::cout << "误差: " << kk * 100 << "%" << std::endl;
              res += kk;
              if (kk > maxv)
              {
                  maxv = kk;
              }
              cnt++;
            }
        }
    }
    std::string s1 = "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_backward_grad/";
    std::cout << str.erase(0, s1.length()) << ": " << std::endl;
    std::cout << "          平均误差: " <<  res / cnt * 100  << "%" << std::endl;
    std::cout << "          最大误差: " <<  maxv * 100 << "%" << std::endl;
    write << str << "   " << res / cnt * 100 << "%" << "   " << maxv * 100 << "%" << std::endl;
    write.close();
    // std::cout << '-' * 50 << std::endl;
    return ;
}

void Error_index(vec4d_t input, std::string str)
{
    float res = 0.0;
    float cnt = 0.0;
    float maxv = 0.0;
    float max_coor = 0;
    std::ifstream in(str);
    assert(in);
    // std::cout << "load: " << str << std::endl;
    std::string line;
    for (uint32_t i = 0; i < input.size(); i++)
    {
        for (uint32_t j = 0; j < input[0].size(); j++)
        {
            for (uint32_t k = 0; k < input[0][0].size(); k++)
            {
                for (uint32_t l = 0; l < input[0][0][0].size(); l++)
                {
                    std::getline(in, line);
                    auto temp = atof(line.c_str());
                    if (temp != 0.0)
                    {
                        float kk = std::fabs((temp - input[i][j][k][l]) / temp);
                        // std::cout << "误差: " << kk * 100 << "%" << std::endl;
                        res += kk;
                        if (kk > 0.01)
                        {
                          std::cout << "["<< i << " " << j << " " << k << " " << l << "]      " << "[python: " << temp << ", Cpp: " << input[i][j][k][l] << "]    误差: " << kk * 100 << "%" << std::endl;
                        }
                        if (kk > maxv)
                        {
                            maxv = kk;
                        }
                        cnt++;
                    }
                }
            }
        }
    }
    std::string s1 = "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_forward_f16_mo/";
    std::cout << str.erase(0, s1.length()) << ": " << std::endl;
    std::cout << "          平均误差: " <<  res / cnt * 100  << "%" << std::endl;
    std::cout << "          最大误差: " <<  maxv * 100 << "%" << std::endl;
    // std::cout << '-' * 50 << std::endl;
    return ;
}

void WriteTensor(vec4d_t input, std::string str)
{
    std::ofstream outputFile(str);
    if (outputFile.is_open())
    {
        for (auto x_1 : input)
        {
            for (auto x_2 : x_1)
            {
                for (auto x_3 : x_2)
                {
                    for (auto x_4 : x_3)
                    {
                        outputFile << std::fixed << std::setprecision(80) << x_4 << "\n";
                    }
                }
            }
        }
        outputFile.close();
        std::cout << "数据已成功写入文件。" << std::endl;
    }
    else 
    {
        std::cout << "无法打开输出文件。" << std::endl;
    }
    return ;
}

void WriteTensor(vec2d_t input, std::string str)
{
    std::ofstream outputFile(str);
    if (outputFile.is_open())
    {
        for (auto x_1 : input)
        {
            for (auto x_2 : x_1)
            {
              outputFile << std::fixed << std::setprecision(80) << x_2 << "\n";
            }
        }
        outputFile.close();
        std::cout << "数据已成功写入文件。" << std::endl;
    }
    else 
    {
        std::cout << "无法打开输出文件。" << std::endl;
    }
    return ;
}

void WriteTensor(vec1d_t input, std::string str)
{
    std::ofstream outputFile(str);
    if (outputFile.is_open())
    {
        for (auto x_1 : input)
        {
          outputFile << std::fixed << std::setprecision(80) << x_1 << "\n";
        }
        outputFile.close();
        std::cout << "数据已成功写入文件。" << std::endl;
    }
    else 
    {
        std::cout << "无法打开输出文件。" << std::endl;
    }
    return ;
}

vec5d_t _split_heads(vec3d_t &fused_qkv, BLOOMConfig config) {
  int batch_size = fused_qkv.size();
  int seq_length = fused_qkv[0].size();
  int head_dim = fused_qkv[0][0].size();
  vec5d_t result(3, vec4d_t(batch_size, vec3d_t(seq_length, vec2d_t(config.n_head, vec1d_t(config.hidden_size / config.n_head, 0.0)))));

  
  for (int b = 0; b < batch_size; b++) {
    for(int s = 0; s < seq_length; s++) {
      for (int h = 0; h < config.n_head; h++) {
        for (int d = 0; d < config.hidden_size / config.n_head; d++) {
          for (int i = 0; i < 3; i++) {
            result[i][b][s][h][d] = fused_qkv[b][s][h * (config.hidden_size / config.n_head) * 3 + i * (config.hidden_size / config.n_head) + d];
          }
        }
      }
    }
  }

  return result;
}

vec4d_t_i _expand_mask(vec2d_t_i mask, int tgt_length) {
  int batch_size = mask.size(), src_length = mask[0].size();
  vec4d_t_i expanded_mask(batch_size, vec3d_t_i(1, vec2d_t_i(tgt_length, vec1d_t_i(src_length))));
  
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < tgt_length; j++) {
      expanded_mask[i][0][j] = mask[i];
    }
  }

  return expanded_mask;
}

vec4d_t_i _make_causal_mask(std::tuple<int, int> input_shape, int past_key_values_length) {
  int batch_size = std::get<0>(input_shape), target_length = std::get<1>(input_shape);

  vec2d_t_i mask(target_length, vec1d_t_i(target_length + past_key_values_length, 0));

  for (int i = 0; i < target_length; i++) {
    for (int j = 0; j < target_length; j++) {
      mask[i][j + past_key_values_length] = i < j;
    }
  }

  if(past_key_values_length > 0) {
    for(int i = 0; i < target_length; i++) {
      for (int j = 0; j < past_key_values_length; j++)
      {
        mask[i][j] = 0;
      }
    }
  }

  vec4d_t_i expanded_mask(batch_size, vec3d_t_i(1, vec2d_t_i(target_length, vec1d_t_i(target_length + past_key_values_length))));

  for (int i = 0; i < batch_size; i++)
  {
    expanded_mask[i][0] = mask;
  }

  return expanded_mask;
}

vec3d_t _merge_heads(vec3d_t x, BLOOMConfig config) {
  vec3d_t output;

  int batch_size_and_num_heads = x.size();
  int seq_length = x[0].size();
  int batch_size = batch_size_and_num_heads / config.n_head;

  vec4d_t temp = view(vec4d_t(1, x), batch_size, config.n_head, seq_length, config.hidden_size / config.n_head);
  temp = permute(temp, 0, 2, 1, 3);
  output = reshape(temp, batch_size, seq_length, config.hidden_size);

  return output;
}

vec4d_t_i _prepare_attn_mask(vec2d_t_i attention_mask, std::tuple<int, int> input_shape, int past_key_values_length) {
  int src_length = std::get<1>(input_shape);
  vec4d_t_i combined_attention_mask;

  if(src_length > 1) {
    combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length);
    return combined_attention_mask;
  }

  auto expanded_attention_mask = _expand_mask(attention_mask, src_length);

  if(combined_attention_mask.size() == 0) {
    combined_attention_mask = expanded_attention_mask;
  }
  else{
    for (int i = 0; i < expanded_attention_mask.size(); i++) {
      for (int j = 0; j < expanded_attention_mask[0].size(); j++) {
        for (int m = 0; m < expanded_attention_mask[0][0].size(); m++) {
          for (int n = 0; n < expanded_attention_mask[0][0][0].size(); n++) {
            combined_attention_mask[i][j][m][n] &= expanded_attention_mask[i][j][m][n];
          }
        }
      }
    }
  }

  return combined_attention_mask;
}



vec3d_t _merge_heads_backward(vec3d_t x, BLOOMConfig config) {
  vec4d_t temp;
  vec3d_t output;

  int batch_size = x.size();
  int seq_length = x[0].size();
  std::cout << "batch_size: " << batch_size << ", seq_length: " << seq_length << ", config.n_head: " << config.n_head << ", config.hidden_size: " << config.hidden_size / config.n_head << std::endl;
  temp = reshape(x, batch_size, seq_length, config.n_head, config.hidden_size / config.n_head);
  // std::cout << "1----------------------" << std::endl;
  temp = permute(temp, 0, 2, 1, 3);
  // std::cout << "2----------------------" << std::endl;
  // temp = view(temp, batch_size, config.n_head, seq_length, config.hidden_size / config.n_head);
  output = reshape(temp, batch_size * config.n_head, seq_length, config.hidden_size / config.n_head);

  return output;
}

vec3d_t BLOOMMLP(BLOOMConfig config, vec3d_t hidden_states, vec3d_t residual, std::string weights_path)
{
    uint32_t hidden_size = config.hidden_size;
    std::map<std::string, vec4d_t(*)(vec4d_t)> ACT2FN{ {"gelu_new", NewGELUActivation} };
    
    vec2d_t lm_weights_1(4 * hidden_size, vec1d_t(hidden_size));
    vec1d_t lm_bias_1(4 * hidden_size);

    if (config.training)
    {
      std::string name = weights_path + ".dense_h_to_4h.weight";
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      lm_weights_1 = WeightMap[name].v2d;

      name = weights_path + ".dense_h_to_4h.bias";
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      lm_bias_1 = WeightMap[name].v1d;
    }
    else
    {
      LoadInput(lm_weights_1, PATH + weights_path + ".dense_h_to_4h.weight.txt");
      LoadInput(lm_bias_1, PATH + weights_path + ".dense_h_to_4h.bias.txt");
    }    

    if (config.training)
    {
      std::string name = weights_path + ".dense_h_to_4h_in";
      std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
      ForwardMap[name].v4d = vec4d_t(1, hidden_states);
    }

    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".dense_h_to_4h_in.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".dense_h_to_4h_in.txt"); 
    hidden_states = Linear(vec4d_t(1, hidden_states), lm_weights_1, lm_bias_1)[0];
    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".dense_h_to_4h_out.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".dense_h_to_4h_out.txt");  
    if (config.training)
    {
      std::string name = weights_path + ".gelu_impl_in";
      std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
      ForwardMap[name].v4d = vec4d_t(1, hidden_states);
    }
    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".gelu_impl_in.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".gelu_impl_in.txt");
    hidden_states = ACT2FN[config.activation_function](vec4d_t(1, hidden_states))[0];
    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".gelu_impl_out.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".gelu_impl_out.txt"); 
    
    if (config.training)
    {
      std::string name = weights_path + ".dense_4h_to_h_in";
      std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
      ForwardMap[name].v4d = vec4d_t(1, hidden_states);
    }

    vec2d_t lm_weights_2(hidden_size, vec1d_t(4 * hidden_size));
    vec1d_t lm_bias_2(hidden_size);

    if(config.training)
    {
      std::string name = weights_path + ".dense_4h_to_h.weight";
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      lm_weights_2 = WeightMap[name].v2d;

      name = weights_path + ".dense_4h_to_h.bias";
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      lm_bias_2 = WeightMap[name].v1d;
    }
    else 
    {
      LoadInput(lm_weights_2, PATH + weights_path + ".dense_4h_to_h.weight.txt");
      LoadInput(lm_bias_2, PATH + weights_path + ".dense_4h_to_h.bias.txt");
    }    

    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".dense_4h_to_h_in.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".dense_4h_to_h_in.txt"); 
    hidden_states = Linear(vec4d_t(1, hidden_states), lm_weights_2, lm_bias_2)[0];
    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".dense_4h_to_h_out.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".dense_4h_to_h_out.txt"); 

    vec3d_t output(hidden_states.size(), vec2d_t(hidden_states[0].size(), vec1d_t(hidden_states[0][0].size(), 0)));
    output = dropout_add(hidden_states, residual, config.hidden_dropout, config.training);
    return output;
}

std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>, vec4d_t> BLOOMAttention(BLOOMConfig config, vec3d_t hiddenstates, vec3d_t residual, vec3d_t alibi, 
                                                                vec4d_t_i attention_mask, std::tuple<vec3d_t, vec3d_t> layer_past,
                                                                vec4d_t_i head_mask, bool use_cache, bool output_attentions, std::string path) 
{

  vec2d_t query_key_value_weights(3 * config.hidden_size, vec1d_t(hiddenstates[0][0].size(), 0.0));
  vec1d_t query_key_value_bias(3 * config.hidden_size);

  std::cout << "alibi: " << alibi.size() << " " << alibi[0].size() << " " << alibi[0][0].size() << std::endl;

  if (config.training)
  {
    std::string name = path + "_in";
    std::cout << "SAVE DATA: " << name << std::endl;
    ForwardMap[name].v3d = hiddenstates; 
  }

  if (config.training)
  {
    std::string name = path + ".query_key_value.weight";
    std::cout << "Read WeightMap:" << ' ' << name <<  std::endl;
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    query_key_value_weights = WeightMap[name].v2d;

    name = path + ".query_key_value.bias";
    std::cout << "Read WeightMap:" << ' ' << name <<  std::endl;
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    query_key_value_bias = WeightMap[name].v1d;
  }
  else
  {
    LoadInput(query_key_value_weights, PATH + path + ".query_key_value.weight.txt");
    LoadInput(query_key_value_bias, PATH + path + ".query_key_value.bias.txt");
  }
  if(write) WriteTensor(vec4d_t(1, hiddenstates), PATH_TEST + path + ".query_key_value_in.txt");
  if(berr) Error_FOR(vec4d_t(1, hiddenstates), PATH_FOR + path + ".query_key_value_in.txt");
  vec3d_t fused_qkv = Linear(vec4d_t(1, hiddenstates), query_key_value_weights, query_key_value_bias)[0];
  if(write) WriteTensor(vec4d_t(1, fused_qkv), PATH_TEST + path + ".query_key_value_out.txt");
  if(berr) Error_FOR(vec4d_t(1, fused_qkv), PATH_FOR + path + ".query_key_value_out.txt");
  vec5d_t qkv = _split_heads(fused_qkv, config);
  vec4d_t query_layer = qkv[0];
  vec4d_t key_layer = qkv[1];
  vec4d_t value_layer = qkv[2];

  int batch_size = query_layer.size();
  int q_length = query_layer[0].size();

  query_layer = transpose(query_layer, 1, 2);
  key_layer = permute(key_layer, 0, 2, 3, 1);
  value_layer = transpose(value_layer, 1, 2);

  vec3d_t _query_layer = reshape(query_layer, batch_size * config.n_head, q_length, config.hidden_size / config.n_head);
  vec3d_t _key_layer = reshape(key_layer, batch_size * config.n_head, config.hidden_size / config.n_head , q_length);
  vec3d_t _value_layer = reshape(value_layer, batch_size * config.n_head, q_length, config.hidden_size / config.n_head);

  if(std::get<0>(layer_past).size() != 0 && std::get<1>(layer_past).size()) {
    vec3d_t past_key = std::get<0>(layer_past), past_value = std::get<1>(layer_past);
    _key_layer = cat(past_key, _key_layer, 2);
    _value_layer = cat(past_value, _value_layer, 1);
  }

  int kv_length = _key_layer[0][0].size();
  std::tuple<vec3d_t, vec3d_t> present;

  if(use_cache) {
    present = std::make_tuple(_key_layer, _value_layer);
  } 

  float inv_norm_factor = 1.0 / std::sqrt(config.hidden_size / config.n_head);
  float beta = 1.0;

  if(write) WriteTensor(vec4d_t(1, alibi), PATH_TEST + path + ".baddbmm_in.txt");
  if(berr) Error_FOR(vec4d_t(1, alibi), PATH_FOR + path + ".baddbmm_in.txt");
  if(write) WriteTensor(vec4d_t(1, _query_layer), PATH_TEST + path + ".query_layer.txt");
  if(write) WriteTensor(vec4d_t(1, _key_layer), PATH_TEST + path + ".key_layer.txt");
  if (config.training)
  {
    std::string name = path + ".query_layer";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v3d = _query_layer;

    name = path + ".key_layer";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v3d = _key_layer;
  }
  vec3d_t matmul_result = baddbmm(alibi, _query_layer, _key_layer, beta, inv_norm_factor);
  if(write) WriteTensor(vec4d_t(1, matmul_result), PATH_TEST + path + ".baddbmm_out.txt");
  if(berr) Error_FOR(vec4d_t(1, matmul_result), PATH_FOR + path + ".baddbmm_out.txt");
  vec4d_t attention_scores = view(vec4d_t(1, matmul_result), batch_size, config.n_head, q_length, kv_length);

  vec4d_t attn_weights = masked_fill(attention_scores, attention_mask, -3.4028234663852886e+38);//std::numeric_limits<float>::min());

  if(write) WriteTensor(attn_weights, PATH_TEST + path + ".softmax_in.txt");
  if(berr) Error_FOR(attn_weights, PATH_FOR + path + ".softmax_in.txt");
  // if (bloomsplit == 1)
  // {
  //   WriteOutput(attn_weights, "2.log");
  // }
  vec4d_t attention_probs = softmax(attn_weights);
  if(write) WriteTensor(attention_probs, PATH_TEST + path + ".softmax_out.txt");
  if(berr) Error_FOR(attention_probs, PATH_FOR + path + ".softmax_out.txt");
  // if (bloomsplit == 1)
  // {
  //   WriteOutput(attention_probs, "3.log");
  //   exit(1);
  // }

  if (config.training)
  {
    std::string name = path + ".softmax_out";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v4d = attention_probs;
  }
  if(write) WriteTensor(attention_probs, PATH_TEST + path + ".attention_dropout_in.txt");
  if(berr) Error_FOR(attention_probs, PATH_FOR + path + ".attention_dropout_in.txt");
  if(write) WriteTensor(attention_probs, PATH_TEST + path + ".attention_dropout_out.txt");
  if(berr) Error_FOR(attention_probs, PATH_FOR + path + ".attention_dropout_out.txt");
  vec3d_t attention_probs_reshaped = reshape(attention_probs, batch_size * config.n_head, q_length, kv_length);
  
  // if (head_mask)

  if(write) WriteTensor(vec4d_t(1, attention_probs_reshaped), PATH_TEST + path + ".bmm_in.txt");
  if(berr) Error_FOR(vec4d_t(1, attention_probs_reshaped), PATH_FOR + path + ".bmm_in.txt");
  if(write) WriteTensor(vec4d_t(1, attention_probs_reshaped), PATH_TEST + path + ".attention_probs_reshaped.txt");
  if(write) WriteTensor(vec4d_t(1, _value_layer), PATH_TEST + path + ".value_layer.txt");
  
  if (config.training)
  {
    std::string name = path + ".attention_probs_reshaped";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v3d = attention_probs_reshaped;

    name = path + ".value_layer";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v3d = _value_layer;
  }
  
  vec3d_t context_layer = bmm(attention_probs_reshaped, _value_layer);
  if(write) WriteTensor(vec4d_t(1, context_layer), PATH_TEST + path + ".bmm_out.txt");
  if(berr) Error_FOR(vec4d_t(1, context_layer), PATH_FOR + path + ".bmm_out.txt");

  context_layer = _merge_heads(context_layer, config);

  vec3d_t output_tensor(context_layer.size(), vec2d_t(context_layer[0].size(), vec1d_t(context_layer[0][0].size(), 0.0)));

  vec2d_t dense_weights(config.hidden_size, vec1d_t(config.hidden_size));
  vec1d_t dense_bias(config.hidden_size);
  if (config.training)
  {
    std::string name = path + ".dense.weight";
    std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    dense_weights = WeightMap[name].v2d;
    dense_weights = WeightMap[name].v2d;

    name = path + ".dense.bias";
    std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    dense_bias = WeightMap[name].v1d;
  }
  else
  {
    LoadInput(dense_weights, PATH + path + ".dense.weight.txt");
    LoadInput(dense_bias, PATH + path + ".dense.bias.txt");
  }
  if(write) WriteTensor(vec4d_t(1, context_layer), PATH_TEST + path + ".dense_in.txt");
  if(berr) Error_FOR(vec4d_t(1, context_layer), PATH_FOR + path + ".dense_in.txt");

  if (config.training)
  {
    std::string name = path + ".dense_in";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v3d = context_layer;
  }

  if(config.pretraining_tp > 1 && config.slow_but_exact) {
    int slices = config.hidden_size / config.pretraining_tp;
    for (int i = 0; i < config.pretraining_tp; i++) {
      vec3d_t splitout(context_layer.size(), vec2d_t(context_layer[0].size(), vec1d_t(slices, 0.0)));
      for(int j = 0; j < splitout[0].size(); j++)
      {
        for (int k = 0; k < splitout[0][0].size(); k++)
        {
          splitout[0][j][k] = context_layer[0][j][i*slices+k];
        }
      }
      vec2d_t splitweight(context_layer[0][0].size(), vec1d_t(slices, 0.0));
      for (int j = 0; j < splitweight.size(); j++)
      {
        for (int k = 0; k < splitweight[0].size(); k++)
        {
          splitweight[j][k] = dense_weights[j][i*slices+k];
        }
      }
      vec3d_t outputsplit = Linear(vec4d_t(1, splitout), splitweight, dense_bias)[0];
      output_tensor = AddVector(vec4d_t(1, outputsplit), vec4d_t(1, output_tensor))[0];
    }
  }
  else{
    output_tensor = Linear(vec4d_t(1, context_layer), dense_weights, dense_bias)[0];
  }
  if(write) WriteTensor(vec4d_t(1, output_tensor), PATH_TEST + path + ".dense_out.txt");
  if(berr) Error_FOR(vec4d_t(1, output_tensor), PATH_FOR + path + ".dense_out.txt");

  output_tensor = dropout_add(output_tensor, residual, 0, false);
  
  std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>, vec4d_t> outputs(output_tensor, present, vec4d_t());

  if (output_attentions)
  {
    std::get<2>(outputs) = attention_probs;
  }
  return outputs;
}

std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>, vec4d_t> BLOOMBlock(BLOOMConfig config, vec3d_t hidden_states, vec3d_t alibi, vec4d_t_i attention_mask, std::tuple<vec3d_t, vec3d_t> layer_past, vec4d_t_i head_mask, bool use_cache, bool output_attentions, std::string weights_path) {
  
  if (config.training)
  {
    std::string name = weights_path + ".input_layernorm_in";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v4d = vec4d_t(1, hidden_states);
  }
  
  vec3d_t layernorm_output(hidden_states.size(), vec2d_t(hidden_states[0].size(), vec1d_t(hidden_states[0][0].size())));
  vec1d_t ln_weights_1(hidden_states[0][0].size(), 0.0);
  vec1d_t ln_bias_1(hidden_states[0][0].size(), 0.0);
  if (config.training)
  {
    std::string name = weights_path + ".input_layernorm.weight";
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    ln_weights_1 = WeightMap[name].v1d;

    name = weights_path + ".input_layernorm.bias";
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    ln_bias_1 = WeightMap[name].v1d;
  }
  else
  {
    LoadInput(ln_weights_1, PATH + weights_path + ".input_layernorm.weight.txt");
    LoadInput(ln_bias_1, PATH + weights_path + ".input_layernorm.bias.txt");
  }
  if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".input_layernorm_in.txt");
  if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".input_layernorm_in.txt");
  
  layernorm_output = LayerNorm(vec4d_t(1, hidden_states), ln_weights_1, ln_bias_1, config.layer_norm_epsilon)[0];
  if(write) WriteTensor(vec4d_t(1, layernorm_output), PATH_TEST + weights_path + ".input_layernorm_out.txt");
  if(berr) Error_FOR(vec4d_t(1, layernorm_output), PATH_FOR + weights_path + ".input_layernorm_out.txt");  

  vec3d_t residual(hidden_states.size(), vec2d_t(hidden_states[0].size(), vec1d_t(hidden_states[0][0].size())));
  if (config.apply_residual_connection_post_layernorm)
  {
      residual = layernorm_output;
  }
  else 
  {
      residual = hidden_states;
  }
  
  if(write) WriteTensor(vec4d_t(1, layernorm_output), PATH_TEST + weights_path + ".self_attention_in.txt"); 
  if(berr) Error_FOR(vec4d_t(1, layernorm_output), PATH_FOR + weights_path + ".self_attention_in.txt"); 
  // bloomAttention
  std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>, vec4d_t> attn_outputs = BLOOMAttention(config, layernorm_output, residual, alibi, attention_mask, layer_past, head_mask, use_cache, output_attentions, weights_path + ".self_attention");
  // std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>> attn_outputs;
  vec3d_t attention_output = std::get<0>(attn_outputs);
  // std::tuple<vec3d_t, vec3d_t> outputs = std::get<1>(attn_outputs);
  std::tuple<std::tuple<vec3d_t, vec3d_t>, vec4d_t> outputs(std::get<1>(attn_outputs), std::get<2>(attn_outputs));
  if(write) WriteTensor(vec4d_t(1, attention_output), PATH_TEST + weights_path + ".self_attention_out.txt"); 
  if(berr) Error_FOR(vec4d_t(1, attention_output), PATH_FOR + weights_path + ".self_attention_out.txt"); 
  // vec3d_t attention_output(layernorm_output.size(), vec2d_t(layernorm_output[0].size(), vec1d_t(layernorm_output[0][0].size())));
  // LoadInput(attention_output, PATH_FOR + weights_path + ".post_attention_layernorm" + "_in.txt");


  if (config.training)
  {
    std::string name = weights_path + ".post_attention_layernorm_in";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v4d = vec4d_t(1, attention_output);
  }

  vec1d_t ln_weights_2(attention_output[0][0].size());
  vec1d_t ln_bias_2(attention_output[0][0].size());
  if(config.training)
  {
    std::string name = weights_path + ".post_attention_layernorm.weight";
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    ln_weights_2 = WeightMap[name].v1d;

    name = weights_path + ".post_attention_layernorm.bias";
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    ln_bias_2 = WeightMap[name].v1d;
  }
  else
  {
    LoadInput(ln_weights_2, PATH + weights_path + ".post_attention_layernorm.weight.txt");
    LoadInput(ln_bias_2, PATH + weights_path + ".post_attention_layernorm.bias.txt");
  }
  if(write) WriteTensor(vec4d_t(1, attention_output), PATH_TEST + weights_path + ".post_attention_layernorm_in.txt"); 
  if(berr) Error_FOR(vec4d_t(1, attention_output), PATH_FOR + weights_path + ".post_attention_layernorm_in.txt"); 
  layernorm_output = LayerNorm(vec4d_t(1, attention_output), ln_weights_2, ln_bias_2, config.layer_norm_epsilon)[0];
  if(write) WriteTensor(vec4d_t(1, layernorm_output), PATH_TEST + weights_path + ".post_attention_layernorm_out.txt"); 
  if(berr) Error_FOR(vec4d_t(1, layernorm_output), PATH_FOR + weights_path + ".post_attention_layernorm_out.txt"); 
  if (config.apply_residual_connection_post_layernorm)
  {
      residual = layernorm_output;
  }
  else 
  {
      residual = attention_output;
  }
  if(write) WriteTensor(vec4d_t(1, layernorm_output), PATH_TEST + weights_path + ".mlp_in.txt");
  if(berr) Error_FOR(vec4d_t(1, layernorm_output), PATH_FOR + weights_path + ".mlp_in.txt");
  vec3d_t output = BLOOMMLP(config, layernorm_output, residual, weights_path + ".mlp");
  if(write) WriteTensor(vec4d_t(1, output), PATH_TEST + weights_path + ".mlp_out.txt");
  if(berr) Error_FOR(vec4d_t(1, output), PATH_FOR + weights_path + ".mlp_out.txt"); 
  if (config.use_cache)
    return std::make_tuple(output, std::get<0>(outputs), std::get<1>(outputs));
  else 
  {
    return std::make_tuple(output, std::make_tuple(vec3d_t(), vec3d_t()), std::get<1>(outputs));
  }
}


std::tuple<vec3d_t, std::vector<std::tuple<vec3d_t, vec3d_t>>, std::vector<vec3d_t>, std::vector<vec4d_t>> 
BLOOMModel(BLOOMConfig config, vec2d_t input_ids, std::vector<std::tuple<vec3d_t, vec3d_t>> past_key_values, 
          vec2d_t_i attention_mask, vec5d_t_i head_mask, vec3d_t inputs_embeds, 
          bool use_cache, bool output_attentions, bool output_hidden_states, bool return_dict,
          std::string weights_path) 
{
    bool gradient_checkpointing = false;

    int batch_size, seq_length;
    if(input_ids.size() != 0 && inputs_embeds.size() != 0) {
      std::cerr << "You cannot specify both input_ids and inputs_embeds at the same time\n";
      exit(-1);
    }
    else if(input_ids.size() != 0) {
      batch_size = input_ids.size();
      seq_length = input_ids[0].size();
    }
    else if(inputs_embeds.size() != 0){
      batch_size = inputs_embeds.size();
      seq_length = inputs_embeds[0].size();
    }
    else{
      std::cerr << "You have to specify either input_ids or inputs_embeds\n";
      exit(-1);
    }

    if(past_key_values.size() == 0) {
      for (int i = 0; i < config.n_layer; i++)
        past_key_values.emplace_back(std::make_tuple(vec3d_t(), vec3d_t()));
    }

    head_mask = getHeadMask(head_mask, config.n_layer);

    if(inputs_embeds.size() == 0) {
      vec2d_t word_embeddings(config.vocab_size, vec1d_t(config.hidden_size));
      
      if (config.training)
      {
        std::string name = weights_path + ".word_embeddings.weight";
        std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
        if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
        word_embeddings = WeightMap[name].v2d;
      }
      else 
      {
        LoadInput(word_embeddings,PATH + weights_path + ".word_embeddings.weight.txt");
      }
      inputs_embeds = Embedding(word_embeddings, input_ids);
      if(berr) Error_FOR(vec4d_t(1, vec3d_t(1, input_ids)), PATH_FOR + weights_path + ".word_embeddings_in.txt");
    }
    if(berr) Error_FOR(vec4d_t(1, inputs_embeds), PATH_FOR + weights_path + ".word_embeddings_out.txt");

    if (config.training)
    {
      std::string name = weights_path + ".word_embeddings_layernorm_in";
      std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
      ForwardMap[name].v4d = vec4d_t(1, inputs_embeds); 
    }

    vec1d_t embedding_weight(config.hidden_size);
    vec1d_t embedding_bias(config.hidden_size);
    if (config.training)
    {
      std::string name = weights_path + ".word_embeddings_layernorm.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      embedding_weight = WeightMap[name].v1d;

      name = weights_path + ".word_embeddings_layernorm.bias";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      embedding_bias = WeightMap[name].v1d;
    }
    else 
    {
      LoadInput(embedding_weight, PATH + weights_path + ".word_embeddings_layernorm.weight.txt");
      LoadInput(embedding_bias, PATH + weights_path + ".word_embeddings_layernorm.bias.txt");
    }
    if(write) WriteTensor(vec4d_t(1, inputs_embeds), PATH_TEST + weights_path + ".word_embeddings_layernorm_in.txt");
    if(berr) Error_FOR(vec4d_t(1, inputs_embeds), PATH_FOR + weights_path + ".word_embeddings_layernorm_in.txt");
    vec3d_t hidden_states = LayerNorm(vec4d_t(1, inputs_embeds), embedding_weight, embedding_bias, config.layer_norm_epsilon)[0];
    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".word_embeddings_layernorm_out.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".word_embeddings_layernorm_out.txt");

    std::vector<std::tuple<vec3d_t, vec3d_t>> presents;
    std::vector<vec4d_t> all_self_attentions;
    std::vector<vec3d_t> all_hidden_states;

    if(gradient_checkpointing && config.training) {
      if(use_cache){
        std::cerr << "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...";
      }
      use_cache = false;
    }

    int seq_length_with_past = seq_length;
    int past_key_values_length = 0;
    if(std::get<0>(past_key_values[0]).size() != 0 && std::get<1>(past_key_values[0]).size() != 0) {
      past_key_values_length = std::get<0>(past_key_values[0])[0][0].size();
      seq_length_with_past = seq_length_with_past + past_key_values_length;
    }

    if(attention_mask.size() == 0) {
      attention_mask.resize(batch_size, vec1d_t_i(seq_length_with_past));
      for (int i = 0; i < batch_size; i++)
      {
        for (int j = 0; j < seq_length_with_past; j++) {
          attention_mask[i][j] = 1;
        }
      }
    }

    vec3d_t alibi = build_alibi(attention_mask, config.n_head);
    // WriteTensor(vec4d_t(1, alibi),"/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/alibi_7b.txt");
    // vec3d_t alibi;

    vec4d_t_i causal_mask = _prepare_attn_mask(attention_mask, std::make_tuple(batch_size, seq_length), past_key_values_length);
    int a = 1;
    int zero_num = 0;
    for (auto x : causal_mask)
    {
      for( auto x_1 : x)
      {
        for ( auto x_2 : x_1 )
        {
          for ( auto x_3 : x_2 )
          {
            if (a == x_3) zero_num += 1;
          }
        }
      }
    }
    std::cout << "zero_num: " << zero_num << std::endl;
    std::cout << "causal_mask: " << causal_mask.size() << " " << causal_mask[0].size() << " " << causal_mask[0][0].size() << " " << causal_mask[0][0][0].size() << std::endl;
    
    for (int i = 0; i < config.n_layer; i++)
    {
      std::tuple<vec3d_t, std::tuple<vec3d_t, vec3d_t>, vec4d_t> block_output;
      if (output_hidden_states)
      {
        all_hidden_states.push_back(hidden_states);
      }

      if(gradient_checkpointing && config.training) {
        
      }
      else{
        if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".h." + std::to_string(i) + "_in.txt");
        if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".h." + std::to_string(i) + "_in.txt");
        if (i ==23) WriteTensor(vec4d_t(1, hidden_states), PATHWeight + weights_path + ".h." + std::to_string(i) + "_in.txt");
        block_output = BLOOMBlock(config, hidden_states, alibi, causal_mask, past_key_values[i], head_mask[i], use_cache, output_attentions, weights_path + ".h." + std::to_string(i));
      }

      hidden_states = std::get<0>(block_output);
      if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".h." + std::to_string(i) + "_out.txt");
      if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".h." + std::to_string(i) + "_out.txt");

      if(use_cache)
        presents.push_back(std::get<1>(block_output));

      if(output_attentions)
        all_self_attentions.push_back(std::get<2>(block_output));
      
      // bloomsplit++;
    }

    if (config.training)
    {
      std::string name = weights_path + ".ln_f_in";
      std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
      ForwardMap[name].v4d = vec4d_t(1, hidden_states);
    }

    vec1d_t ln_f_weight(hidden_states[0][0].size());
    vec1d_t ln_f_bias(hidden_states[0][0].size());
    if (config.training)
    {
      std::string name = weights_path + ".ln_f.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      ln_f_weight = WeightMap[name].v1d;

      name = weights_path + ".ln_f.bias";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      ln_f_bias = WeightMap[name].v1d;
    }
    else
    {
      LoadInput(ln_f_weight, PATH + weights_path + ".ln_f.weight.txt");
      LoadInput(ln_f_bias, PATH + weights_path + ".ln_f.bias.txt");
    }
    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".ln_f_in.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".ln_f_in.txt");
    hidden_states = LayerNorm(vec4d_t(1, hidden_states), ln_f_weight, ln_f_bias, config.layer_norm_epsilon)[0];
    if(write) WriteTensor(vec4d_t(1, hidden_states), PATH_TEST + weights_path + ".ln_f_out.txt");
    if(berr) Error_FOR(vec4d_t(1, hidden_states), PATH_FOR + weights_path + ".ln_f_out.txt");
    if(output_hidden_states){
      all_hidden_states.emplace_back(hidden_states);
    }


    return std::make_tuple(hidden_states, presents, all_hidden_states, all_self_attentions);
}


std::tuple<vec1d_t, vec3d_t, std::vector<std::tuple<vec3d_t, vec3d_t>>, std::vector<vec3d_t>, std::vector<vec4d_t>>
BloomForCausalLM(BLOOMConfig config, vec2d_t input_ids, std::vector<std::tuple<vec3d_t, vec3d_t>> past_key_values, vec2d_t_i attention_mask, 
                vec5d_t_i head_mask, vec3d_t inputs_embeds, vec2d_t_i labels, bool use_cache, bool output_attentions, bool output_hidden_states, bool return_dict,
                std::string weights_path) {
  
  // return_dict = return_dict ? return_dict : config.use_return_dict;

  std::tuple<vec3d_t, std::vector<std::tuple<vec3d_t, vec3d_t>>, std::vector<vec3d_t>, std::vector<vec4d_t>>
      transformer_outputs = BLOOMModel(config, input_ids, past_key_values, attention_mask, head_mask, inputs_embeds, 
                                      use_cache, output_attentions, output_hidden_states, return_dict, weights_path + "");

  vec3d_t hidden_states = std::get<0>(transformer_outputs);

  vec2d_t lm_logits_weights(config.vocab_size, vec1d_t(config.hidden_size, 0));
  if (config.training)
  {
    std::string name = weights_path + ".lm_head.weight";
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
    lm_logits_weights = WeightMap[name].v2d;
  }
  else
  {
    LoadInput(lm_logits_weights, weights_path + ".lm_head.weight");
  }
  
  if (config.training)
  {
    std::string name = "lm_forward_in";
    std::cout << "SAVE DATA:" << ' ' << name <<  std::endl;
    ForwardMap[name].v4d = vec4d_t(1, hidden_states);
  }
  vec3d_t lm_logits = Linear_Nobias(vec4d_t(1, hidden_states), lm_logits_weights)[0];


  vec1d_t loss;

  if(labels.size() != 0) {
    vec3d_t shift_logits(lm_logits.size(), vec2d_t(lm_logits[0].size(), vec1d_t(lm_logits[0][0].size() - 1, 0)));
    vec2d_t_i shift_labels(labels.size(), vec1d_t_i(labels[0].size(), 0));

    for (int i = 0; i < shift_logits.size(); i++) {
      for (int j = 0; j < shift_logits[0].size(); j++) {
        for(int k = 0; k < shift_logits[0][0].size(); k++) {
          shift_logits[i][j][k] = lm_logits[i][j][k];
        }
      }
    }

    for (int i = 0; i < shift_labels.size(); i++) {
      for (int j = 0; j < shift_labels[0].size(); j++) {
        shift_labels[i][j] = labels[i][j+1];
      }
    }

    vec2d_t _shift_logits(lm_logits.size() * lm_logits[0].size(), vec1d_t(lm_logits[0][0].size(), 0));
    vec1d_t_i _shift_labels(shift_labels.size() * shift_labels[0].size(), 0);
    for(int i = 0; i < shift_logits.size(); i++) {
      for(int j = 0; j < shift_logits[0].size(); j++) {
        _shift_logits.push_back(shift_logits[i][j]);
        _shift_labels[i * shift_logits[0].size() + j] = shift_labels[i][j];
      }
    }
    
    loss = CrossEntropyLoss(_shift_logits, _shift_labels);
  }

  if(!return_dict) {
    if(loss.size() != 0)
      return std::make_tuple(loss, lm_logits, std::get<1>(transformer_outputs), std::get<2>(transformer_outputs), std::get<3>(transformer_outputs));
    else
      return std::make_tuple(vec1d_t(), lm_logits, std::get<1>(transformer_outputs), std::get<2>(transformer_outputs), std::get<3>(transformer_outputs));
  }

  return std::make_tuple(loss, lm_logits, std::get<1>(transformer_outputs), std::get<2>(transformer_outputs), std::get<3>(transformer_outputs));
}

vec3d_t BLOOMAttentionBackward(BLOOMConfig config, vec3d_t backward_input, bool use_cache, bool output_attentions, std::string weight_bias_name) {
  
  vec3d_t backward_output;
  vec3d_t dense_dx;
  vec2d_t dense_dw;
  vec1d_t dense_db;
  // // vec3d_t attention_forward_in = ForwardMap.at(weight_bias_name + "_forward_in").v3d;
  vec3d_t attention_forward_in(1, vec2d_t(backward_input[0].size(), vec1d_t(config.hidden_size, 0.0)));
  if (config.training)
  {
    std::string name = weight_bias_name + "_in";
    std::cout << "Read ForwardMap:" << ' ' << name << std::endl;
    if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name << std::endl;
    attention_forward_in = ForwardMap[name].v3d;
  } else {
    LoadInput(attention_forward_in, PATH_TEST + weight_bias_name + "_in.txt");
  }

  if (config.pretraining_tp > 1 && config.slow_but_exact){

  }
  else{
    vec2d_t dense_weights(config.hidden_size, vec1d_t(config.hidden_size, 0.0));
    vec1d_t dense_bias(config.hidden_size);

    if (config.training)
    {
      std::string name = weight_bias_name + ".dense.weight";
      std::cout << "Read WeightMap:" << ' ' << name << std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name << std::endl;
      dense_weights = WeightMap[name].v2d;

      name = weight_bias_name + ".dense.bias";
      std::cout << "Read WeightMap:" << ' ' << name << std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name << std::endl;
      dense_bias = WeightMap[name].v1d;
    } else {
      LoadInput(dense_weights, PATH + weight_bias_name + ".dense.weight.txt");
      LoadInput(dense_bias, PATH + weight_bias_name + ".dense.bias.txt");
    }
    vec3d_t dense_forward(1, vec2d_t(backward_input[0].size(), vec1d_t(config.hidden_size, 0.0)));

    if (config.training)
    {
      std::string name = weight_bias_name + ".dense_in";
      std::cout << "Read ForwardMap:" << ' ' << name << std::endl;
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name << std::endl;
      dense_forward = ForwardMap[name].v3d;
    } else { 
      LoadInput(dense_forward, PATH_TEST + weight_bias_name + ".dense_in.txt");
    }
    if(write) WriteTensor(vec4d_t(1, backward_input), PATHBACK + weight_bias_name + ".dense_in.txt");
    if(berr) Error(vec4d_t(1, backward_input), PATH_BACK + weight_bias_name + ".dense_in.txt");
    if (tran) {
      transition(backward_input);
      transition(dense_weights);
      transition(dense_forward);
    }
    dense_dx = LinearDx(vec4d_t(1, backward_input), dense_weights)[0];
    std::cout << "dense_dx: " << dense_dx.size() << " " << dense_dx[0].size() << " " << dense_dx[0][0].size() << std::endl;
    if(write) WriteTensor(vec4d_t(1, dense_dx), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/testdata/560m/cpp的输入/dense_out.txt");
    dense_dw = LinearDw(vec4d_t(1, dense_forward), vec4d_t(1, backward_input)); 
    dense_db = LinearDb(vec4d_t(1, backward_input));
    if (config.training)
    {
      std::string name = weight_bias_name + ".dense.weight";
      UpdateWeight(config, dense_weights, dense_dw);
      if (update) {
        WriteTensor(dense_weights, PATHWeight + weight_bias_name + ".dense.weight.txt");
        // WriteTensor(dense_dw, PATHUPDATE + weight_bias_name + ".dense.weight.txt");
        // Error_grad(dense_dw, PATHGRAD + name + ".txt");
      }
      WeightMap[name].v2d = dense_weights;

      name = weight_bias_name + ".dense.bias";
      UpdateBias(config, dense_bias, dense_db);
      if (update) {
        WriteTensor(dense_bias, PATHWeight + weight_bias_name + ".dense.bias.txt");
        // WriteTensor(dense_db, PATHUPDATE + weight_bias_name + ".dense.bias.txt");
        // Error_grad(vec2d_t(1, dense_db), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = dense_bias;
    }
  }
  if(write) WriteTensor(vec4d_t(1, dense_dx), PATHBACK + weight_bias_name + ".dense_out.txt");
  if(berr) Error(vec4d_t(1, dense_dx), PATH_BACK + weight_bias_name + ".dense_out.txt");
  dense_dx = _merge_heads_backward(dense_dx, config);


  vec3d_t bmm_forward_in1(config.n_head, vec2d_t(backward_input[0].size(), vec1d_t(backward_input[0].size(), 0.0)));
  vec3d_t bmm_forward_in2(config.n_head, vec2d_t(backward_input[0].size(), vec1d_t(config.hidden_size / config.n_head, 0.0)));
  if (config.training)
  {
    std::string name = weight_bias_name + ".attention_probs_reshaped";
    std::cout << "Read ForwardMap: " << name << std::endl;
    if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name << std::endl;
    bmm_forward_in1 = ForwardMap[name].v3d;

    name = weight_bias_name + ".value_layer";
    std::cout << "Read ForwardMap: " << name << std::endl;
    if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name << std::endl;
    bmm_forward_in2 = ForwardMap[name].v3d;
  } else {
    LoadInput(bmm_forward_in1, PATH_TEST + weight_bias_name + ".attention_probs_reshaped.txt");
    LoadInput(bmm_forward_in2, PATH_TEST + weight_bias_name + ".value_layer.txt");
  }
  if(write) WriteTensor(vec4d_t(1, dense_dx), PATHBACK + weight_bias_name + ".bloombmm_in.txt");
  if(berr) Error(vec4d_t(1, dense_dx), PATH_BACK + weight_bias_name + ".bloombmm_in.txt");
  
  if (tran) {
    transition(dense_dx);
    transition(bmm_forward_in1);
    transition(bmm_forward_in2);
  }
  vec3d_t bmm_dx = bmmDx(dense_dx, bmm_forward_in2);
  if(write) WriteTensor(vec4d_t(1, bmm_dx), PATHBACK + weight_bias_name + ".bloombmm_out.txt");
  if(berr) Error(vec4d_t(1, bmm_dx), PATH_BACK + weight_bias_name + ".bloombmm_out.txt");
  vec3d_t _value_layer_dx = bmmDw(bmm_forward_in1, dense_dx);
  std::cout << "bmm_forward_in1: " << bmm_forward_in1.size() << " " << bmm_forward_in1[0].size() << " " << bmm_forward_in1[0][0].size() << std::endl;
  std::cout << "_value_layer_dx: " << _value_layer_dx.size() << " " << _value_layer_dx[0].size() << " " << _value_layer_dx[0][0].size() << std::endl;
  // return _value_layer_dx;
  std::cout << "dense_dx: " << dense_dx.size() << " " << dense_dx[0].size() << " " << dense_dx[0][0].size() << std::endl;
  vec4d_t attention_probs_reshaped_dx = reshape(bmm_dx, bmm_dx.size() / config.n_head, config.n_head, bmm_dx[0][0].size(), bmm_dx[0][0].size());
  std::cout << "bmm_dx: [" << bmm_dx.size() << " " << bmm_dx[0].size() << " " << bmm_dx[0][0].size() << "]" << std::endl;
  std::cout << "attention_probs_reshaped_dx: [" << bmm_dx.size() / config.n_head << " " << config.n_head << " " << bmm_dx[0][0].size() << " " <<  bmm_dx[0][0].size() << "]" << std::endl;
  
  
  // if(head_mask.size() != 0) {
  //   for (int b = 0; b < head_mask.size(); b++) {
  //     for(int h = 0; h < head_mask[0].size(); h++) {
  //       for(int q_len = 0; q_len < head_mask[0][0].size(); q_len++) {
  //         for(int kv_len = 0; kv_len < head_mask[0][0][0].size(); kv_len++) {
  //           attention_probs_reshaped_dx[b][h][q_len][kv_len] = head_mask[b][h][q_len][kv_len] ? attention_probs_reshaped_dx[b][h][q_len][kv_len] : 0;
  //         }
  //       }
  //     }
  //   }
  // }

  // vec4d_t attention_probs_mask = config.training == true ? ForwardMap.at(weight_bias_name + "_attention_probs_mask").v4d : vec4d_t();
  vec4d_t attention_probs_mask;

  if(write) WriteTensor(attention_probs_reshaped_dx, PATHBACK + weight_bias_name + ".attention_dropout_in.txt");
  if(berr) Error(attention_probs_reshaped_dx, PATH_BACK + weight_bias_name + ".attention_dropout_in.txt");
  vec4d_t attention_probs_dx = dropoutBackward(attention_probs_reshaped_dx, config.attention_dropout, attention_probs_mask, false);
  if(write) WriteTensor(attention_probs_dx, PATHBACK + weight_bias_name + ".attention_dropout_out.txt");
  if(berr) Error(attention_probs_dx, PATH_BACK + weight_bias_name + ".attention_dropout_out.txt");

  vec4d_t softmax_forward_out(1, vec3d_t(config.n_head, vec2d_t(backward_input[0].size(), vec1d_t(backward_input[0].size(), 0.0))));

  if (config.training)
  {
    std::string name = weight_bias_name + ".softmax_out";
    std::cout << "Read ForwardMap: " << name << std::endl;
    if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name << std::endl;
    softmax_forward_out = ForwardMap[name].v4d;
  } else {
    LoadInput(softmax_forward_out, PATH_TEST + weight_bias_name + ".softmax_out.txt");
  }
  if(write) WriteTensor(attention_probs_dx, PATHBACK + weight_bias_name + ".softmax_in.txt");
  if(berr) Error(attention_probs_dx, PATH_BACK + weight_bias_name + ".softmax_in.txt");
  std::cout << "attention_scores_dx: " << attention_probs_dx.size() << " " << attention_probs_dx[0].size() << " " << attention_probs_dx[0][0].size() << attention_probs_dx[0][0][0].size() <<  std::endl;
  if(write) WriteTensor(attention_probs_dx, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/testdata/560m/cpp的输入/softmax_in.txt");
  vec4d_t softmax_dx = Softmax_back(softmax_forward_out, attention_probs_dx);
  if(write) WriteTensor(softmax_dx, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/testdata/560m/cpp的输入/softmax_out.txt");
  if(write) WriteTensor(softmax_dx, PATHBACK + weight_bias_name + ".softmax_out.txt");
  if(berr) Error(softmax_dx, PATH_BACK + weight_bias_name + ".softmax_out.txt");

  
  vec4d_t masked_fill_dx = softmax_dx; // masked_fill(softmax_dx, attention_mask, 0);
  
  vec3d_t attention_scores_dx = reshape(masked_fill_dx, masked_fill_dx.size() * masked_fill_dx[0].size(), masked_fill_dx[0][0].size(), masked_fill_dx[0][0][0].size());

  float inv_norm_factor = 1.0 / std::sqrt(config.hidden_size / config.n_head);
  float beta = 1.0;

  vec3d_t baddbmm_forward_1(config.n_head, vec2d_t(backward_input[0].size(), vec1d_t(config.hidden_size / config.n_head, 0.0)));
  vec3d_t baddbmm_forward_2(config.n_head, vec2d_t(config.hidden_size / config.n_head, vec1d_t(backward_input[0].size(), 0.0)));
  if (config.training)
  {
    std::string name = weight_bias_name + ".query_layer";
    std::cout << "Read ForwardMap: " << name << std::endl;
    if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name << std::endl;
    baddbmm_forward_1 = ForwardMap[name].v3d;

    name = weight_bias_name + ".key_layer";
    std::cout << "Read ForwardMap: " << name << std::endl;
    if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name << std::endl;
    baddbmm_forward_2 = ForwardMap[name].v3d;
  } else {
    LoadInput(baddbmm_forward_1, PATH_TEST + weight_bias_name + ".query_layer.txt");
    LoadInput(baddbmm_forward_2, PATH_TEST + weight_bias_name + ".key_layer.txt");
  }
  std::cout << "嘿嘿～" << std::endl;
  std::cout << "attention_scores_dx: " << attention_scores_dx.size() << " " << attention_scores_dx[0].size() << " " << attention_scores_dx[0][0].size() << std::endl;
  // return attention_scores_dx;
  std::tuple<vec3d_t, vec3d_t> baddmm_dx = baddbmmBackward(attention_scores_dx, baddbmm_forward_1, baddbmm_forward_2, beta, inv_norm_factor);
  vec3d_t _query_layer_dx = std::get<0>(baddmm_dx);
  vec3d_t _key_layer_dx = std::get<1>(baddmm_dx);
  if(write) WriteTensor(vec4d_t(1, _query_layer_dx), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/testdata/560m/cpp的输入/query_layer.txt");
  if(write) WriteTensor(vec4d_t(1, _key_layer_dx), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/testdata/560m/cpp的输入/key_layer.txt");
  std::cout << "_query_layer_dx: " << _query_layer_dx.size() << " " << _query_layer_dx[0].size() << " " << _query_layer_dx[0][0].size() << std::endl;
  std::cout << "_key_layer_dx: " << _key_layer_dx.size() << " " << _key_layer_dx[0].size() << " " << _key_layer_dx[0][0].size() << std::endl;
  // return _key_layer_dx;
  std::cout << "嘿嘿1～" << std::endl;
  std::cout << "attention_forward_in: " << attention_forward_in.size() << std::endl;
  vec4d_t query_layer = reshape(_query_layer_dx, attention_forward_in.size(), config.n_head, attention_forward_in[0].size(), config.hidden_size / config.n_head);
  std::cout << "query_layer: " << query_layer.size() << " " << query_layer[0].size() << " " << query_layer[0][0].size() <<" " << query_layer[0][0][0].size() << std::endl;
  vec4d_t key_layer = reshape(_key_layer_dx, attention_forward_in.size(), config.n_head, config.hidden_size / config.n_head, attention_forward_in[0].size());
  std::cout << "key_layer: " << key_layer.size() << " " << key_layer[0].size() << " " << key_layer[0][0].size() << " " << key_layer[0][0][0].size() << std::endl;
  vec4d_t value_layer = reshape(_value_layer_dx, attention_forward_in.size(), config.n_head, attention_forward_in[0].size(), config.hidden_size / config.n_head);
  std::cout << "value_layer: " << value_layer.size() << " " << value_layer[0].size() << " " << value_layer[0][0].size() << " " <<  value_layer[0][0][0].size() << std::endl;
  std::cout << "嘿嘿22～" << std::endl;
  query_layer = transpose(query_layer, 1, 2);
  std::cout << "嘿嘿221～" << std::endl;
  key_layer = permute(key_layer, 0, 3, 1, 2);
  std::cout << "key_layer: " << key_layer.size() << " " << key_layer[0].size() << " " << key_layer[0][0].size() << std::endl;
  std::cout << "嘿嘿222～" << std::endl;
  value_layer = transpose(value_layer, 1, 2);
       
  std::cout << "嘿嘿2～" << std::endl;
  vec3d_t fuse_qkv(attention_forward_in.size(), vec2d_t(attention_forward_in[0].size(), vec1d_t(3 * config.hidden_size, 0)));
  
  //split_head backward
  for (int i = 0; i < attention_forward_in.size(); i++) {
    for (int j = 0; j < attention_forward_in[0].size(); j++) {
      for (int m = 0; m < config.n_head; m++) {
        for (int n = 0; n < config.hidden_size / config.n_head; n++) {
          fuse_qkv[i][j][m * (config.hidden_size / config.n_head) * 3 + n] = query_layer[i][j][m][n];
        }
        for (int n = 0; n < config.hidden_size / config.n_head; n++) {
          fuse_qkv[i][j][m * (config.hidden_size / config.n_head) * 3 + config.hidden_size / config.n_head + n] = key_layer[i][j][m][n];
        }
        for (int n = 0; n < config.hidden_size / config.n_head; n++) {
          fuse_qkv[i][j][m * (config.hidden_size / config.n_head) * 3 + 2 * (config.hidden_size / config.n_head) + n] = value_layer[i][j][m][n];
        }
      }
    }
  }
  std::cout << "嘿嘿3～" << std::endl;
  if(write) WriteTensor(vec4d_t(1, fuse_qkv), PATHBACK + weight_bias_name + ".query_key_value_in.txt");
  if(berr) Error(vec4d_t(1, fuse_qkv), PATH_BACK + weight_bias_name + ".query_key_value_in.txt");

  vec2d_t qkv_weight(config.hidden_size * 3, vec1d_t(config.hidden_size, 0.0));
  vec1d_t qkv_bias(config.hidden_size*3, 0.0);
  if (config.training)
  {
    std::string name = weight_bias_name + ".query_key_value.weight";
    std::cout << "Read WeightMap: " << name << std::endl;
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name << std::endl;
    qkv_weight = WeightMap[name].v2d;

    name = weight_bias_name + ".query_key_value.bias";
    std::cout << "Read WeightMap: " << name << std::endl;
    if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name << std::endl;
    qkv_bias = WeightMap[name].v1d;
  } else {
    LoadInput(qkv_weight, PATH + weight_bias_name + ".query_key_value.weight.txt");
    LoadInput(qkv_bias, PATH + weight_bias_name + ".query_key_value.bias.txt");
  }

  // return fuse_qkv;
  std::cout << "attention_forward_in: [" << attention_forward_in.size() << ", " << attention_forward_in[0].size() << ", " << attention_forward_in[0][0].size() << "]\n";
  std::cout << "fuse_qkv: [" << fuse_qkv.size() << ", " << fuse_qkv[0].size() << ", " << fuse_qkv[0][0].size() << "]\n";
  
  if(write) WriteTensor(vec4d_t(1, fuse_qkv), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/build/testdata/560m/cpp的输入/qkv_in.txt");
  if (tran)
  {
    transition(fuse_qkv);
    transition(qkv_weight);
    transition(attention_forward_in);
  }
  vec3d_t qkv_dx = LinearDx(vec4d_t(1, fuse_qkv), qkv_weight)[0];

  if(write) WriteTensor(vec4d_t(1, qkv_dx), PATHBACK + weight_bias_name + ".query_key_value_out.txt");
  if(berr) Error(vec4d_t(1, qkv_dx), PATH_BACK + weight_bias_name + ".query_key_value_out.txt");
  vec2d_t qkv_dw = LinearDw(vec4d_t(1, attention_forward_in), vec4d_t(1, fuse_qkv));
  vec1d_t qkv_db = LinearDb(vec4d_t(1, fuse_qkv));
  if (config.training)
  {
    std::string name = weight_bias_name + ".query_key_value.weight";
    UpdateWeight(config, qkv_weight, qkv_dw);
    if (update) {
      WriteTensor(qkv_weight, PATHWeight + weight_bias_name + ".query_key_value.weight.txt");
      // WriteTensor(qkv_dw, PATHUPDATE + weight_bias_name + ".query_key_value.weight.txt");
      // Error_grad(qkv_dw, PATHGRAD + name + ".txt");
    }
    WeightMap[name].v2d = qkv_weight;

    name = weight_bias_name + ".query_key_value.bias";
    UpdateBias(config, qkv_bias, qkv_db);
    if (update) {
      WriteTensor(qkv_bias, PATHWeight + weight_bias_name + ".query_key_value.bias.txt");
      // WriteTensor(qkv_db, PATHUPDATE + weight_bias_name + ".query_key_value.bias.txt");
      // Error_grad(vec2d_t(1, qkv_db), PATHGRAD + name + ".txt");
      std::cout << "name: " << weight_bias_name << std::endl;
      // WriteTensor(qkv_db, "output.txt");
    }
    WeightMap[name].v1d = qkv_bias;
  }

  return qkv_dx;
}

vec3d_t BLOOMMLPBackward(BLOOMConfig config, vec3d_t backward_input, std::string weights_path)
{
    uint32_t hidden_size = config.hidden_size;

    vec2d_t weights_proj(backward_input[0][0].size(), vec1d_t(4 * hidden_size, 0.0));
    
    if (config.training)
    {
      std::string name = weights_path + ".dense_4h_to_h.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      weights_proj = WeightMap[name].v2d;
    }
    else
    {
      LoadInput(weights_proj, PATH + weights_path + ".dense_4h_to_h.weight.txt");
    }

    if(write) WriteTensor(vec4d_t(1, backward_input), PATHBACK + weights_path + "._in.txt");
    if(berr) Error(vec4d_t(1, backward_input), PATH_BACK + weights_path + "_in.txt");
    if(write) WriteTensor(vec4d_t(1, backward_input), PATHBACK + weights_path + ".dense_4h_to_h_in.txt");
    if(berr) Error(vec4d_t(1, backward_input), PATH_BACK + weights_path + ".dense_4h_to_h_in.txt");
    vec4d_t proj_dx = LinearDx(vec4d_t(1, backward_input), weights_proj);
    if(write) WriteTensor(proj_dx, PATHBACK + weights_path + ".dense_4h_to_h_out.txt");
    if(berr) Error(proj_dx, PATH_BACK + weights_path + ".dense_4h_to_h_out.txt");
    
    vec4d_t forward_proj(1, vec3d_t(backward_input.size(), vec2d_t(backward_input[0].size(), vec1d_t(4 * hidden_size))));
    if (config.training)
    {
      std::string name = weights_path + ".dense_4h_to_h_in";
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_proj = ForwardMap[name].v4d;
    }
    else
    {
      LoadInput(forward_proj, PATH_TEST + weights_path + ".dense_4h_to_h" + "_in.txt");
    }
    
    vec4d_t proj_dw = LinearBackDw(forward_proj, vec4d_t(1, backward_input));
    vec1d_t proj_db = LinearBackDb(vec4d_t(1, backward_input));

    if (config.training)
    {
      std::string name = weights_path + ".dense_4h_to_h.weight";
      std::cout << "proj_dw: " << proj_dw[0][0].size() << ' ' << proj_dw[0][0][0].size() << std::endl;
      UpdateWeight(config, weights_proj, proj_dw[0][0]);
      if (update) {
        WriteTensor(weights_proj, PATHWeight + weights_path + ".dense_4h_to_h.weight.txt");
        // WriteTensor(proj_dw, PATHUPDATE + weights_path + ".dense_4h_to_h.weight.txt");
        // Error_grad(proj_dw[0][0], PATHGRAD + name + ".txt");
      }
      WeightMap[name].v2d = weights_proj;
    
      name = weights_path + ".dense_4h_to_h.bias";
      vec1d_t bias_proj = WeightMap[name].v1d;
      UpdateBias(config, bias_proj, proj_db);
      if (update) {
        WriteTensor(bias_proj, PATHWeight + weights_path + ".dense_4h_to_h.bias.txt");
        // WriteTensor(proj_db, PATHUPDATE + weights_path + ".dense_4h_to_h.bias.txt");
        // Error_grad(vec2d_t(1, proj_db), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = bias_proj;
    }


    vec4d_t gelu_for(1, vec3d_t(proj_dx[0].size(), vec2d_t(proj_dx[0][0].size(), vec1d_t(proj_dx[0][0][0].size()))));
    if (config.training)
    {
      std::string name = weights_path + ".gelu_impl_in";
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      gelu_for = ForwardMap[name].v4d;
    }
    else
    {
      LoadInput(gelu_for, PATH_TEST + weights_path + ".gelu_impl" + "_in.txt");  
    }

    if(write) WriteTensor(proj_dx, PATHBACK + weights_path + ".gelu_impl_in.txt");
    if(berr) Error(proj_dx, PATH_BACK + weights_path + ".gelu_impl_in.txt");
    vec3d_t gelu_back = BACK_NewGELUActivation(gelu_for, proj_dx);
    if(write) WriteTensor(vec4d_t(1, gelu_back), PATHBACK + weights_path + ".gelu_impl_out.txt");
    if(berr) Error(vec4d_t(1, gelu_back), PATH_BACK + weights_path + ".gelu_impl_out.txt");
  

    vec2d_t weights_fc(gelu_back[0][0].size(), vec1d_t(hidden_size, 0.0));
    if (config.training)
    {
      std::string name = weights_path + ".dense_h_to_4h.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      weights_fc = WeightMap[name].v2d;
    }
    else
    {
      LoadInput(weights_fc, PATH + weights_path + ".dense_h_to_4h.weight.txt");
    }

    if(write) WriteTensor(vec4d_t(1, gelu_back), PATHBACK + weights_path + ".dense_h_to_4h_in.txt");
    if(berr) Error(vec4d_t(1, gelu_back), PATH_BACK + weights_path + ".dense_h_to_4h_in.txt");
    vec4d_t fc_dx = LinearBackDx(vec4d_t(1, gelu_back), vec4d_t(1, vec3d_t(1, weights_fc)));
    if(write) WriteTensor(fc_dx, PATHBACK + weights_path + ".dense_h_to_4h_out.txt");
    if(berr) Error(fc_dx, PATH_BACK + weights_path + ".dense_h_to_4h_out.txt");
    vec4d_t forward_fc(1, vec3d_t(gelu_back.size(), vec2d_t(gelu_back[0].size(), vec1d_t(hidden_size))));
    if (config.training)
    {
      std::string name = weights_path + ".dense_h_to_4h_in";
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_fc = ForwardMap[name].v4d;
    }
    else
    {
      LoadInput(forward_fc, PATH_TEST + weights_path + ".dense_h_to_4h" + "_in.txt");
    }
    vec4d_t fc_dw = LinearBackDw(forward_fc, vec4d_t(1, gelu_back));
    vec1d_t fc_db = LinearBackDb(vec4d_t(1, gelu_back));
    
    if (config.training)
    {
      std::string name = weights_path + ".dense_h_to_4h.weight";
      std::cout << "fc_dw: " << fc_dw[0][0].size() << ' ' << fc_dw[0][0][0].size() << std::endl;
      UpdateWeight(config, weights_fc, fc_dw[0][0]);
      if (update) {
        WriteTensor(weights_fc, PATHWeight + weights_path + ".dense_h_to_4h.weight.txt");
        // WriteTensor(fc_dw, PATHUPDATE + weights_path + ".dense_h_to_4h.weight.txt");
        // Error_grad(fc_dw[0][0], PATHGRAD + name + ".txt");
      }
      WeightMap[name].v2d = weights_fc;
      
      std::cout << "嘿嘿~" << std::endl;

      name = weights_path + ".dense_h_to_4h.bias";
      vec1d_t bias_fc = WeightMap[name].v1d;
      UpdateBias(config, bias_fc, fc_db);
      if (update) {
        WriteTensor(bias_fc, PATHWeight + weights_path + ".dense_h_to_4h.bias.txt");
        // WriteTensor(fc_db, PATHUPDATE + weights_path + ".dense_h_to_4h.bias.txt");
        // Error_grad(vec2d_t(1, fc_db), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = bias_fc;
    }

    return fc_dx[0];
}

vec4d_t BLOOMBlockBackward(BLOOMConfig config, vec4d_t dx, std::string weights_path)
{
    vec4d_t mlp_dx = vec4d_t(1, BLOOMMLPBackward(config, dx[0], weights_path + ".mlp"));
    if(write) WriteTensor(mlp_dx, PATHBACK + weights_path + ".mlp_out.txt");
    if(berr) Error(mlp_dx, PATH_BACK + weights_path + ".mlp_out.txt");
    vec4d_t forward_in(mlp_dx.size(), vec3d_t(mlp_dx[0].size(), vec2d_t(mlp_dx[0][0].size(), vec1d_t(mlp_dx[0][0][0].size()))));
    if (config.training)
    {
      std::string name = weights_path + ".post_attention_layernorm_in";
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_in = ForwardMap[name].v4d;
    }
    else
    {
      std::cout << "size: " << mlp_dx.size() << " " << mlp_dx[0].size() << " " << mlp_dx[0][0].size() << " " << mlp_dx[0][0][0].size() << std::endl;
      LoadInput(forward_in, PATH_TEST + weights_path + ".post_attention_layernorm" + "_in.txt");
    }

    vec1d_t forward_weight(config.hidden_size);
    vec1d_t forward_bias(config.hidden_size);
    if (config.training)
    {
      std::string name = weights_path + ".post_attention_layernorm.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_weight = WeightMap[name].v1d;

      name = weights_path + ".post_attention_layernorm.bias";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_bias = WeightMap[name].v1d;
    }
    else 
    {
      LoadInput(forward_weight, PATH + weights_path + ".post_attention_layernorm.weight.txt");
      LoadInput(forward_bias, PATH + weights_path + ".post_attention_layernorm.bias.txt");
    }
    if(write) WriteTensor(mlp_dx, PATHBACK + weights_path + ".post_attention_layernorm_inc.txt");
    if(berr) Error(mlp_dx, PATH_BACK + weights_path + ".post_attention_layernorm_in.txt");
    vec4d_t pal_dx = BLOOMLayerNormDxBackward(forward_in, mlp_dx, forward_weight, forward_bias, config.layer_norm_epsilon);
    if(write) WriteTensor(pal_dx, PATHBACK + weights_path + ".post_attention_layernorm_out.txt");
    if(berr) Error(pal_dx, PATH_BACK + weights_path + ".post_attention_layernorm_out.txt");



    vec1d_t pal_dw = BLOOMLayerNormDwBackward(forward_in, mlp_dx, config.layer_norm_epsilon);
    vec1d_t pal_db = BLOOMLayerNormDbBackward(mlp_dx);
    if (config.training)
    {
      std::string name = weights_path + ".post_attention_layernorm.weight";
      UpdateWeight(config, forward_weight, pal_dw);
      if (update) {
        WriteTensor(forward_weight, PATHWeight + weights_path + ".post_attention_layernorm.weight.txt");
        // WriteTensor(pal_dw, PATHUPDATE + weights_path + ".post_attention_layernorm.weight.txt");
        // Error_grad(vec2d_t(1, pal_dw), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = forward_weight;

      name = weights_path + ".post_attention_layernorm.bias";
      UpdateBias(config, forward_bias, pal_db);
      if (update) {
        WriteTensor(forward_bias, PATHWeight + weights_path + ".post_attention_layernorm.bias.txt");
        // WriteTensor(pal_db, PATHUPDATE + weights_path + ".post_attention_layernorm.bias.txt");
        // Error_grad(vec2d_t(1, pal_db), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = forward_bias;
    }
    /*
        BLOOKAttentionBack
    */
    pal_dx = AddVector(pal_dx, dx);

    if(write) WriteTensor(pal_dx, PATHBACK + weights_path + ".self_attention_in.txt");
    if(berr) Error(pal_dx, PATH_BACK + weights_path + ".self_attention_in.txt");
    vec4d_t attn(pal_dx.size(), vec3d_t(pal_dx[0].size(), vec2d_t(pal_dx[0][0].size(), vec1d_t(pal_dx[0][0][0].size()))));
    attn[0] = BLOOMAttentionBackward(config, pal_dx[0], false, false, weights_path + ".self_attention");
    // LoadInput(attn, PATH_BACK + weights_path + ".input_layernorm" + "_in.txt");
    if(write) WriteTensor(attn, PATHBACK + weights_path + ".self_attention_out.txt");
    if(berr) Error(attn, PATH_BACK + weights_path + ".self_attention_out.txt");

    vec4d_t forward_sec(1, vec3d_t(1, vec2d_t(attn[0][0].size(), vec1d_t(attn[0][0][0].size()))));
    if (config.training)
    {
      std::string name = weights_path + ".input_layernorm_in";
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_sec = ForwardMap[name].v4d;
    }
    else
    {
      LoadInput(forward_sec, PATH_TEST + weights_path + ".input_layernorm" + "_in.txt");
    }
    vec1d_t forward_weight_sec(config.hidden_size);
    vec1d_t forward_bias_sec(config.hidden_size);

    if (config.training)
    {
      std::string name = weights_path + ".input_layernorm.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_weight_sec = WeightMap[name].v1d;

      name = weights_path + ".input_layernorm.bias";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      forward_bias_sec = WeightMap[name].v1d;
    }
    else
    {
      LoadInput(forward_weight_sec, PATH + weights_path + ".input_layernorm.weight.txt");
      LoadInput(forward_bias_sec, PATH + weights_path + ".input_layernorm.bias.txt");
    }
    if(write) WriteTensor(attn, PATHBACK + weights_path + ".input_layernorm_in.txt");
    if(berr) Error(attn, PATH_BACK + weights_path + ".input_layernorm_in.txt");
    vec4d_t inl_dx = BLOOMLayerNormDxBackward(forward_sec, attn, forward_weight_sec, forward_bias_sec, config.layer_norm_epsilon);
    vec1d_t inl_dw = BLOOMLayerNormDwBackward(forward_sec, attn, config.layer_norm_epsilon);
    vec1d_t inl_db = BLOOMLayerNormDbBackward(attn);
    if(write) WriteTensor(inl_dx, PATHBACK + weights_path + ".input_layernorm_out.txt");
    if(berr) Error(inl_dx, PATH_BACK + weights_path + ".input_layernorm_out.txt");
    if (config.training)
    {
      std::string name = weights_path + ".input_layernorm.weight";
      UpdateWeight(config, forward_weight_sec, inl_dw);
      if (update) {
        WriteTensor(forward_weight_sec, PATHWeight + weights_path + ".input_layernorm.weight.txt");
        // WriteTensor(inl_dw, PATHUPDATE + weights_path + ".input_layernorm.weight.txt");
        // Error_grad(vec2d_t(1, inl_dw), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = forward_weight_sec;

      name = weights_path + ".input_layernorm.bias";
      UpdateBias(config, forward_bias_sec, inl_db);
      if (update) {
        WriteTensor(forward_bias_sec, PATHWeight + weights_path + ".input_layernorm.bias.txt");
        // WriteTensor(inl_db, PATHUPDATE + weights_path + ".input_layernorm.bias.txt");
        // Error_grad(vec2d_t(1, inl_db), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = forward_bias_sec;
    }

    inl_dx = AddVector(pal_dx, inl_dx);
    // for (auto x : inl_db)
    // {
    //     std::cout << x << std::endl;
    // }
    return inl_dx;
}

vec4d_t BLOOMModelBackward(BLOOMConfig config, vec4d_t lm_logits, std::string weights_path)
{
    std::cout << "BLOOM Backward" << std::endl;
    vec4d_t ln_f_in(1, vec3d_t(1, vec2d_t(lm_logits[0][0].size(), vec1d_t(lm_logits[0][0][0].size()))));
    if (config.training)
    {
      std::string name = weights_path + ".ln_f_in";
      std::cout << "Read ForwardMap" << ' ' << name <<  std::endl;
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      ln_f_in = ForwardMap[name].v4d;
    }
    else
    {
      LoadInput(ln_f_in, PATH_TEST + weights_path + ".ln_f" + "_in.txt");
    }
    if(write) WriteTensor(lm_logits, PATHBACK + weights_path + ".ln_f_in.txt");
    if(berr) Error(lm_logits, PATH_BACK + weights_path + ".ln_f_in.txt");
    vec1d_t ln_f_weight(config.hidden_size);
    vec1d_t ln_f_bias(config.hidden_size);

    if (config.training)
    {
      std::string name = weights_path + ".ln_f.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      ln_f_weight = WeightMap[name].v1d;

      name = weights_path + ".ln_f.bias";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      ln_f_bias = WeightMap[name].v1d;
    }
    else
    {
      LoadInput(ln_f_weight, PATH + weights_path + ".ln_f.weight.txt");
      LoadInput(ln_f_bias, PATH + weights_path + ".ln_f.bias.txt");
    }
    vec4d_t ln_f_dx = BLOOMLayerNormDxBackward(ln_f_in, lm_logits, ln_f_weight, ln_f_bias, config.layer_norm_epsilon);
    if(write) WriteTensor(ln_f_dx, PATHBACK + weights_path + ".ln_f_out.txt");
    if(berr) Error(ln_f_dx, PATH_BACK + weights_path + ".ln_f_out.txt");
    vec1d_t ln_f_dw = BLOOMLayerNormDwBackward(ln_f_in, lm_logits, config.layer_norm_epsilon);
    vec1d_t ln_f_db = BLOOMLayerNormDbBackward(lm_logits);
    if (config.training)
    {
      std::string name = weights_path + ".ln_f.weight";
      UpdateWeight(config, ln_f_weight, ln_f_dw);
      if (update) {
        WriteTensor(ln_f_weight, PATHWeight + weights_path + ".ln_f.weight.txt");
        // WriteTensor(ln_f_dw, PATHUPDATE + weights_path + ".ln_f.weight.txt");
        // Error_grad(vec2d_t(1, ln_f_dw), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = ln_f_weight;

      name = weights_path + ".ln_f.bias";
      UpdateBias(config, ln_f_bias, ln_f_db);
      if (update) {
        WriteTensor(ln_f_bias, PATHWeight + weights_path + ".ln_f.bias.txt");
        // WriteTensor(ln_f_db, PATHUPDATE + weights_path + ".ln_f.bias.txt");
        // Error_grad(vec2d_t(1, ln_f_db), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = ln_f_bias;
    }


    // for(auto i : ln_f_db)
    // {
    //   std::cout << i << std::endl;
    // }


    lm_logits = ln_f_dx;
    
    for (int i = 23; i >= 0; i--)
    {
      if(write) WriteTensor(lm_logits, PATHBACK + weights_path + ".h." + std::to_string(i) + "_in.txt");
      if(berr) Error(lm_logits, PATH_BACK + weights_path + ".h." + std::to_string(i) + "_in.txt");
      lm_logits = BLOOMBlockBackward(config, lm_logits, weights_path + ".h." + std::to_string(i));
      if(write) WriteTensor(lm_logits, PATHBACK + weights_path + ".h." + std::to_string(i) + "_out.txt");
      if(berr) Error(lm_logits, PATH_BACK + weights_path + ".h." + std::to_string(i) + "_out.txt");
      if(i ==23) exit(1);
    }
    // LoadInput(lm_logits, PATH_BACK + weights_path + ".word_embeddings_layernorm_in.txt");

    vec4d_t embedding_in(lm_logits.size(), vec3d_t(lm_logits[0].size(), vec2d_t(lm_logits[0][0].size(), vec1d_t(lm_logits[0][0][0].size()))));
    if (config.training)
    {
      std::string name = weights_path + ".word_embeddings_layernorm";
      std::cout << "Read ForwardMap" << ' ' << name <<  std::endl;
      if (ForwardMap.find(name) == ForwardMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      embedding_in = ForwardMap[name].v4d;
    }
    else
    {
     LoadInput(embedding_in, PATH_TEST + weights_path + ".word_embeddings_layernorm" + "_in.txt");
    }
    
    vec1d_t embedding_weight(config.hidden_size);
    vec1d_t embedding_bias(config.hidden_size);
    if (config.training)
    {
      std::string name = weights_path + ".word_embeddings_layernorm.weight";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      embedding_weight = WeightMap[name].v1d;

      name = weights_path + ".word_embeddings_layernorm.bias";
      std::cout << "Read WeightMap" << ' ' << name <<  std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name <<  std::endl;
      embedding_bias = WeightMap[name].v1d;
    }
    else
    {
      LoadInput(embedding_weight, PATH + weights_path + ".word_embeddings_layernorm.weight.txt");
      LoadInput(embedding_bias, PATH + weights_path + ".word_embeddings_layernorm.bias.txt");
    }
    if(write) WriteTensor(lm_logits, PATHBACK + weights_path + ".word_embeddings_layernorm_in.txt");
    if(berr) Error(lm_logits, PATH_BACK + weights_path + ".word_embeddings_layernorm_in.txt");
    vec4d_t embedding_dx = BLOOMLayerNormDxBackward(embedding_in, lm_logits, embedding_weight, embedding_bias, config.layer_norm_epsilon);
    vec1d_t embedding_dw = BLOOMLayerNormDwBackward(embedding_in, lm_logits, config.layer_norm_epsilon);
    vec1d_t embedding_db = BLOOMLayerNormDbBackward(lm_logits);
    if(write) WriteTensor(embedding_dx, PATHBACK + weights_path + ".word_embeddings_layernorm_out.txt");
    if(berr) Error(embedding_dx, PATH_BACK + weights_path + ".word_embeddings_layernorm_out.txt");
    if (config.training)
    {
      std::string name = weights_path + ".word_embeddings_layernorm.weight";
      UpdateWeight(config, embedding_weight, embedding_dw);
      if (update) {
        WriteTensor(embedding_weight, PATHWeight + weights_path + ".word_embeddings_layernorm.weight.txt");
        // WriteTensor(embedding_dw, PATHUPDATE + weights_path + ".word_embeddings_layernorm.weight.txt");
        // Error_grad(vec2d_t(1, embedding_dw), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = embedding_weight;

      name = weights_path + ".word_embeddings_layernorm.bias";
      UpdateWeight(config, embedding_bias, embedding_db);
      if (update) {
        WriteTensor(embedding_bias, PATHWeight + weights_path + ".word_embeddings_layernorm.bias.txt");
        // WriteTensor(embedding_db, PATHUPDATE + weights_path + ".word_embeddings_layernorm.bias.txt");
        // Error_grad(vec2d_t(1, embedding_db), PATHGRAD + name + ".txt");
      }
      WeightMap[name].v1d = embedding_bias;
    }
    

    return embedding_dx;
}



vec4d_t BLOOMForCausalLMBackward(BLOOMConfig config, vec4d_t hidden_states, std::string weights_path)
{
  uint32_t hidden_size = config.hidden_size;

  vec2d_t weights(config.vocab_size, vec1d_t(hidden_size, 0.0));
  LoadInput(weights, PATH + weights_path + ".ln_head.weights.txt");

  vec4d_t ln_dx = LinearBackDx(hidden_states, vec4d_t(1, vec3d_t(1, weights)));

  return ln_dx;
}


float BLOOMTrainingStep(BLOOMConfig config, BLOOMTraining train, vec2d_t inputs, vec1d_t_i labels, vec2d_t_i attention_mask)
{
  std::vector<std::tuple<vec3d_t, vec3d_t>> past_key_values;
  vec5d_t_i head_mask = {};
  vec3d_t input_embeds = {};
  bool use_cache = true;
  bool output_attentions = false;
  bool output_hidden_states = false;
  bool return_dict = true;
  
  config.training = true;

  if (zero_grad)
  {
    fillWightMapFromFile2("transformer.word_embeddings.weight", config.vocab_size, config.hidden_size);
    fillWightMapFromFile1("transformer.word_embeddings_layernorm.weight", config.hidden_size);
    fillWightMapFromFile1("transformer.word_embeddings_layernorm.bias", config.hidden_size);
    fillWightMapFromFile1("transformer.ln_f.weight", config.hidden_size);
    fillWightMapFromFile1("transformer.ln_f.bias", config.hidden_size);
    fillWightMapFromFile2("transformer.lm_head.weight", config.vocab_size, config.hidden_size);
    for (int i = 0; i < 24; i++)
    {
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".input_layernorm.weight", config.hidden_size);
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".input_layernorm.bias", config.hidden_size);
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".post_attention_layernorm.weight", config.hidden_size);
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".post_attention_layernorm.bias", config.hidden_size);
        fillWightMapFromFile2("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight", config.hidden_size * 3, config.hidden_size);
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".self_attention.query_key_value.bias", config.hidden_size * 3);
        fillWightMapFromFile2("transformer.h." + std::to_string(i) + ".self_attention.dense.weight", config.hidden_size, config.hidden_size);
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".self_attention.dense.bias", config.hidden_size);
        fillWightMapFromFile2("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight", config.hidden_size * 4, config.hidden_size);
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.bias", config.hidden_size * 4);
        fillWightMapFromFile2("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight", config.hidden_size, config.hidden_size * 4);
        fillWightMapFromFile1("transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.bias", config.hidden_size);
    }
    fillWightMapFromFile2("transformer.lm_head.weight", config.vocab_size, config.hidden_size);
  }
  
  std::tuple<vec3d_t, std::vector<std::tuple<vec3d_t, vec3d_t>>, std::vector<vec3d_t>, std::vector<vec4d_t>> 
  output = BLOOMModel(config, inputs, past_key_values, attention_mask, head_mask, input_embeds, use_cache, output_attentions, output_hidden_states, return_dict, "transformer");

  vec3d_t hidden_states = std::get<0>(output);

  vec2d_t lm_logits_weights(config.vocab_size, vec1d_t(config.hidden_size, 0.0));
  if (config.training)
  {
      std::string name = "transformer.lm_head.weight";
      std::cout << "Read WeightMap: " << name << std::endl;
      if (WeightMap.find(name) == WeightMap.end()) std::cout << "Training Read: " << name << std::endl;
      lm_logits_weights = WeightMap[name].v2d;
  } else {
      load_input(lm_logits_weights, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/bloom_560m_f32/transformer.lm_head.weight.txt", 250880 * 1024);
  }
  if (berr) Error(vec4d_t(1, hidden_states), PATH_FOR + "transformer.lm_head_in.txt");
  vec2d_t lm_head_output = Linear_Nobias(vec4d_t(1, hidden_states), lm_logits_weights)[0][0];
  if (berr) Error(vec4d_t(1, vec3d_t(1, lm_head_output)), PATH_FOR + "transformer.lm_head_out.txt");
  // Error(vec4d_t(1, vec3d_t(1, lm_head_output)), "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_forward_f32/lm_head_out.txt");
  for (int j = 0; j < lm_head_output[0].size(); j++)
  {
      lm_head_output[lm_head_output.size() - 1][j] = 0;
  }

  for (int i = 0; i < labels.size() - 1; i++)
  {
    labels[i] = labels[i + 1];
  }
  labels[labels.size() - 1] = 0; 
  vec2d_t softmax_lm_logits(lm_head_output.size(), vec1d_t(lm_head_output[0].size(), 0.0));//13 50257
  for (int i = 0; i < softmax_lm_logits.size() - 1; i++) {
    if (labels[i] <= 0) continue;
    softmax_lm_logits[i] = logsoftmax(lm_head_output[i]);
  }
  // if (berr) Error(vec4d_t(1, vec3d_t(1, softmax_lm_logits)), PATH_FOR + "transformer.softmax_out.txt");
  float loss = 0.0;
  int loss_num = 0;
  for (int i = 0; i < labels.size() - 1; ++i)
  {
      if (labels[i] <= 0) continue;
      loss_num++;
      loss += std::abs(softmax_lm_logits[i][labels[i]]);
  }
  loss /= (float)loss_num;
  std::cout << "loss: " << loss << std::endl;

  for (int i = 0; i < labels.size(); ++i) 
  {
    if (labels[i] <= 0) continue;
    for (int j = 0; j < softmax_lm_logits[0].size(); ++j) softmax_lm_logits[i][j] = std::exp(softmax_lm_logits[i][j]);
    softmax_lm_logits[i][labels[i]] -= 1.0;
    for (int j = 0; j < softmax_lm_logits[0].size(); j++) {
        softmax_lm_logits[i][j] = softmax_lm_logits[i][j] / (float)(loss_num * train.gradient_accumulation_steps);
    }
  }

  std::cout << "detch loss: " << loss / (float)train.gradient_accumulation_steps << std::endl;
  
  // vec2d_t softmax_lm_logits(115, vec1d_t(250880, 0.0));
  // load_input(softmax_lm_logits, "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/lm_head_in.txt", 115 * 250880);

  if (berr) Error(vec4d_t(1, vec3d_t(1, softmax_lm_logits)), "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/lm_head_in.txt");
  // WriteTensor(softmax_lm_logits, "lm_head_in.txt");
  vec3d_t input_backward =  LinearDx(vec4d_t(1, vec3d_t(1, softmax_lm_logits)), lm_logits_weights)[0];
  // vec2d_t dense_dw = LinearDw(vec4d_t(1, hidden_states), vec4d_t(1, vec3d_t(1, softmax_lm_logits)));
  // UpdateWeight(config, lm_logits_weights, dense_dw);
  // WriteTensor(lm_logits_weights, PATHUPDATE + "transformer.lm_head.weight");
  // // WriteTensor(input_backward, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/backward_560m/transformer.lm_head_out.txt");
  if (berr) Error(vec4d_t(1, input_backward), "/home/yinxun/phoenix-inst-chat-7b-DLC/phoenix-cpp/test/bloom_560m_backward_f32/lm_head_out.txt");
  
  // // vec3d_t input_backward(1, vec2d_t(41, vec1d_t(1024, 0.0)));
  // // load_input(input_backward, "/home/yinxun/workspace/phoenix-inst-chat-7b-DLC/phoenix-cpp/backward_7b/transformer.h.29_out.txt", 41 * 1024);

  vec4d_t back = BLOOMModelBackward(config, vec4d_t(1, input_backward), "transformer");

  return loss / (float)train.gradient_accumulation_steps;
  // return 0;
}

float BLOOMTrain(BLOOMConfig config, BLOOMTraining training, std::vector<vec> train_dataloader)
{
  
  bool is_in_train = true;
  int _train_batch_size = training.per_device_train_batch_size;


  std::string resume_checkpoint;
  if(training.resume_from_checkpoint) resume_checkpoint = get_last_checkpoint(training.output_dir); 

  int total_train_batch_size = _train_batch_size * training.gradient_accumulation_steps;

  int len_dataloader;
  int num_update_steps_per_epoch;
  int max_steps;
  int num_train_epochs;
  int num_train_samples;
  // -> train_dataloader = self.get_train_dataloader()
  // int train_dataloader_size = train_dataloader.size();
  if (train_dataloader.size())
  { 
    len_dataloader = train_dataloader.size();
    num_update_steps_per_epoch = len_dataloader / training.gradient_accumulation_steps;
    num_update_steps_per_epoch = std::max(num_update_steps_per_epoch, 1);
    if (training.max_steps > 0)
    {
      max_steps = training.max_steps;
      num_train_epochs = training.max_steps / num_update_steps_per_epoch + (training.max_steps % num_update_steps_per_epoch > 0);
      num_train_samples = training.max_steps * total_train_batch_size;
    }
    else 
    {
      max_steps = std::ceil(training.num_train_epochs * num_update_steps_per_epoch);
      num_train_epochs = std::ceil(training.num_train_epochs);
      num_train_samples = train_dataloader.size() * training.num_train_epochs;
    }
  } 
  else if (training.max_steps > 0)
  {
    max_steps = training.max_steps;
    num_train_epochs = std::numeric_limits<int>::max();
    num_update_steps_per_epoch = max_steps;
    num_train_samples = training.max_steps * total_train_batch_size;
  }
  else {
    throw std::runtime_error("training.max_steps must be set to a positive value if dataloader does not have a length, was " + std::to_string(training.max_steps));
  }

  TrainerState trainstate;
  // if (trial.empty()) trainstate.is_hyper_param_search = false;
  // else trainstate.is_hyper_param_search = true;

  if (training.logging_steps != 0.0)
  {
    if (training.logging_steps < 1.0) trainstate.logging_steps = std::ceil((float)max_steps * training.logging_steps);
    else trainstate.logging_steps = training.logging_steps;
  }
  if (training.eval_steps != 0)
  {
    if (training.eval_steps < 1) trainstate.eval_steps = std::ceil((float)max_steps * training.eval_steps);
    else trainstate.eval_steps = training.eval_steps;
  }
  if (training.save_steps != 0)
  {
    if (training.save_steps < 1) trainstate.save_steps = std::ceil(max_steps * training.save_steps);
    else trainstate.save_steps = training.save_steps;
  }

  std::cout << "**********  Running training  **********" << std::endl;
  // std::cout << "   Num examples = " << train_dataloader << std::endl;
  std::cout << "   Num Epochs = " << num_train_epochs << std::endl;
  std::cout << "   Instantaneous batch size per device = " << training.per_device_train_batch_size << std::endl;
  if (training.per_device_train_batch_size != _train_batch_size) {
      std::cout << "   Training with DataParallel so batch size has been adjusted to: " << _train_batch_size << std::endl;
  } 
  std::cout << "   Total train batch size (w. parallel, distributed & accumulation) = " << total_train_batch_size << std::endl;
  std::cout << "   Gradient Accumulation steps = " << training.gradient_accumulation_steps << std::endl;
  std::cout << "   Total optimization steps = " << max_steps << std::endl;

  trainstate.epoch = 0;
  int epochs_trained = 0;
  int steps_trained_in_current_epoch = 0;
  // int steps_trained_progress_bar;

  if (resume_checkpoint.size() != 0 && std::filesystem::exists(resume_checkpoint + "/" + TRAINER_STATE_NAME))
  {
    trainstate = TrainerState::load_from_json(resume_checkpoint + "/" + TRAINER_STATE_NAME);
    epochs_trained = trainstate.global_step / num_update_steps_per_epoch;
    if (!training.ignore_data_skip)
    {
      steps_trained_in_current_epoch = trainstate.global_step % num_update_steps_per_epoch;
      steps_trained_in_current_epoch *= training.gradient_accumulation_steps;
    } else {
      steps_trained_in_current_epoch = 0;
    }

    std::cout << "   Continuing training from checkpoint, will skip to saved global_step" << std::endl;
    std::cout << "   Continuing training from epoch " << epochs_trained << std::endl;
    std::cout << "   Contimuing training from global step " << trainstate.global_step << std::endl;

    if (!training.ignore_data_skip) std::cout << "   Will skip the first " << epochs_trained << " epochs then the first " << steps_trained_in_current_epoch << " batches in the first epoch." << std::endl;
  }

  trainstate.max_steps = max_steps;
  trainstate.num_train_epochs = num_train_epochs;
  trainstate.is_local_process_zero = training.local_process_index == 0;
  trainstate.is_world_process_zero = training.process_index == 0;

  float loss = 0.0;

  float _total_loss_scalar = 0.0;
  int _globalstep_last_logged = trainstate.global_step;

  if (!training.ignore_data_skip)
  {
    for (int epoch = 0; epoch < epochs_trained; ++epoch)
    {
      for (auto batch : train_dataloader) break;
    }
  }

  int total_batched_samples = 0;
  for (int epoch = epochs_trained; epoch < num_train_epochs; ++epoch)
  {
    std::vector<vec>::iterator epoch_iterator = train_dataloader.begin();

    int steps_in_epoch = (len_dataloader != 0) ? std::distance(epoch_iterator, train_dataloader.end()) : training.max_steps * training.gradient_accumulation_steps;

    if (epoch == epochs_trained && resume_checkpoint != "" && steps_trained_in_current_epoch == 0) {}

    bool rng_to_sync = false;
    int steps_skipped = 0;
    if (steps_trained_in_current_epoch > 0)
    {
      std::advance(epoch_iterator, steps_trained_in_current_epoch);
      steps_skipped = steps_trained_in_current_epoch;
      steps_trained_in_current_epoch = 0;
      rng_to_sync = true;
    }

    int step = -1;
    std::ptrdiff_t distanceToEnd = std::distance(epoch_iterator, train_dataloader.end());
    for (epoch_iterator; epoch_iterator != train_dataloader.end(); ++epoch_iterator)
    {
      total_batched_samples += 1;
      if (rng_to_sync) {}

      if (steps_trained_in_current_epoch > 0) {
        steps_trained_in_current_epoch -= 1;
        // if (steps_trained_progress_bar) {}
        if (steps_trained_in_current_epoch == 0) {}
        continue;
      }
      float tr_loss_step = BLOOMTrainingStep(config, training, epoch_iterator->input, epoch_iterator->label, epoch_iterator->attention_mask);
      zero_grad = false;
      std::cout << "tr_loss_step: " << tr_loss_step << std::endl;
      if (training.logging_nan_inf_filter && tr_loss_step != 0.0 && tr_loss_step < 100.0) {
        loss += (loss / (float)(1 + trainstate.global_step - _globalstep_last_logged)); 
      } else {
        loss += tr_loss_step;
      }
      std::cout << "loss: " << loss << std::endl;
      bool is_last_step_and_steps_less_than_grad_acc = ((steps_in_epoch <= training.gradient_accumulation_steps) && ((step + 1) == steps_in_epoch));
      
      if ((total_batched_samples % training.gradient_accumulation_steps == 0) || is_last_step_and_steps_less_than_grad_acc) {
        zero_grad = true;
        trainstate.global_step += 1;
        trainstate.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch;
      }
    }
  }

  std::cout << "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n" << std::endl;

  _total_loss_scalar += loss;
  float train_loss = _total_loss_scalar / (float)trainstate.global_step;
  is_in_train = false;

  return train_loss;
}