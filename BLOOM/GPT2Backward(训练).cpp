#include"GPT2Backward.h"
using namespace std;

inline string PATH = "C:\\Users\\baidiao\\Desktop\\instructgpt\\weight\\";
inline string PRINT = "C:\\Users\\baidiao\\Desktop\\instructgpt\\c++_out\\";
inline string FORWARDPATH = "C:\\Users\\baidiao\\Desktop\\instructgpt\\MiddleValue\\";
inline string FPATH = "C:\\Users\\baidiao\\Desktop\\instructgpt\\MiddleValue\\";

std::map<std::string, std::any>ForwardMap;
std::map<std::string, std::any>WeightMap;

std::map<std::string, uint32_t>lengthMap ={
	{"lm_head_weight", 38597376},
	{"transformer_h_0_attn_bias", 1048576},
	{"transformer_h_0_attn_c_attn_bias", 2304},
	{"transformer_h_0_attn_c_attn_weight", 1769472},
	{"transformer_h_0_attn_c_proj_bias", 768},
	{"transformer_h_0_attn_c_proj_weight", 589824},
	{"transformer_h_0_ln_1_bias", 768},
	{"transformer_h_0_ln_1_weight", 768},
	{"transformer_h_0_ln_2_bias", 768},
	{"transformer_h_0_ln_2_weight", 768},
	{"transformer_h_0_mlp_c_fc_bias", 3072},
	{"transformer_h_0_mlp_c_fc_weight", 2359296},
	{"transformer_h_0_mlp_c_proj_bias", 768},
	{"transformer_h_0_mlp_c_proj_weight", 2359296},
	{"transformer_h_10_attn_bias", 1048576},
	{"transformer_h_10_attn_c_attn_bias", 2304},
	{"transformer_h_10_attn_c_attn_weight", 1769472},
	{"transformer_h_10_attn_c_proj_bias", 768},
	{"transformer_h_10_attn_c_proj_weight", 589824},
	{"transformer_h_10_ln_1_bias", 768},
	{"transformer_h_10_ln_1_weight", 768},
	{"transformer_h_10_ln_2_bias", 768},
	{"transformer_h_10_ln_2_weight", 768},
	{"transformer_h_10_mlp_c_fc_bias", 3072},
	{"transformer_h_10_mlp_c_fc_weight", 2359296},
	{"transformer_h_10_mlp_c_proj_bias", 768},
	{"transformer_h_10_mlp_c_proj_weight", 2359296},
	{"transformer_h_11_attn_bias", 1048576},
	{"transformer_h_11_attn_c_attn_bias", 2304},
	{"transformer_h_11_attn_c_attn_weight", 1769472},
	{"transformer_h_11_attn_c_proj_bias", 768},
	{"transformer_h_11_attn_c_proj_weight", 589824},
	{"transformer_h_11_ln_1_bias", 768},
	{"transformer_h_11_ln_1_weight", 768},
	{"transformer_h_11_ln_2_bias", 768},
	{"transformer_h_11_ln_2_weight", 768},
	{"transformer_h_11_mlp_c_fc_bias", 3072},
	{"transformer_h_11_mlp_c_fc_weight", 2359296},
	{"transformer_h_11_mlp_c_proj_bias", 768},
	{"transformer_h_11_mlp_c_proj_weight", 2359296},
	{"transformer_h_1_attn_bias", 1048576},
	{"transformer_h_1_attn_c_attn_bias", 2304},
	{"transformer_h_1_attn_c_attn_weight", 1769472},
	{"transformer_h_1_attn_c_proj_bias", 768},
	{"transformer_h_1_attn_c_proj_weight", 589824},
	{"transformer_h_1_attn_masked_bias", 0},
	{"transformer_h_1_ln_1_bias", 768},
	{"transformer_h_1_ln_1_weight", 768},
	{"transformer_h_1_ln_2_bias", 768},
	{"transformer_h_1_ln_2_weight", 768},
	{"transformer_h_1_mlp_c_fc_bias", 3072},
	{"transformer_h_1_mlp_c_fc_weight", 2359296},
	{"transformer_h_1_mlp_c_proj_bias", 768},
	{"transformer_h_1_mlp_c_proj_weight", 2359296},
	{"transformer_h_2_attn_bias", 1048576},
	{"transformer_h_2_attn_c_attn_bias", 2304},
	{"transformer_h_2_attn_c_attn_weight", 1769472},
	{"transformer_h_2_attn_c_proj_bias", 768},
	{"transformer_h_2_attn_c_proj_weight", 589824},
	{"transformer_h_2_ln_1_bias", 768},
	{"transformer_h_2_ln_1_weight", 768},
	{"transformer_h_2_ln_2_bias", 768},
	{"transformer_h_2_ln_2_weight", 768},
	{"transformer_h_2_mlp_c_fc_bias", 3072},
	{"transformer_h_2_mlp_c_fc_weight", 2359296},
	{"transformer_h_2_mlp_c_proj_bias", 768},
	{"transformer_h_2_mlp_c_proj_weight", 2359296},
	{"transformer_h_3_attn_bias", 1048576},
	{"transformer_h_3_attn_c_attn_bias", 2304},
	{"transformer_h_3_attn_c_attn_weight", 1769472},
	{"transformer_h_3_attn_c_proj_bias", 768},
	{"transformer_h_3_attn_c_proj_weight", 589824},
	{"transformer_h_3_ln_1_bias", 768},
	{"transformer_h_3_ln_1_weight", 768},
	{"transformer_h_3_ln_2_bias", 768},
	{"transformer_h_3_ln_2_weight", 768},
	{"transformer_h_3_mlp_c_fc_bias", 3072},
	{"transformer_h_3_mlp_c_fc_weight", 2359296},
	{"transformer_h_3_mlp_c_proj_bias", 768},
	{"transformer_h_3_mlp_c_proj_weight", 2359296},
	{"transformer_h_4_attn_bias", 1048576},
	{"transformer_h_4_attn_c_attn_bias", 2304},
	{"transformer_h_4_attn_c_attn_weight", 1769472},
	{"transformer_h_4_attn_c_proj_bias", 768},
	{"transformer_h_4_attn_c_proj_weight", 589824},
	{"transformer_h_4_ln_1_bias", 768},
	{"transformer_h_4_ln_1_weight", 768},
	{"transformer_h_4_ln_2_bias", 768},
	{"transformer_h_4_ln_2_weight", 768},
	{"transformer_h_4_mlp_c_fc_bias", 3072},
	{"transformer_h_4_mlp_c_fc_weight", 2359296},
	{"transformer_h_4_mlp_c_proj_bias", 768},
	{"transformer_h_4_mlp_c_proj_weight", 2359296},
	{"transformer_h_5_attn_bias", 1048576},
	{"transformer_h_5_attn_c_attn_bias", 2304},
	{"transformer_h_5_attn_c_attn_weight", 1769472},
	{"transformer_h_5_attn_c_proj_bias", 768},
	{"transformer_h_5_attn_c_proj_weight", 589824},
	{"transformer_h_5_ln_1_bias", 768},
	{"transformer_h_5_ln_1_weight", 768},
	{"transformer_h_5_ln_2_bias", 768 },
	{"transformer_h_5_ln_2_weight", 768 },
	{"transformer_h_5_mlp_c_fc_bias", 3072 },
	{"transformer_h_5_mlp_c_fc_weight", 2359296 },
	{"transformer_h_5_mlp_c_proj_bias", 768 },
	{"transformer_h_5_mlp_c_proj_weight", 2359296 },
	{"transformer_h_6_attn_bias", 1048576 },
	{"transformer_h_6_attn_c_attn_bias", 2304 },
	{"transformer_h_6_attn_c_attn_weight", 1769472 },
	{"transformer_h_6_attn_c_proj_bias", 768 },
	{"transformer_h_6_attn_c_proj_weight", 589824 },
	{"transformer_h_6_ln_1_bias", 768 },
	{"transformer_h_6_ln_1_weight", 768 },
	{"transformer_h_6_ln_2_bias", 768 },
	{"transformer_h_6_ln_2_weight", 768 },
	{"transformer_h_6_mlp_c_fc_bias", 3072 },
	{"transformer_h_6_mlp_c_fc_weight", 2359296 },
	{"transformer_h_6_mlp_c_proj_bias", 768 },
	{"transformer_h_6_mlp_c_proj_weight", 2359296 },
	{"transformer_h_7_attn_bias", 1048576 },
	{"transformer_h_7_attn_c_attn_bias", 2304 },
	{"transformer_h_7_attn_c_attn_weight", 1769472 },
	{"transformer_h_7_attn_c_proj_bias", 768 },
	{"transformer_h_7_attn_c_proj_weight", 589824 },
	{"transformer_h_7_ln_1_bias", 768 },
	{"transformer_h_7_ln_1_weight", 768 },
	{"transformer_h_7_ln_2_bias", 768},
	{"transformer_h_7_ln_2_weight", 768},
	{"transformer_h_7_mlp_c_fc_bias", 3072},
	{"transformer_h_7_mlp_c_fc_weight", 2359296},
	{"transformer_h_7_mlp_c_proj_bias", 768},
	{"transformer_h_7_mlp_c_proj_weight", 2359296},
	{"transformer_h_8_attn_bias", 1048576},
	{"transformer_h_8_attn_c_attn_bias", 2304},
	{"transformer_h_8_attn_c_attn_weight", 1769472},
	{"transformer_h_8_attn_c_proj_bias", 768},
	{"transformer_h_8_attn_c_proj_weight", 589824},
	{"transformer_h_8_ln_1_bias", 768},
	{"transformer_h_8_ln_1_weight", 768},
	{"transformer_h_8_ln_2_bias", 768},
	{"transformer_h_8_ln_2_weight", 768},
	{"transformer_h_8_mlp_c_fc_bias", 3072},
	{"transformer_h_8_mlp_c_fc_weight", 2359296},
	{"transformer_h_8_mlp_c_proj_bias", 768},
	{"transformer_h_8_mlp_c_proj_weight", 2359296},
	{"transformer_h_9_attn_bias", 1048576},
	{"transformer_h_9_attn_c_attn_bias", 2304},
	{"transformer_h_9_attn_c_attn_weight", 1769472},
	{"transformer_h_9_attn_c_proj_bias", 768},
	{"transformer_h_9_attn_c_proj_weight", 589824},
	{"transformer_h_9_ln_1_bias", 768},
	{"transformer_h_9_ln_1_weight", 768},
	{"transformer_h_9_ln_2_bias", 768},
	{"transformer_h_9_ln_2_weight", 768},
	{"transformer_h_9_mlp_c_fc_bias", 3072},
	{"transformer_h_9_mlp_c_fc_weight", 2359296},
	{"transformer_h_9_mlp_c_proj_bias", 768},
	{"transformer_h_9_mlp_c_proj_weight", 2359296},
	{"transformer_ln_f_bias", 768},
	{"transformer_ln_f_weight", 768},
	{"transformer_wpe_weight", 786432},
	{"transformer_wte_weight", 38597376}
};

tuple<vec4d_t, vec5d_t> GPT2Attention(GPT2Config config, vec3d_t hidden_states, string weights_path)
{
	uint32_t hidden_size = hidden_states[0][0].size();
	uint32_t embed_dim = hidden_size;
	uint32_t head_dim = embed_dim / config.num_heads;//64 = 768 / 12
	vec2d_t attn_c_attn_weight(3 * config.n_embd, vec1d_t(config.n_embd));
	if (config.is_training) {
		string name = weights_path + "_c_attn_forward_in";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, hidden_states);
	}
	vec2d_t c_attn_weight(config.n_embd, vec1d_t(3 * embed_dim));
	vec1d_t c_attn_bias(3 * embed_dim);
	if (config.is_training) {
		string name = weights_path + "_c_attn_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_attn_weight = std::any_cast<vec2d_t>(WeightMap[name]);

		name = weights_path + "_c_attn_bias";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_attn_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	else {
		LoadInput(c_attn_weight, weights_path + "_c_attn_weight.txt");
		LoadInput(c_attn_bias, weights_path + "_c_attn_bias.txt");
	}
	//vec4d_t qkv = Conv1D(vec4d_t(1, hidden_states), 3 * embed_dim, PATH + weights_path + "_c_attn_weight.txt", PATH + weights_path + "_c_attn_bias.txt");//[1 key_length 768] [768 2304] = [1 key_length 2304]
	vec4d_t qkv = Conv1D(vec4d_t(1, hidden_states), vec4d_t(1, vec3d_t(1, c_attn_weight)), c_attn_bias);
	
	vec4d_t query(1, vec3d_t(qkv[0].size(), vec2d_t(qkv[0][0].size(), vec1d_t(qkv[0][0][0].size() / 3))));
	vec4d_t key(1, vec3d_t(qkv[0].size(), vec2d_t(qkv[0][0].size(), vec1d_t(qkv[0][0][0].size() / 3))));
	vec4d_t value(1, vec3d_t(qkv[0].size(), vec2d_t(qkv[0][0].size(), vec1d_t(qkv[0][0][0].size() / 3))));
	for (int i = 0; i < qkv.size(); i++) {
		for (int j = 0; j < qkv[0].size(); j++) {
			for (int k = 0; k < qkv[0][0].size(); k++) {
				int temp = qkv[0][0][0].size() / 3;
				for (int t = 0; t < temp; t++) {
					query[i][j][k][t] = qkv[i][j][k][t];
				}
				for (int t = temp; t < temp * 2; t++) {
					key[i][j][k][t - temp] = qkv[i][j][k][t];
				}
				for (int t = temp * 2; t < temp * 3; t++) {
					value[i][j][k][t - temp * 2] = qkv[i][j][k][t];
				}
			}
		}
	}

	query = view(query, hidden_states.size(), hidden_states[0].size(), config.num_heads, head_dim);
	key = view(key, hidden_states.size(), hidden_states[0].size(), config.num_heads, head_dim);
	value = view(value, hidden_states.size(), hidden_states[0].size(), config.num_heads, head_dim);

	query = permute(query, 0, 2, 1, 3);
	key = permute(key, 0, 2, 1, 3);
	value = permute(value, 0, 2, 1, 3);

	if (config.is_training)
	{
		string name = weights_path + "_query";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = query;

		name = weights_path + "_key";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = key;

		name = weights_path + "_value";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = value;
	}
	vec5d_t present{ NULL };
	if (config.use_cache) {
		present.push_back(key);
		present.push_back(value);
	}

	vec4d_t attn_weights = MatMul(query, transpose(key, 2, 3));//[1 12 key_length 64]*[1 12 64 key_length]=[1 12 key_length key_length]
	if (config.scale_attn_weights) {
		for (auto& i : attn_weights) {
			for (auto& j : i) {
				for (auto& k : j) {
					for (auto& l : k) {
						l = l / sqrt(value[0][0][0].size());
					}
				}
			}
		}
	}

	uint32_t query_length = query[0][0].size();
	uint32_t key_length = key[0][0].size();

	for (int m = 0; m < attn_weights.size(); m++) {
		for (int t = 0; t < attn_weights[0].size(); t++) {
			for (int i = 0; i < key_length; i++) {
				for (int j = i + 1; j < key_length; j++) {
					attn_weights[m][t][i][j] = -10000;
				}
			}
		}
	}

	//WriteOutput(attn_weights, PRINT + weights_path + "_attn_weights.txt");
	attn_weights = softmax(attn_weights);
	if (config.is_training) {
		string name = weights_path + "_attn_weights_after_softmax";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = attn_weights;
	}

	vec4d_t attn_output = MatMul(attn_weights, value);//[1 12 key_length key_length]*[1 12 key_length 64] = [1 12 key_length 64]

	attn_output = permute(attn_output, 0, 2, 1, 3);//[1 key_length 12 64]
	attn_output = view(attn_output, 1, attn_output.size(), attn_output[0].size(), config.num_heads * head_dim);

	if (config.is_training) {
		string name = weights_path + "_c_proj_forward_in";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = attn_output;
	}

	vec2d_t c_proj_weight(config.n_embd, vec1d_t(embed_dim));
	vec1d_t c_proj_bias(embed_dim);
	if (config.is_training) {
		string name = weights_path + "_c_proj_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_proj_weight = std::any_cast<vec2d_t>(WeightMap[name]);

		name = weights_path + "_c_proj_bias";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_proj_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	else {
		LoadInput(c_proj_weight, weights_path + "_c_proj_weight.txt");
		LoadInput(c_proj_bias, weights_path + "_c_proj_bias.txt");
	}
	//attn_output = Conv1D(attn_output, embed_dim, PATH + weights_path + "_c_proj_weight.txt", PATH + weights_path + "_c_proj_bias.txt");
	attn_output = Conv1D(attn_output, vec4d_t(1, vec3d_t(1, c_proj_weight)), c_proj_bias);
	return make_tuple(attn_output, present);
}

vec3d_t GPT2MLP(GPT2Config config, vec3d_t hidden_states, string weights_path)
{
	uint32_t hidden_size = hidden_states[0][0].size();
	uint32_t embed_dim = hidden_size;
	map<string, vec4d_t(*)(vec4d_t)> ACT2FN{ {"gelu_new", NewGELUActivation} };
	vec2d_t c_fc_weight(config.n_embd, vec1d_t(config.fc));
	vec1d_t c_fc_bias(config.fc);
	if (config.is_training) {
		string name = weights_path + "_c_fc_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_fc_weight = std::any_cast<vec2d_t>(WeightMap[name]);

		name = weights_path + "_c_fc_bias";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_fc_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	else {
		LoadInput(c_fc_weight, weights_path + "_c_fc_weight.txt");
		LoadInput(c_fc_bias, weights_path + "_c_fc_bias.txt");
	}
	
	if (config.is_training) {
		string name = weights_path + "_c_fc";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, hidden_states);
	}
	//hidden_states = Conv1D(vec4d_t(1, hidden_states), 4 * hidden_size, PATH + weights_path + "_c_fc_weight.txt", PATH + weights_path + "_c_fc_bias.txt")[0];
	hidden_states = Conv1D(vec4d_t(1, hidden_states), vec4d_t(1, vec3d_t(1, c_fc_weight)), c_fc_bias)[0];

	if (config.is_training) {
		string name = weights_path + "_act";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, hidden_states);
	}
	hidden_states = ACT2FN[config.activation_function](vec4d_t(1, hidden_states))[0];
	if (config.is_training) {
		string name = weights_path + "_c_proj";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, hidden_states);
	}
	vec2d_t c_proj_weight(config.fc, vec1d_t(embed_dim));
	vec1d_t c_proj_bias(embed_dim);
	if (config.is_training) {
		string name = weights_path + "_c_proj_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_proj_weight = std::any_cast<vec2d_t>(WeightMap[name]);

		name = weights_path + "_c_proj_bias";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		c_proj_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	else {
		LoadInput(c_proj_weight, weights_path + "_c_proj_weight.txt");
		LoadInput(c_proj_bias, weights_path + "_c_proj_bias.txt");
	}
	//hidden_states = Conv1D(vec4d_t(1, hidden_states), embed_dim, PATH + weights_path + "_c_proj_weight.txt", PATH + weights_path + "_c_proj_bias.txt")[0];
	hidden_states = Conv1D(vec4d_t(1, hidden_states), vec4d_t(1, vec3d_t(1, c_proj_weight)), c_proj_bias)[0];

	return hidden_states;
}

std::tuple<vec4d_t, vec5d_t> GPT2Block(GPT2Config config, vec3d_t hidden_states, vec3d_t encoder_hidden_sates, std::string weights_path) {
	vec3d_t residual(hidden_states);
	vec1d_t ln_1_weights = vec1d_t(hidden_states[0][0].size(), 0.0);
	vec1d_t ln_1_bias = vec1d_t(hidden_states[0][0].size(), 0.0);
	if (config.is_training) {
		string name = weights_path + "_ln1_in";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, residual);
	}
	if (config.is_training) {
		string name = weights_path + "_ln_1_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_1_weights = std::any_cast<vec1d_t>(WeightMap[name]);

		name = weights_path + "_ln_1_bias";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_1_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	//hidden_states = LayerNorm(vec4d_t(1, residual), PATH + weights_path + "_ln_1")[0];
	hidden_states = LayerNorm(vec4d_t(1, residual), ln_1_weights, ln_1_bias)[0];
	
	std::tuple<vec4d_t, vec5d_t> attn_outputs = GPT2Attention(config, hidden_states, weights_path + "_attn");
	vec3d_t attn_output = std::get<0>(attn_outputs)[0];
	//WriteOutput(vec4d_t(1, attn_output), PRINT + weights_path + "_attn_out.txt");
	vec5d_t outputs = std::get<1>(attn_outputs);
	// residual connection
	hidden_states = AddVector(vec4d_t(1, attn_output), vec4d_t(1, residual))[0];
	
	if (!encoder_hidden_sates.empty()) {
		residual = hidden_states;
		hidden_states = LayerNorm(vec4d_t(1, residual), PATH + weights_path + "_ln_2")[0];
		std::tuple<vec4d_t, vec5d_t> cross_attn_outputs = GPT2Attention(config, hidden_states, weights_path + "_cross");

		attn_output = std::get<0>(cross_attn_outputs)[0];
		//residual connection
		residual.insert(residual.end(), attn_output.begin(), attn_output.end());
	}

	residual = hidden_states;
	//WriteOutput(hidden_states, PRINT + "cpp_mlp_in.txt");
	if (config.is_training) {
		string name = weights_path + "_ln2_in";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, hidden_states);
	}
	vec1d_t ln_2_weights(hidden_states[0][0].size());
	vec1d_t ln_2_bias(hidden_states[0][0].size());
	if (config.is_training) {
		string name = weights_path + "_ln_2_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_2_weights = std::any_cast<vec1d_t>(WeightMap[name]);

		name = weights_path + "_ln_2_bias";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_2_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	//hidden_states = LayerNorm(vec4d_t(1, hidden_states), PATH + weights_path + "_ln_2")[0];
	hidden_states = LayerNorm(vec4d_t(1, hidden_states), ln_2_weights, ln_2_bias)[0];
	
	vec3d_t feed_forwad = GPT2MLP(config, hidden_states, weights_path + "_mlp");
	//WriteOutput(vec4d_t(1, feed_forwad), PRINT + weights_path + "_mlp_out.txt");
	// residual connection
	hidden_states = AddVector(vec4d_t(1, residual), vec4d_t(1, feed_forwad))[0];

	if (config.use_cache)
		return std::make_tuple(vec4d_t(1, hidden_states), outputs);
	else {
		outputs.erase(outputs.begin());
		return std::make_tuple(vec4d_t(1, hidden_states), outputs);
	}
}

tuple<vec3d_t, vec6d_t> GPT2Model(GPT2Config config, vec2d_t input_ids, string weights_path)
{
	vec6d_t presents;
	uint32_t past_length = 0;//if past_key_values is None
	vec2d_t position_ids(1, vec1d_t(input_ids[0].size()));
	for (int i = 0; i < input_ids[0].size(); i++) {
		position_ids[0][i] = past_length + i;
	}
	int wte_weight_length = 50257;//=========================modify
	if (input_ids.size() > 1) {
		wte_weight_length = 50258;
	}
	vec2d_t wte_weight(wte_weight_length, vec1d_t(config.n_embd));
	if (config.is_training) {
		string name = weights_path + "_wte_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		wte_weight = std::any_cast<vec2d_t>(WeightMap[name]);
	}
	else {
		if (input_ids.size() > 1) LoadInput(wte_weight, PATH + "DoubleHead_" + weights_path + "_wte_weight.txt");
		else LoadInput(wte_weight, PATH + weights_path + "_wte_weight.txt");
	}
	vec3d_t inputs_embeds = Embedding(wte_weight, input_ids);
	//WriteOutput(vec4d_t(1, inputs_embeds), PRINT + weights_path + "_inputs_embeds.txt");
	vec2d_t wpe_weight(1024, vec1d_t(config.n_embd));
	if (config.is_training) {
		string name = weights_path + "_wpe_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		wpe_weight = std::any_cast<vec2d_t>(WeightMap[name]);
	}
	else LoadInput(wpe_weight, PATH + weights_path + "_wpe_weight.txt");
	vec3d_t position_embeds = Embedding(wpe_weight, position_ids);
	
	vec3d_t hidden_states = inputs_embeds;//===============modify
	for (int i = 0; i < hidden_states.size(); i++) {
		for (int j = 0; j < hidden_states[0].size(); j++) {
			for (int t = 0; t < hidden_states[0][0].size(); t++) {
				hidden_states[i][j][t] += position_embeds[0][j][t];
			}
		}
	}

	for (int i = 0; i < config.n_layer; i++) {
		cout << "now in block " << i << endl;
		vec3d_t encoder_hidden_states;
		tuple<vec4d_t, vec5d_t> block_output = GPT2Block(config, hidden_states, encoder_hidden_states, weights_path + "_h_" + to_string(i));
		hidden_states = get<0>(block_output)[0];
		WriteOutput(vec4d_t(1, hidden_states), PRINT + weights_path + "_h_" + to_string(i) + "out.txt");
		vec5d_t present = get<1>(block_output);
		presents.push_back(present);
	}
	if (config.is_training) {
		string name = weights_path + "_ln_f_in";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, hidden_states);
	}
	vec1d_t ln_f_weights(hidden_states[0][0].size());
	vec1d_t ln_f_bias(hidden_states[0][0].size());
	if (config.is_training) {
		string name = weights_path + "_ln_f_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_f_weights = std::any_cast<vec1d_t>(WeightMap[name]);

		name = weights_path + "_ln_f_bias";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_f_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}

	hidden_states = LayerNorm(vec4d_t(1, hidden_states), ln_f_weights, ln_f_bias)[0];
	return make_tuple(hidden_states, presents);
}

tuple<vec3d_t, float> GPT2LMHeadModel(GPT2Config config, vec2d_t input_ids, vec1d_t labels)
{
	tuple<vec3d_t, vec6d_t> Model_output = GPT2Model(config, input_ids, "transformer");
	vec3d_t hidden_states = get<0>(Model_output);//[2 7 768]
	vec2d_t lm_weights(50257, vec1d_t(768));
	if (config.is_training) {
		string name = "lm_head_weight";
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		lm_weights = std::any_cast<vec2d_t>(WeightMap[name]);
	}else LoadInput(lm_weights, PATH + "lm_head_weight.txt");

	if (config.is_training) {
		string name = "lm_forward_in";
		cout << COLOR::YELLOW << "SAVE DATR" << ' ' << name << COLOR::WHITE << endl;
		ForwardMap[name] = vec4d_t(1, hidden_states);
	}
	vec3d_t lm_logits = Linear(vec4d_t(1, hidden_states), lm_weights)[0];//[1 2 7 768] * [768 50258] = [1 2 7 50258] 
	WriteOutput(vec4d_t(1, lm_logits), PRINT + "forward_lm_logits.txt");
	float loss = -1;
	if (!labels.empty()) {
		//calc loss
		vec2d_t new_lm_logits(lm_logits[0].size() - 1, vec1d_t(lm_logits[0][0].size()));
		for (int i = 0; i < new_lm_logits.size(); i++) {
			for (int j = 0; j < new_lm_logits[0].size(); j++)
			{
				new_lm_logits[i][j] = lm_logits[0][i][j];
			}
		}
		vec1d_t new_labels(labels.size() - 1);
		for (int i = 0; i < new_labels.size(); i++)
		{
			new_labels[i] = labels[i + 1];
		}
		//labels[labels.size() - 1] = 0;
		loss = CrossEntropyLoss(new_lm_logits, new_labels);
	}

	return make_tuple(lm_logits, loss);
}

tuple<vec3d_t, vec4d_t> GPT2DoubleHeadsModel(GPT2Config config, vec3d_t input_ids, vec2d_t mc_token_ids) {
	tuple<vec3d_t, vec6d_t> Model_output = GPT2Model(config, input_ids[0], "transformer");
	vec3d_t hidden_states = get<0>(Model_output);
	WriteOutput(vec4d_t(1, hidden_states), PRINT + "DoubleHeads_model_output.txt");

	vec2d_t lm_weights(50258, vec1d_t(768));
	LoadInput(lm_weights, PATH + "DoubleHeads_lm_head_weight.txt");
	vec3d_t lm_logits = Linear(vec4d_t(1, hidden_states), lm_weights)[0];
	WriteOutput(vec4d_t(1, lm_logits), PRINT + "DoubleHeads_lm_logits.txt");

	vec4d_t cls_index(mc_token_ids.size(), vec3d_t(mc_token_ids[0].size(), vec2d_t(1, vec1d_t(hidden_states[0][0].size()))));//1 2 1 768
	for (int i = 0; i < cls_index.size(); i++) {
		for (int j = 0; j < cls_index[0].size(); j++) {
			for (int m = 0; m < cls_index[0][0].size(); m++) {
				for (int n = 0; n < cls_index[0][0][0].size(); n++) {
					cls_index[i][j][m][n] = mc_token_ids[i][j];
				}
			}
		}
	}
	vec3d_t output(cls_index[0].size(), vec2d_t(cls_index[0][0].size(), vec1d_t(cls_index[0][0][0].size())));//2 1 768
	cout << output.size() << ' ' << output[0].size() << ' ' << output[0][0].size() << endl;
	for (int i = 0; i < output.size(); i++) {//2
		for (int j = 0; j < output[0].size(); j++) {//1--->7
			for (int t = 0; t < output[0][0].size(); t++) {//768
				output[i][j][t] = hidden_states[i][cls_index[0][i][j][t]][t];
			}
		}
	}

	vec2d_t summary_weight(1, vec1d_t(768));
	LoadInput(summary_weight, PATH + "DoubleHeads_summary_weight.txt");
	vec4d_t mc_logits = Linear(vec4d_t(1, output), summary_weight);//[5.4372 5.3301]
	WriteOutput(mc_logits, PRINT + "DoubleHeads_mc_logits.txt");

	return make_tuple(lm_logits, mc_logits);
}

void UpdateWeight(GPT2Config config, vec2d_t& weight, vec2d_t grad)
{
	if (!config.is_training) return;
	std::cout << COLOR::ORANGE << "Update weight " << weight.size() << ' ' << weight[0].size() << COLOR::WHITE << endl;
	for (int i = 0; i < weight.size(); i++) {
		for (int j = 0; j < weight[0].size(); j++) {
			weight[i][j] = weight[i][j] - config.lr * grad[i][j];
		}
	}
}

void UpdateWeight(GPT2Config config, vec1d_t& weight, vec1d_t grad)
{
	if (!config.is_training) return;
	std::cout << COLOR::ORANGE << "Update weight " << weight.size() << COLOR::WHITE << endl;
	for (int i = 0; i < weight.size(); i++) {
		weight[i] = weight[i] - config.lr * grad[i];
	}
}

void UpdateBias(GPT2Config config, vec1d_t& bias, vec1d_t grad)
{
	if (!config.is_training) return;
	std::cout << COLOR::ORANGE << "Update bias " << bias.size() << COLOR::WHITE << endl;
	for (int i = 0; i < bias.size(); i++){
		bias[i] = bias[i] - config.lr * bias[i];
	}
}

vec4d_t GPT2AttentionBackward(GPT2Config config, vec4d_t dx, std::string weights_path)
{
	uint32_t batch_size = dx[0].size();
	uint32_t key_length = dx[0][0].size();//13
	uint32_t hidden_size = dx[0][0][0].size();//768
	uint32_t embed_dim = hidden_size;
	uint32_t head_dim = embed_dim / config.num_heads;//64 = 768 / 12
	vec4d_t attn_c_proj_forward_in(1, vec3d_t(1, vec2d_t(13, vec1d_t(768))));
	vec1d_t attn_c_proj_bias_grad = Conv1DBackDb(dx);
	if (config.is_training) {
		string name = weights_path + "_c_proj_forward_in";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		attn_c_proj_forward_in = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(attn_c_proj_forward_in, FORWARDPATH + weights_path + "_c_proj_forward_in.txt");
	vec4d_t attn_c_proj_weight_grad = Conv1DBackDw(attn_c_proj_forward_in, dx);
	vec2d_t attn_c_proj_weight(768, vec1d_t(768));
	if (config.is_training) {
		string name = weights_path + "_c_proj_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		attn_c_proj_weight = std::any_cast<vec2d_t>(WeightMap[name]);
	}
	else LoadInput(attn_c_proj_weight, PATH + weights_path + "_c_proj_weight.txt");
	dx = Conv1DBackDx(dx, attn_c_proj_weight);

	if (config.is_training) {
		string name = weights_path + "_c_proj_weight";
		UpdateWeight(config, attn_c_proj_weight, attn_c_proj_weight_grad[0][0]);
		WeightMap[name] = attn_c_proj_weight;
	}
	if (config.is_training) {
		vec1d_t attn_c_proj_bias(config.fc);
		string name = weights_path + "_c_proj_bias";
		attn_c_proj_bias = std::any_cast<vec1d_t>(WeightMap[name]);
		UpdateBias(config, attn_c_proj_bias, attn_c_proj_bias_grad);
		WeightMap[name] = attn_c_proj_bias;
	}
	dx = view(dx, 1, key_length, config.num_heads, head_dim);
	dx = permute(dx, 0, 2, 1, 3);
	
	vec4d_t attn_weights_after_softamx(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(key_length))));
	vec4d_t value(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(head_dim))));
	vec4d_t key(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(head_dim))));
	vec4d_t query(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(head_dim))));
	
	if (config.is_training)
	{
		string name = weights_path + "_attn_weights_after_softmax";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		attn_weights_after_softamx = std::any_cast<vec4d_t>(ForwardMap[name]);

		name = weights_path + "_value";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		value = std::any_cast<vec4d_t>(ForwardMap[name]);

		name = weights_path + "_key";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		key = std::any_cast<vec4d_t>(ForwardMap[name]);

		name = weights_path + "_query";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		query = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else {
		LoadInput(attn_weights_after_softamx, FORWARDPATH + weights_path + "_attn_weights_after_softmax.txt");
		LoadInput(value, FORWARDPATH + weights_path + "_value.txt");
		LoadInput(key, FORWARDPATH + weights_path + "_key.txt");
		LoadInput(query, FORWARDPATH + weights_path + "_query.txt");
	}
	vec4d_t dx_attn_weights = MatmulBackDx(value, dx);
	vec4d_t dx_from_value = MatmulBackDw(attn_weights_after_softamx, dx);
	vec4d_t dx_softmax = Softmax_back(attn_weights_after_softamx, dx_attn_weights);
	
	if (config.scale_attn_weights) {
		for (auto& i : dx_softmax) {
			for (auto& j : i) {
				for (auto& k : j) {
					for (auto& l : k) {
						l = l / sqrt(value[0][0][0].size());
					}
				}
			}
		}
	}
	vec4d_t dx_from_query = MatmulBackDx(transpose(key, 2, 3), dx_softmax);//[1 12 13 64]
	vec4d_t dx_from_key = MatmulBackDw(query, dx_softmax);
	dx_from_key = transpose(dx_from_key, 2, 3);//[1 12 13 64]
	dx_from_query = permute(dx_from_query, 0, 2, 1, 3);
	dx_from_key = permute(dx_from_key, 0, 2, 1, 3);
	dx_from_value = permute(dx_from_value, 0, 2, 1, 3);
	dx_from_query = view(dx_from_query, 1, batch_size, key_length, hidden_size);
	dx_from_key = view(dx_from_key, 1, batch_size, key_length, hidden_size);
	dx_from_value = view(dx_from_value, 1, batch_size, key_length, hidden_size);
	vec4d_t dx_qkv(dx_from_query.size(), vec3d_t(dx_from_query[0].size(), vec2d_t(dx_from_query[0][0].size(), vec1d_t(dx_from_query[0][0][0].size() * 3))));
	for (int i = 0; i < dx_qkv.size(); i++)
	{
		for (int j = 0; j < dx_qkv[0].size(); j++)
		{
			for (int k = 0; k < dx_qkv[0][0].size(); k++)
			{
				int temp = dx_qkv[0][0][0].size() / 3;
				for (int t = 0; t < temp; t++) {
					dx_qkv[i][j][k][t] = dx_from_query[i][j][k][t];
				}
				for (int t = temp; t < temp * 2; t++) {
					dx_qkv[i][j][k][t] = dx_from_key[i][j][k][t - temp];
				}
				for (int t = temp * 2; t < temp * 3; t++) {
					dx_qkv[i][j][k][t] = dx_from_value[i][j][k][t - temp * 2];
				}
			}
		}
	}
	vec4d_t attn_c_attn_forward_in(1, vec3d_t(1, vec2d_t(13, vec1d_t(768))));
	if (config.is_training)
	{
		string name = weights_path + "_c_attn_forward_in";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		attn_c_attn_forward_in = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(attn_c_attn_forward_in, FORWARDPATH + weights_path + "_c_attn_forward_in.txt");

	vec1d_t attn_c_attn_bias_grad = Conv1DBackDb(dx_qkv);
	if (config.is_training) {
		vec1d_t attn_c_attn_bias(config.fc);
		string name = weights_path + "_c_attn_bias";
		attn_c_attn_bias = std::any_cast<vec1d_t>(WeightMap[name]);
		UpdateBias(config, attn_c_attn_bias, attn_c_attn_bias_grad);
		WeightMap[name] = attn_c_attn_bias;
	}
	vec2d_t attn_c_attn_weight(768, vec1d_t(2304));
	if (config.is_training) {
		string name = weights_path + "_c_attn_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		attn_c_attn_weight = std::any_cast<vec2d_t>(WeightMap[name]);
	}
	vec4d_t dx_out = Conv1DBackDx(dx_qkv, attn_c_attn_weight);
	vec4d_t attn_c_attn_weight_grad = Conv1DBackDw(attn_c_attn_forward_in, dx_qkv);
	if (config.is_training) {
		string name = weights_path + "_c_attn_weight";
		UpdateWeight(config, attn_c_attn_weight, attn_c_attn_weight_grad[0][0]);
		WeightMap[name] = attn_c_attn_weight;
	}
	return dx_out;
}

vec3d_t GPT2MLPBackward(GPT2Config config, vec3d_t backward_input, std::string weights_path) {
	vec4d_t proj_dx;
	vec4d_t proj_dw;
	vec1d_t proj_db;
	vec4d_t proj_forward(1, vec3d_t(1, vec2d_t(13, vec1d_t(3072))));
	vec4d_t act_dx;
	vec4d_t act_forward(1, vec3d_t(1, vec2d_t(13, vec1d_t(3072))));
	vec4d_t fc_dx;
	vec4d_t fc_dw;
	vec1d_t fc_db;
	vec4d_t fc_forward(1, vec3d_t(1, vec2d_t(13, vec1d_t(768))));;
	vec2d_t proj_weight(config.fc, vec1d_t(config.n_embd));
	if (config.is_training) {
		string name = weights_path + "_c_proj_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		proj_weight = std::any_cast<vec2d_t>(WeightMap[name]);
	}
	else LoadInput(proj_weight, PATH + "_c_proj_weight.txt");
	proj_dx = Conv1DBackDx(vec4d_t(1, backward_input), proj_weight);

	if (config.is_training)
	{
		string name = weights_path + "_c_proj";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		proj_forward = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(proj_forward, FPATH + weights_path + "_c_proj.txt");
	proj_dw = Conv1DBackDw(proj_forward, vec4d_t(1, backward_input));
	proj_db = Conv1DBackDb(vec4d_t(1, backward_input));
	//updata	
	if (config.is_training) {
		string name = weights_path + "_c_proj_weight";
		UpdateWeight(config, proj_weight, proj_dw[0][0]);
		WeightMap[name] = proj_weight;
	}
	if (config.is_training) {
		vec1d_t proj_bias(config.fc);
		string name = weights_path + "_c_proj_bias";
		proj_bias = std::any_cast<vec1d_t>(WeightMap[name]);
		UpdateBias(config, proj_bias, proj_db);
		WeightMap[name] = proj_bias;
	}

	if (config.is_training)
	{
		string name = weights_path + "_act";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		act_forward = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(act_forward, FPATH + weights_path + "_act.txt");
	act_dx = NewGELUActivationBackward(act_forward, proj_dx);

	//WriteOutput(act_dx, PRINT + weights_path + "_act_dx.txt");
	vec2d_t fc_weight(config.n_embd, vec1d_t(config.fc));
	if (config.is_training) {
		string name = weights_path + "_c_fc_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		fc_weight = std::any_cast<vec2d_t>(WeightMap[name]);
	}
	else LoadInput(fc_weight, PATH + "_c_fc_weight.txt");
	proj_dx = Conv1DBackDx(act_dx, fc_weight);

	//WriteOutput(proj_dx, PRINT + weights_path + "_c_fc_dx.txt");

	if (config.is_training)
	{
		string name = weights_path + "_c_fc";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		fc_forward = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(fc_forward, FPATH + weights_path + "_c_fc.txt");
	proj_dw = Conv1DBackDw(fc_forward, act_dx);
	proj_db = Conv1DBackDb(act_dx);
	//updata

	if (config.is_training) {
		string name = weights_path + "_c_fc_weight";
		UpdateWeight(config, fc_weight, proj_dw[0][0]);
		WeightMap[name] = fc_weight;
	}
	if (config.is_training) {
		vec1d_t fc_bias(config.fc);
		string name = weights_path + "_c_fc_bias";
		fc_bias = std::any_cast<vec1d_t>(WeightMap[name]);
		UpdateBias(config, fc_bias, proj_db);
		WeightMap[name] = fc_bias;
	}

	return proj_dx[0];
}

vec4d_t GPT2BlockBackward(GPT2Config config, vec4d_t dx, std::string weights_path)
{
	vec4d_t mlp_dx = vec4d_t(1, GPT2MLPBackward(config, dx[0], weights_path + "_mlp"));

	//WriteOutput(ln2_dx, PRINT + weights_path + "_ln2_dx.txt");

	vec4d_t ln2_forward_in(1, vec3d_t(1, vec2d_t(13, vec1d_t(768))));
	if (config.is_training)
	{
		string name = weights_path + "_ln2_in";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln2_forward_in = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(ln2_forward_in, FORWARDPATH + weights_path + "_ln2_in.txt");
	//vec4d_t ln2_dx = GPT2LayerNormDxBackward(ln2_forward_in, mlp_dx, PATH + weights_path + "_ln_2");

	vec1d_t ln2_weight(config.n_embd);
	vec1d_t ln2_bias(config.n_embd);
	if (config.is_training) {
		string name = weights_path + "_ln_2_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln2_weight = std::any_cast<vec1d_t>(WeightMap[name]);

		name = weights_path + "_ln_2_bias";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln2_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	vec4d_t ln2_dx = GPT2LayerNormDxBackward(ln2_forward_in, mlp_dx, ln2_weight, ln2_bias);

	vec1d_t ln2_weight_grad = GPT2LayerNormDwBackward(ln2_forward_in, mlp_dx);
	if (config.is_training) {
		string name = weights_path + "_ln_2_weight";
		UpdateWeight(config, ln2_weight, ln2_weight_grad);
		WeightMap[name] = ln2_weight;
	}
	vec1d_t ln2_bias_grad = GPT2LayerNormDbBackward(mlp_dx);
	if (config.is_training) {
		string name = weights_path + "_ln_2_bias";
		UpdateBias(config, ln2_bias, ln2_bias_grad);
		WeightMap[name] = ln2_bias;
	}
	dx = AddVector(dx, ln2_dx);
	vec4d_t attn_dx = GPT2AttentionBackward(config, dx, weights_path + "_attn");
	
	//WriteOutput(ln2_dx, PRINT + weights_path + "_attn_dx.txt");

	vec4d_t ln1_forward_in(1, vec3d_t(1, vec2d_t(13, vec1d_t(768)))); 
	if (config.is_training)
	{
		string name = weights_path + "_ln1_in";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln1_forward_in = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(ln1_forward_in, FORWARDPATH + weights_path + "_ln1_in.txt");
	vec1d_t ln1_weight(config.n_embd);
	vec1d_t ln1_bias(config.n_embd);
	if (config.is_training) {
		string name = weights_path + "_ln_1_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln1_weight = std::any_cast<vec1d_t>(WeightMap[name]);

		name = weights_path + "_ln_1_bias";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln1_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	//vec4d_t ln1_dx = GPT2LayerNormDxBackward(ln1_forward_in, attn_dx, PATH + weights_path + "_ln_1");
	vec4d_t ln1_dx = GPT2LayerNormDxBackward(ln1_forward_in, attn_dx, ln1_weight, ln1_bias);

	//WriteOutput(ln1_dx, PRINT + weights_path + "_ln1_dx.txt");

	vec1d_t ln1_weight_grad = GPT2LayerNormDwBackward(ln1_forward_in, attn_dx);
	if (config.is_training) {
		string name = weights_path + "_ln_1_weight";
		UpdateWeight(config, ln1_weight, ln1_weight_grad);
		WeightMap[name] = ln1_weight;
	}
	vec1d_t ln1_bias_grad = GPT2LayerNormDbBackward(attn_dx);
	if (config.is_training) {
		string name = weights_path + "_ln_1_bias";
		UpdateBias(config, ln1_bias, ln1_bias_grad);
		WeightMap[name] = ln1_bias;
	}
	dx = AddVector(dx, ln1_dx);
	return dx;
}

vec4d_t GPT2ModelBackward(GPT2Config config, vec4d_t dx, std::string weights_path)
{
	vec4d_t ln_f_in(1, vec3d_t(1, vec2d_t(13, vec1d_t(768))));
	if (config.is_training)
	{
		string name = weights_path + "_ln_f_in";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_f_in = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	else LoadInput(ln_f_in, FORWARDPATH + weights_path + "_ln_f_in.txt");
	//vec4d_t ln_f_dx = GPT2LayerNormDxBackward(ln_f_in, dx, PATH + weights_path + "_ln_f");

	vec1d_t ln_f_weight(config.n_embd);
	vec1d_t ln_f_bias(config.n_embd);
	if (config.is_training) {
		string name = weights_path + "_ln_f_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_f_weight = std::any_cast<vec1d_t>(WeightMap[name]);

		name = weights_path + "_ln_f_bias";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		ln_f_bias = std::any_cast<vec1d_t>(WeightMap[name]);
	}
	vec4d_t ln_f_dx = GPT2LayerNormDxBackward(ln_f_in, dx, ln_f_weight, ln_f_bias);
	vec1d_t ln_f_weight_grad = GPT2LayerNormDwBackward(ln_f_in, dx);
	if (config.is_training) {
		string name = weights_path + "_ln_f_weight";
		UpdateWeight(config, ln_f_weight, ln_f_weight_grad);
		WeightMap[name] = ln_f_weight;
	}
	vec1d_t ln_f_bias_grad = GPT2LayerNormDbBackward(dx);
	if (config.is_training) {
		string name = weights_path + "_ln_f_bias";
		UpdateBias(config, ln_f_bias, ln_f_bias_grad);
		WeightMap[name] = ln_f_bias;
	}

	dx = ln_f_dx;//1.5%
	for (int i = config.n_layer - 1; i >= 0; i--)
	{
		dx = GPT2BlockBackward(config, dx, weights_path + "_h_" + std::to_string(i));
		WriteOutput(dx, PRINT + weights_path + "_h_" + std::to_string(i)+"_dx.txt");
		//return dx;
	}
	return dx;
}

void GPT2LMHeadModelBackward(GPT2Config config, vec2d_t lm_logits, vec1d_t labels)
{
	for (int j = 0; j < lm_logits[0].size(); j++)
	{
		lm_logits[lm_logits.size() - 1][j] = 0;
	}
	for (int i = 0; i < labels.size() - 1; i++)
	{
		labels[i] = labels[i + 1];
	}
	labels[labels.size() - 1] = 0;
	WriteOutput(vec4d_t(1, vec3d_t(1, lm_logits)), PRINT + "lm_logits.txt");
	vec2d_t softmax_lm_logits(lm_logits.size(), vec1d_t(lm_logits[0].size(), 0.0));//13 50257
	for (int i = 0; i < lm_logits.size() -1; i++) {
		softmax_lm_logits[i] = softmax(lm_logits[i]);
		softmax_lm_logits[i][labels[i]] -= 1.0;
		for (int j = 0; j < lm_logits[0].size(); j++) {
			softmax_lm_logits[i][j] /= (lm_logits.size() - 1);
		}
	}
	WriteOutput(vec4d_t(1, vec3d_t(1,softmax_lm_logits)), PRINT + "softmax_lm_logits.txt");
	
	vec2d_t lm_weights(config.vocab_size, vec1d_t(config.n_embd));
	if (config.is_training) {
		string name = "lm_head_weight";
		cout << COLOR::GREEN << "Read WeightMap" << ' ' << name << COLOR::WHITE << endl;
		if (WeightMap.find(name) == WeightMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		lm_weights = std::any_cast<vec2d_t>(WeightMap[name]);
	}
	else LoadInput(lm_weights, PATH + "lm_head_weight.txt");
	vec4d_t dx = LinearBackDx(lm_weights, vec4d_t(1, vec3d_t(1, softmax_lm_logits))/*softmax_lm_logits*/);//[13 50257] * [50257 768] = [13 768]
	WriteOutput(dx, PRINT + "Linear_Back_dx(softmax_double).txt");
	vec4d_t lm_forward_in(1, vec3d_t(1, vec2d_t(lm_logits.size(), vec1d_t(config.n_embd))));
	if (config.is_training)
	{
		string name = "lm_forward_in";
		if (ForwardMap.find(name) == ForwardMap.end()) cout << COLOR::RED << "!!!!" << name << COLOR::WHITE << endl;
		lm_forward_in = std::any_cast<vec4d_t>(ForwardMap[name]);
	}
	vec4d_t lm_weights_grad = LinearBackDw(lm_forward_in, vec4d_t(1, vec3d_t(1, softmax_lm_logits)));
	if (config.is_training) {
		string name = "lm_head_weight";
		UpdateWeight(config, lm_weights, lm_weights_grad[0][0]);
		WeightMap[name] = lm_weights;
	}

	dx = GPT2ModelBackward(config, dx, "transformer");
	WriteOutput(dx, PRINT + "lm_head_model_dx.txt");
}

int main()
{
	//training
	if (1)
	{
		map<std::string, uint32_t>::iterator iter = lengthMap.begin();
		while (iter != lengthMap.end()) {
			if (iter->second == 1769472) {
				vec2d_t weight(768, vec1d_t(2304));
				LoadInput(weight, PATH + iter->first + ".txt");
				WeightMap[iter->first] = weight;
			}
			else if (iter->second == 589824) {
				vec2d_t weight(768, vec1d_t(768));
				LoadInput(weight, PATH + iter->first + ".txt");
				WeightMap[iter->first] = weight;
			}
			else if (iter->second == 2359296) {
				if (iter->first == "transformer_h_0_mlp_c_fc_weight" ||
					iter->first == "transformer_h_10_mlp_c_fc_weight" ||
					iter->first == "transformer_h_11_mlp_c_fc_weight" ||
					iter->first == "transformer_h_1_mlp_c_fc_weight" ||
					iter->first == "transformer_h_2_mlp_c_fc_weight" ||
					iter->first == "transformer_h_3_mlp_c_fc_weight" ||
					iter->first == "transformer_h_4_mlp_c_fc_weight" ||
					iter->first == "transformer_h_5_mlp_c_fc_weight" ||
					iter->first == "transformer_h_6_mlp_c_fc_weight" ||
					iter->first == "transformer_h_7_mlp_c_fc_weight" ||
					iter->first == "transformer_h_8_mlp_c_fc_weight" ||
					iter->first == "transformer_h_9_mlp_c_fc_weight") {
					vec2d_t weight(768, vec1d_t(3072));
					LoadInput(weight, PATH + iter->first + ".txt");
					WeightMap[iter->first] = weight;
				}
				else {
					vec2d_t weight(3072, vec1d_t(768));
					LoadInput(weight, PATH + iter->first + ".txt");
					WeightMap[iter->first] = weight;
				}
			}
			else if (iter->second == 38597376) {
				vec2d_t weight(50257, vec1d_t(768));
				LoadInput(weight, PATH + iter->first + ".txt");
				WeightMap[iter->first] = weight;
			}
			else if (iter->second == 786432) {
				vec2d_t weight(1024, vec1d_t(768));
				LoadInput(weight, PATH + iter->first + ".txt");
				WeightMap[iter->first] = weight;
			}
			else {
				vec1d_t weight(iter->second);
				LoadInput(weight, PATH + iter->first + ".txt");
				WeightMap[iter->first] = weight;
			}
			iter++;
		}
		GPT2Config config;
		config.is_training = 1;
		vec1d_t labels = { 8241, 318, 7455, 8436, 38, 273, 5633, 7455, 8436, 38, 273, 318, 257 };
		vec2d_t input_ids(1, vec1d_t(13));
		LoadInput(input_ids, "C:\\Users\\baidiao\\Desktop\\instructgpt\\pytorch_out\\input_ids.txt");
		float loss;
		for (int i = 0; i < 5; i++) {
			auto forward_result = GPT2LMHeadModel(config, input_ids, labels);
			vec3d_t lm_logits = get<0>(forward_result);
			loss = get<1>(forward_result);
			std::cout << COLOR::GREEN << "Loss: " << loss << COLOR::WHITE <<std::endl;
			GPT2LMHeadModelBackward(config, lm_logits[0], labels);
		}
		
	}
}