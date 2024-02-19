#include"GPT2Backward.h"
using namespace std;

inline string PATH = "C:\\Users\\baidiao\\Desktop\\instructgpt\\weight\\";
inline string PRINT = "C:\\Users\\baidiao\\Desktop\\instructgpt\\c++_out\\";
inline string FORWARDPATH = "C:\\Users\\baidiao\\Desktop\\instructgpt\\MiddleValue\\";
inline string FPATH = "C:\\Users\\cltech\\Desktop\\GPT2\\forward\\";

vec4d_t GPT2AttentionBackward(GPT2Config config, vec4d_t dx, std::string weights_path)
{
	// 存储dx的维度信息
	uint32_t batch_size = dx[0].size();
	uint32_t key_length = dx[0][0].size();//13
	uint32_t hidden_size = dx[0][0][0].size();//768
	uint32_t embed_dim = hidden_size;
	uint32_t head_dim = embed_dim / config.num_heads;//64 = 768 / 12
	// 创建一个四维张量对象，并从文件中加载值。
	vec4d_t attn_c_proj_forward_in(1, vec3d_t(1, vec2d_t(13, vec1d_t(768))));

	// 使用dx计算attn_c_proj_bias_grad和attn_c_proj_weight_grad。最后，使用dx计算新的dx值
	vec1d_t attn_c_proj_bias_grad = Conv1DBackDb(dx);
	LoadInput(attn_c_proj_forward_in, FORWARDPATH + weights_path + "_c_proj_forward_in.txt");
	vec4d_t attn_c_proj_weight_grad = Conv1DBackDw(attn_c_proj_forward_in, dx);
	dx = Conv1DBackDx(dx, 768, PATH + weights_path + "_c_proj_weight.txt");

	// 对dx进行形状变换和维度交换操作
	dx = view(dx, 1, key_length, config.num_heads, head_dim);
	dx = permute(dx, 0, 2, 1, 3);
	
	// 创建了一些四维张量对象，并从文件中加载之。然后使用value和dx计算dx_attn_weights
	vec4d_t attn_weights_after_softamx(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(key_length))));
	LoadInput(attn_weights_after_softamx, FORWARDPATH + weights_path + "_attn_weights_after_softmax.txt");
	vec4d_t value(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(head_dim))));
	LoadInput(value, FORWARDPATH + weights_path + "_value.txt");
	vec4d_t key(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(head_dim))));
	LoadInput(key, FORWARDPATH + weights_path + "_key.txt");
	vec4d_t query(1, vec3d_t(config.num_heads, vec2d_t(key_length, vec1d_t(head_dim))));
	LoadInput(query, FORWARDPATH + weights_path + "_query.txt");
	vec4d_t dx_attn_weights = MatmulBackDx(value, dx);
	
	// 使用attn_weights_after_softmax和dx计算dx_from_value和dx_softmax
	vec4d_t dx_from_value = MatmulBackDw(attn_weights_after_softamx, dx);
	vec4d_t dx_softmax = Softmax_back(attn_weights_after_softamx, dx_attn_weights);
	
	// 如果config中的scale_attn_weights为真，则对dx_softmax中的值进行缩放操作
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

	// 使用key和dx_softmax计算query_grad，使用query和dx_softmax计算key_grad，使用dx_softmax和query计算value_grad		(dx_softmax中dx是梯度)
	vec4d_t dx_from_query = MatmulBackDx(transpose(key, 2, 3), dx_softmax);//[1 12 13 64]
	vec4d_t dx_from_key = MatmulBackDw(query, dx_softmax);
	
	// 对dx_from_key进行转置操作，将维度2和维度3进行交换
	dx_from_key = transpose(dx_from_key, 2, 3);//[1 12 13 64]
	// 这行代码对dx_from_query进行位置置换操作，将维度0、2、1、3重新排列
	dx_from_query = permute(dx_from_query, 0, 2, 1, 3);
	// 对dx_from_key、dx_from_query进行维度置换操作，将维度0、2、1、3重新排列
	dx_from_key = permute(dx_from_key, 0, 2, 1, 3);
	dx_from_value = permute(dx_from_value, 0, 2, 1, 3);

	// 对dx_from_query、dx_from_key、dx_from_value进行视图变换，重新调整维度大小为1、batch_size、key_length和hidden_size
	dx_from_query = view(dx_from_query, 1, batch_size, key_length, hidden_size);
	dx_from_key = view(dx_from_key, 1, batch_size, key_length, hidden_size);
	dx_from_value = view(dx_from_value, 1, batch_size, key_length, hidden_size);
	
	// 维度与dx_from_query相同，并且最内层维度的大小为dx_from_query最内层维度大小的三倍
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
	LoadInput(attn_c_attn_forward_in, FORWARDPATH + weights_path + "_c_attn_forward_in.txt");

	// 调用Conv1DBackDb函数计算dx_qkv的偏置梯度，
	vec1d_t attn_c_attn_bias_grad = Conv1DBackDb(dx_qkv);
	// 调用Conv1DBackDx函数计算dx_qkv的输入梯度
	vec4d_t dx_out = Conv1DBackDx(dx_qkv, 768, PATH + weights_path + "_c_attn_weight.txt");
	// 调用Conv1DBackDw函数计算attn_c_attn_forward_in和dx_qkv的权重梯度
	vec4d_t attn_c_attn_weight_grad = Conv1DBackDw(attn_c_attn_forward_in, dx_qkv);

	return dx_out;
}

vec3d_t GPT2MLPBackward(GPT2Config config, vec3d_t backward_input, std::string weights_path, std::map<std::string, std::any> forward) {
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

	proj_dx = Conv1DBackDx(vec4d_t(1, backward_input), config.fc, PATH + weights_path + "_c_proj_weight.txt");
	//act_forward = std::any_cast<vec4d_t>(forward.at(weights_path + "_c_proj"));
	LoadInput(proj_forward, FPATH + weights_path + "_c_proj.txt");
	proj_dw = Conv1DBackDw(proj_forward, vec4d_t(1, backward_input));
	proj_db = Conv1DBackDb(vec4d_t(1, backward_input));
	//updata	

	//act_forward = std::any_cast<vec4d_t>(forward.at(weights_path + "_act"));
	LoadInput(act_forward, FPATH + weights_path + "_act.txt");
	act_dx = NewGELUActivationBackward(act_forward, proj_dx);


	proj_dx = Conv1DBackDx(act_dx, config.n_embd, PATH + weights_path + "_c_fc_weight.txt");
	//act_forward = std::any_cast<vec4d_t>(forward.at(weights_path + "_c_fc"));
	LoadInput(fc_forward, FPATH + weights_path + "_c_fc.txt");
	proj_dw = Conv1DBackDw(fc_forward, act_dx);
	proj_db = Conv1DBackDb(act_dx);
	//updata

	return proj_dx[0];
}