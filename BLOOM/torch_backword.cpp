#include "torch.h"


vec4d_t Tanh(vec4d_t input_vec) {

	for (auto& x_1 : input_vec) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				for (auto& x_4 : x_3) {
					if (x_4 > 7.6) {
						x_4 = 1.0;
					}
					else if (x_4 < -7.6) {
						x_4 = -1.0;
					}
					else {
						x_4 = (exp(x_4) - exp(-x_4)) / (exp(x_4) + exp(-x_4));
					}
				}
			}
		}
	}

	return input_vec;
}

vec4d_t Cosh(vec4d_t input_vec) {
	for (auto& x_1 : input_vec) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				for (auto& x_4 : x_3) {
					double temp = exp(double(x_4)) + exp(double(-x_4));
					x_4 = temp / 2;
				}
			}
		}
	}

	return input_vec;
}

vec2d_t matmul(vec2d_t input_1, vec2d_t input_2) {
	uint32_t row_1 = input_1.size();
	uint32_t col_1 = input_1[0].size();
	uint32_t col_2 = input_2[0].size();
	vec2d_t output(row_1, vec1d_t(col_2));

	for (int i = 0; i < row_1; i++) {
		for (int j = 0; j < col_2; j++) {
			for (int k = 0; k < col_1; k++) {
				output[i][j] += input_1[i][k] * input_2[k][j];
			}
		}
	}

	return output;
}

vec4d_t MatMul(vec4d_t mat_1, vec4d_t mat_2)
{
	int B = mat_1.size();
	int features_in = mat_1[0][0][0].size();
	assert(features_in == mat_2[0][0].size());
	int features_out = mat_2[0][0][0].size();
	vec4d_t output(B, vec3d_t(mat_1[0].size(),
		vec2d_t(mat_1[0][0].size(), vec1d_t(features_out))));

	for (int i = 0; i < B; i++)
	{
		for (int j = 0; j < mat_1[0].size(); j++)
		{
			for (int k = 0; k < mat_1[0][0].size(); k++)
			{
				for (int l = 0; l < features_out; l++)
				{
					for (int m = 0; m < features_in; m++)
					{
						output[i][j][k][l] += mat_1[i][j][k][m] * mat_2[i][j][m][l];
					}
				}
			}
		}
	}
	return output;
}

vec4d_t MatMulForWeight(vec4d_t mat_1, vec4d_t mat_2)
{
	int B = mat_1.size();
	int features_in = mat_1[0][0][0].size();
	assert(features_in == mat_2[0][0].size());
	int features_out = mat_2[0][0][0].size();
	vec4d_t output(B, vec3d_t(mat_1[0].size(),
		vec2d_t(mat_1[0][0].size(), vec1d_t(features_out))));

	for (int i = 0; i < B; i++)
	{
		for (int j = 0; j < mat_1[0].size(); j++)
		{
			for (int k = 0; k < mat_1[0][0].size(); k++)
			{
				for (int l = 0; l < features_out; l++)
				{
					for (int m = 0; m < features_in; m++)
					{
						output[i][j][k][l] += mat_1[i][j][k][m] * mat_2[0][0][m][l];
					}
				}
			}
		}
	}
	return output;
}

vec4d_t NewGELUActivation(vec4d_t input_vec) {
	vec4d_t temp_1(input_vec);
	float temp_val = sqrt(2.0 / PI);

	for (auto& x_1 : temp_1) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				for (auto& x_4 : x_3) {
					x_4 = temp_val * (x_4 + 0.044715 * pow(x_4, 3));
				}
			}
		}
	}

	temp_1 = Tanh(temp_1);

	for (uint32_t dims_4 = 0; dims_4 < input_vec.size(); dims_4++) {
		for (uint32_t dims_3 = 0; dims_3 < input_vec[0].size(); dims_3++) {
			for (uint32_t dims_2 = 0; dims_2 < input_vec[0][0].size(); dims_2++) {
				for (uint32_t dims_1 = 0; dims_1 < input_vec[0][0][0].size(); dims_1++) {
					float x = input_vec[dims_4][dims_3][dims_2][dims_1];
					input_vec[dims_4][dims_3][dims_2][dims_1] = 0.5 * x * (1.0 + temp_1[dims_4][dims_3][dims_2][dims_1]);
				}
			}
		}
	}

	return input_vec;
}

vec4d_t Addmm(vec4d_t mat_1, vec4d_t mat_2, vec1d_t bias, uint32_t beta, uint32_t alpha) {
	vec4d_t output(mat_1.size(), vec3d_t(mat_1[0].size(), vec2d_t(mat_1[0][0].size(), vec1d_t(mat_2[0][0][0].size(), 0))));
	output = MatMulForWeight(mat_1, mat_2);
	for (uint32_t i = 0; i < output.size(); i++) {
		for (uint32_t j = 0; j < output[0].size(); j++) {
			for (uint32_t x = 0; x < output[0][0].size(); x++) {
				for (uint32_t y = 0; y < output[0][0][0].size(); y++) {
					output[i][j][x][y] = beta * bias[y] + alpha * output[i][j][x][y];
				}
			}
		}
	}
	return output;
}

vec4d_t Conv1D(vec4d_t x, uint32_t nf, std::string weights_path, std::string bias_path) {
	//vec4d_t weights(x.size(), vec3d_t(x[0].size(), vec2d_t(x[0][0][0].size(), vec1d_t(nf, 0.0))));
	vec4d_t weights(1, vec3d_t(1, vec2d_t(x[0][0][0].size(), vec1d_t(nf, 0.0))));
	vec1d_t bias(nf, 0.0);
	vec4d_t output;

	std::cout << x[0][0][0].size() << " " << nf << std::endl;
	LoadInput(weights, weights_path);
	LoadInput(bias, bias_path);

	output = Addmm(x, weights, bias);
	return output;
}

vec1d_t softmax(vec1d_t& input) {
	//double sum = 0.0;
	float sum = 0.0;
	int n = input.size();
	vec1d_t output(n);
	for (int i = 0; i < n; i++) {
		sum += exp(input[i]);
	}
	for (int i = 0; i < n; i++) {
		output[i] = exp(input[i]) / sum;
	}
	return output;
}

vec4d_t softmax(vec4d_t& input) {
	int dim0 = input.size();
	int dim1 = input[0].size();
	int dim2 = input[0][0].size();
	int dim3 = input[0][0][0].size();
	vec4d_t output;
	for (int i = 0; i < dim0; i++) {
		vec3d_t output_3d;
		for (int j = 0; j < dim1; j++) {
			vec2d_t output_2d;
			for (int k = 0; k < dim2; k++) {
				output_2d.push_back(softmax(input[i][j][k]));
			}
			output_3d.push_back(output_2d);
		}
		output.push_back(output_3d);
	}
	return output;
}

vec4d_t transpose(vec4d_t hidden_states, uint32_t dim_1, uint32_t dim_2) {
	uint32_t dim[4] = { hidden_states.size(), hidden_states[0].size(), hidden_states[0][0].size(), hidden_states[0][0][0].size() };
	uint32_t temp[4];
	std::swap(dim[dim_1], dim[dim_2]);
	vec4d_t output(dim[0], vec3d_t(dim[1], vec2d_t(dim[2], vec1d_t(dim[3]))));
	uint32_t output_B = 0, output_C = 0, output_H = 0, output_W = 0;


	for (output_B = 0, temp[0] = 0; output_B < dim[0]; output_B++, temp[0]++) {
		for (output_C = 0, temp[1] = 0; output_C < dim[1]; output_C++, temp[1]++) {
			for (output_H = 0, temp[2] = 0; output_H < dim[2]; output_H++, temp[2]++) {
				for (output_W = 0, temp[3] = 0; output_W < dim[3]; output_W++, temp[3]++) {
					std::swap(temp[dim_1], temp[dim_2]);
					output[output_B][output_C][output_H][output_W] = hidden_states[temp[0]][temp[1]][temp[2]][temp[3]];
					std::swap(temp[dim_1], temp[dim_2]);
				}
			}
		}
	}

	return output;
}

vec4d_t permute(vec4d_t hidden_states, const uint32_t dim_0, const uint32_t dim_1, const  uint32_t dim_2, const uint32_t dim_3) {
	uint32_t dim[4] = { hidden_states.size(), hidden_states[0].size(), hidden_states[0][0].size(), hidden_states[0][0][0].size() };
	uint32_t temp[4];
	vec4d_t output(dim[dim_0], vec3d_t(dim[dim_1], vec2d_t(dim[dim_2], vec1d_t(dim[dim_3]))));
	uint32_t output_B = 0, output_C = 0, output_H = 0, output_W = 0;

	for (output_B = 0, temp[dim_0] = 0; output_B < dim[dim_0]; output_B++, temp[dim_0]++) {
		for (output_C = 0, temp[dim_1] = 0; output_C < dim[dim_1]; output_C++, temp[dim_1]++) {
			for (output_H = 0, temp[dim_2] = 0; output_H < dim[dim_2]; output_H++, temp[dim_2]++) {
				for (output_W = 0, temp[dim_3] = 0; output_W < dim[dim_3]; output_W++, temp[dim_3]++) {
					output[output_B][output_C][output_H][output_W] = hidden_states[temp[0]][temp[1]][temp[2]][temp[3]];
				}
			}
		}
	}

	return output;
}

vec4d_t view(vec4d_t hidden_states, uint32_t dim_0, uint32_t dim_1, uint32_t dim_2, uint32_t dim_3) {
	vec4d_t output(dim_0, vec3d_t(dim_1, vec2d_t(dim_2, vec1d_t(dim_3))));
	uint32_t output_B = 0, output_C = 0, output_H = 0, output_W = 0;

	for (uint32_t i = 0; i < hidden_states.size(); i++) {
		for (uint32_t j = 0; j < hidden_states[0].size(); j++) {
			for (uint32_t l = 0; l < hidden_states[0][0].size(); l++) {
				for (uint32_t m = 0; m < hidden_states[0][0][0].size(); m++) {
					output[output_B][output_C][output_H][output_W] = hidden_states[i][j][l][m];

					output_W++;
					if (output_W == dim_3) {
						output_W = 0;
						output_H++;
						if (output_H == dim_2) {
							output_H = 0;
							output_C++;
							if (output_C == dim_1) {
								output_C = 0;
								output_B++;
							}
						}
					}
				}
			}
		}
	}

	return output;
}

vec2d_t Embedding(vec2d_t Embedding, vec1d_t indices)
{
	int n = indices.size();
	vec2d_t output;
	for (int i = 0; i < n; i++) {
		int idx = indices[i];
		output.push_back(Embedding[idx]);
	}
	return output;
}

vec3d_t Embedding(vec2d_t Embedding, vec2d_t indices) {
	vec3d_t output(indices.size(), vec2d_t(indices[0].size()));

	for (uint32_t i = 0; i < indices.size(); i++) {
		for (uint32_t j = 0; j < indices[0].size(); j++) {
			output[i][j] = Embedding[int(indices[i][j])];
		}
	}

	return output;
}

vec4d_t LayerNorm(vec4d_t&& input_vec, std::string path, float eps) {
	vec4d_t output_vec(input_vec.size(), vec3d_t(input_vec[0].size(), vec2d_t(input_vec[0][0].size(), vec1d_t(input_vec[0][0][0].size()))));
	uint32_t B = input_vec.size();
	uint32_t C = input_vec[0].size();
	uint32_t H = input_vec[0][0].size();
	uint32_t W = input_vec[0][0][0].size();
	float mean;
	float var;
	vec1d_t ln_weights = vec1d_t(input_vec[0][0][0].size(), 0.0);
	vec1d_t ln_bias = vec1d_t(input_vec[0][0][0].size(), 0.0);
	std::cout << input_vec[0][0][0].size() << std::endl;
	LoadInput(ln_weights, path + "_weight.txt");
	LoadInput(ln_bias, path + "_bias.txt");

	for (uint32_t batch = 0; batch < B; batch++) {
		for (uint32_t c = 0; c < C; c++) {
			for (uint32_t h = 0; h < H; h++) {
				mean = 0;
				var = 0;
				for (uint32_t w = 0; w < W; w++) {
					mean += input_vec[batch][c][h][w];
				}
				mean = mean / W;

				for (uint32_t w = 0; w < W; w++) {
					var += pow(input_vec[batch][c][h][w] - mean, 2);
				}
				var = var / W;

				for (uint32_t w = 0; w < W; w++) {
					output_vec[batch][c][h][w] = (input_vec[batch][c][h][w] - mean) / sqrt(var + eps) * ln_weights[w] + ln_bias[w];
				}
			}
		}
	}

	return output_vec;
}

vec4d_t LayerNorm(vec4d_t& input_vec, std::string path, float eps) {
	vec4d_t output_vec(input_vec.size(), vec3d_t(input_vec[0].size(), vec2d_t(input_vec[0][0].size(), vec1d_t(input_vec[0][0][0].size()))));
	uint32_t B = input_vec.size();
	uint32_t C = input_vec[0].size();
	uint32_t H = input_vec[0][0].size();
	uint32_t W = input_vec[0][0][0].size();
	float mean;
	float var;
	vec1d_t ln_weights = vec1d_t(input_vec[0][0][0].size(), 0.0);
	vec1d_t ln_bias = vec1d_t(input_vec[0][0][0].size(), 0.0);
	LoadInput(ln_weights, path + "weight.txt");
	LoadInput(ln_bias, path + "bias.txt");

	for (uint32_t batch = 0; batch < B; batch++) {
		for (uint32_t c = 0; c < C; c++) {
			for (uint32_t h = 0; h < H; h++) {
				mean = 0;
				var = 0;
				for (uint32_t w = 0; w < W; w++) {
					mean += input_vec[batch][c][h][w];
				}
				mean = mean / W;

				for (uint32_t w = 0; w < W; w++) {
					var += pow(input_vec[batch][c][h][w] - mean, 2);
				}
				var = var / W;

				for (uint32_t w = 0; w < W; w++) {
					output_vec[batch][c][h][w] = (input_vec[batch][c][h][w] - mean) / sqrt(var + eps) * ln_weights[w] + ln_bias[w];
				}
			}
		}
	}

	return output_vec;
}

vec4d_t Linear(vec4d_t input, vec2d_t weight)
{
	int B = input.size();
	int features_in = input[0][0][0].size();
	assert(features_in == weight[0].size());
	int features_out = weight.size();
	vec4d_t output(B, vec3d_t(input[0].size(), vec2d_t(input[0][0].size(), vec1d_t(features_out))));

	for (int i = 0; i < B; i++)
	{
		for (int j = 0; j < input[0].size(); j++)
		{
			for (int k = 0; k < input[0][0].size(); k++)
			{
				for (int l = 0; l < features_out; l++)
				{
					for (int m = 0; m < features_in; m++)
					{
						output[i][j][k][l] += input[i][j][k][m] * weight[l][m];
					}
				}
			}
		}
	}
	return output;
}

vec4d_t NewGELUActivationBackward(vec4d_t input_vec, vec4d_t back_ward) {
	vec4d_t temp_1(input_vec);
	vec4d_t temp_2(input_vec);
	vec4d_t temp_3(input_vec);
	float sqrt_pi = sqrt(PI);

	for (auto& x_1 : temp_1) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				for (auto& x_4 : x_3) {
					x_4 = (sqrt(2) * (0.044715 * pow(x_4, 3) + x_4)) / sqrt_pi;
				}
			}
		}
	}

	temp_2 = Tanh(temp_1);
	temp_3 = Cosh(temp_1);

	for (auto& x_1 : temp_3) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				for (auto& x_4 : x_3) {
					x_4 = pow(1 / x_4, 2);
				}
			}
		}
	}

	for (uint32_t dim_1 = 0; dim_1 < temp_3.size(); dim_1++) {
		for (uint32_t dim_2 = 0; dim_2 < temp_3[0].size(); dim_2++) {
			for (uint32_t dim_3 = 0; dim_3 < temp_3[0][0].size(); dim_3++) {
				for (uint32_t dim_4 = 0; dim_4 < temp_3[0][0][0].size(); dim_4++) {
					temp_3[dim_1][dim_2][dim_3][dim_4] = (sqrt(2) * input_vec[dim_1][dim_2][dim_3][dim_4]
						* (0.134145 * pow(input_vec[dim_1][dim_2][dim_3][dim_4], 2) + 1)
						* temp_3[dim_1][dim_2][dim_3][dim_4])
						/ sqrt(PI) + 1;
				}
			}
		}
	}

	for (uint32_t dim_1 = 0; dim_1 < temp_3.size(); dim_1++) {
		for (uint32_t dim_2 = 0; dim_2 < temp_3[0].size(); dim_2++) {
			for (uint32_t dim_3 = 0; dim_3 < temp_3[0][0].size(); dim_3++) {
				for (uint32_t dim_4 = 0; dim_4 < temp_3[0][0][0].size(); dim_4++) {
					temp_3[dim_1][dim_2][dim_3][dim_4] = (temp_2[dim_1][dim_2][dim_3][dim_4]
						+ temp_3[dim_1][dim_2][dim_3][dim_4]) / 2;
				}
			}
		}
	}

	for (uint32_t dim_1 = 0; dim_1 < temp_3.size(); dim_1++) {
		for (uint32_t dim_2 = 0; dim_2 < temp_3[0].size(); dim_2++) {
			for (uint32_t dim_3 = 0; dim_3 < temp_3[0][0].size(); dim_3++) {
				for (uint32_t dim_4 = 0; dim_4 < temp_3[0][0][0].size(); dim_4++) {
					temp_3[dim_1][dim_2][dim_3][dim_4] = temp_3[dim_1][dim_2][dim_3][dim_4] * back_ward[dim_1][dim_2][dim_3][dim_4];
				}
			}
		}
	}

	return temp_3;
}

vec4d_t GPT2LayerNormDxBackward(vec4d_t forward_input, vec4d_t backward_input, std::string weights_path, float eps) {
	uint32_t dim_1 = forward_input.size();
	uint32_t dim_2 = forward_input[0].size();
	uint32_t dim_3 = forward_input[0][0].size();
	uint32_t dim_4 = forward_input[0][0][0].size();
	float mean;
	float variance;
	float temp_1;
	float temp_2;
	float temp_3;
	float temp_4;
	float _x;
	vec4d_t output_vec(dim_1, vec3d_t(dim_2, vec2d_t(dim_3, vec1d_t(dim_4, 0.0))));
	vec1d_t weights(dim_4);
	vec1d_t bias(dim_4);

	LoadInput(weights, weights_path + "_weight.txt");
	LoadInput(bias, weights_path + "_bias.txt");

	for (uint32_t i = 0; i < dim_1; i++) {
		for (uint32_t j = 0; j < dim_2; j++) {
			for (uint32_t batch = 0; batch < dim_3; batch++) {
				mean = 0;
				variance = 0;
				temp_3 = 0;
				temp_4 = 0;
				_x = 0;

				for (uint32_t ch = 0; ch < dim_4; ch++) {
					mean += forward_input[i][j][batch][ch];
				}

				mean = mean / dim_4;

				for (uint32_t ch = 0; ch < dim_4; ch++) {
					/*variance += pow((forward_input[i][j][batch][ch] - mean), 2);*/
					variance += (forward_input[i][j][batch][ch] - mean) * (forward_input[i][j][batch][ch] - mean);
				}

				variance = sqrt(variance / dim_4);

				for (uint32_t ch = 0; ch < dim_4; ch++) {
					_x = (forward_input[i][j][batch][ch] - mean) / sqrt(pow(variance, 2) + eps);

					temp_3 += backward_input[i][j][batch][ch] * weights[ch] * _x;

					temp_4 += backward_input[i][j][batch][ch] * weights[ch];
				}

				for (uint32_t ch = 0; ch < dim_4; ch++) {
					_x = (forward_input[i][j][batch][ch] - mean) / sqrt(pow(variance, 2) + eps);

					output_vec[i][j][batch][ch] = (backward_input[i][j][batch][ch] * weights[ch] - (temp_4 + _x * temp_3) / dim_4) / sqrt(pow(variance, 2) + eps);
				}
			}
		}
	}

	return output_vec;
}

vec1d_t GPT2LayerNormDwBackward(vec4d_t forward_input, vec4d_t backward_input, float eps) {
	uint32_t dim_1 = forward_input.size();
	uint32_t dim_2 = forward_input[0].size();
	uint32_t dim_3 = forward_input[0][0].size();
	uint32_t dim_4 = forward_input[0][0][0].size();
	float mean;
	float variance;
	float temp_1;
	float temp_2;
	float temp_3;
	float temp_4;
	float _x;
	vec1d_t output_vec(dim_4, 0.0);

	for (uint32_t i = 0; i < dim_1; i++) {
		for (uint32_t j = 0; j < dim_2; j++) {
			for (uint32_t batch = 0; batch < dim_3; batch++) {
				mean = 0;
				variance = 0;
				temp_3 = 0;
				temp_4 = 0;
				_x = 0;

				for (uint32_t ch = 0; ch < dim_4; ch++) {
					mean += forward_input[i][j][batch][ch];
				}

				mean = mean / dim_4;

				for (uint32_t ch = 0; ch < dim_4; ch++) {
					variance += pow((forward_input[i][j][batch][ch] - mean), 2);
				}

				variance = sqrt(variance / dim_4);

				for (uint32_t ch = 0; ch < dim_4; ch++) {
					_x = (forward_input[i][j][batch][ch] - mean) / sqrt(pow(variance, 2) + eps);

					output_vec[ch] += backward_input[i][j][batch][ch] * _x;
				}
			}
		}
	}

	return output_vec;
}

vec1d_t GPT2LayerNormDbBackward(vec4d_t backward_input) {
	vec1d_t output(backward_input[0][0][0].size());

	for (uint32_t i = 0; i < backward_input.size(); i++) {
		for (uint32_t j = 0; j < backward_input[0].size(); j++) {
			for (uint32_t batch = 0; batch < backward_input[0][0].size(); batch++) {
				for (uint32_t ch = 0; ch < backward_input[0][0][0].size(); ch++) {
					output[ch] += backward_input[i][j][batch][ch];
				}
			}
		}
	}

	return output;
}

vec4d_t MatmulBackDw(vec4d_t forward_input, vec4d_t backward_input) {
	uint32_t B = forward_input.size();
	uint32_t dim1 = forward_input[0].size();
	uint32_t dim2 = forward_input[0][0].size();
	assert(dim2 == backward_input[0][0].size());
	uint32_t ch_in = forward_input[0][0][0].size();
	uint32_t ch_out = backward_input[0][0][0].size();
	vec4d_t X_T = transpose(forward_input, 2, 3);
	vec4d_t backward_output(B, vec3d_t(dim1, vec2d_t(ch_in, vec1d_t(ch_out))));

	for (uint32_t i = 0; i < B; i++)
		for (uint32_t j = 0; j < dim1; j++) {
			backward_output[i][j] = matmul(X_T[i][j], backward_input[i][j]);
		}
	return backward_output;
}

vec4d_t MatmulBackDx(vec4d_t forward_input, vec4d_t backward_input) {
	uint32_t B = forward_input.size();
	uint32_t dim1 = forward_input[0].size();
	uint32_t dim2 = forward_input[0][0][0].size();
	assert(dim2 == backward_input[0][0][0].size());
	uint32_t ch_in = forward_input[0][0][0].size();
	uint32_t ch_out = backward_input[0][0][0].size();
	vec4d_t X_T = transpose(forward_input, 2, 3);
	vec4d_t backward_output(B, vec3d_t(dim1, vec2d_t(ch_in, vec1d_t(ch_out))));
	/*cout << "======================" << endl;
	PrintDIM(backward_input);
	PrintDIM(X_T);*/
	for (uint32_t i = 0; i < B; i++)
		for (uint32_t j = 0; j < dim1; j++) {
			backward_output[i][j] = matmul(backward_input[i][j], X_T[i][j]);//[1 1 13 2304] * [2304 768]
		}
	return backward_output;
}

vec4d_t Conv1DBackDx(vec4d_t dx, uint32_t nf, std::string weights_path)
{
	vec4d_t weights(1, vec3d_t(1, vec2d_t(nf, vec1d_t(dx[0][0][0].size(), 0.0))));
	std::cout << weights[0][0].size() << ' ' << weights[0][0][0].size() << std::endl;
	LoadInput(weights, weights_path);
	dx = MatmulBackDx(weights, dx);
	return dx;
}

vec4d_t Conv1DBackDw(vec4d_t forward_input, vec4d_t backward_input)
{
	vec4d_t result = MatmulBackDw(forward_input, backward_input);
	return result;
}

vec1d_t Conv1DBackDb(vec4d_t dx)
{
	vec1d_t bias_grad(dx[0][0][0].size(),0.0);
	for (int i = 0; i < dx.size(); i++)
	{
		for (int j = 0; j < dx[0].size(); j++)
		{
			for (int k = 0; k < dx[0][0].size(); k++)
			{
				for (int t = 0; t < dx[0][0][0].size(); t++)
				{
					bias_grad[t] += dx[i][j][k][t];
				}
			}
		}
	}
	return bias_grad;
}

vec2d_t vec1d_mul(vec1d_t x) {
	int len = x.size();
	vec2d_t result(len, vec1d_t(len, 0.0));

	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			result[i][j] = x[i] * x[j];
		}
	}
	return result;
}

vec2d_t diag(vec1d_t x) {
	int len = x.size();
	vec2d_t result(len, vec1d_t(len, 0.0));

	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			if (i == j) {
				result[i][j] = x[i];
			}
		}
	}
	return result;
}

vec4d_t Softmax_back(vec4d_t& forward_output, vec4d_t& backward_input) {
	uint32_t dim0 = forward_output.size();
	uint32_t dim1 = forward_output[0].size();
	uint32_t dim2 = forward_output[0][0].size();
	uint32_t dim3 = forward_output[0][0][0].size();
	vec4d_t output(dim0, vec3d_t(dim1, vec2d_t(dim2, vec1d_t(dim3))));
	for (uint32_t i = 0; i < dim0; i++)
		for (uint32_t j = 0; j < dim1; j++)
			for (uint32_t k = 0; k < dim2; k++) {
				vec1d_t temp = forward_output[i][j][k];
				vec2d_t diag_y = diag(temp);
				vec2d_t y_TMuly = vec1d_mul(temp);

				vec2d_t dW_dS = SubVector(vec4d_t(1, vec3d_t(1, diag_y)), vec4d_t(1, vec3d_t(1, y_TMuly)))[0][0];
				vec2d_t res(1, backward_input[i][j][k]);
				output[i][j][k] = MatMul(vec4d_t(1, vec3d_t(1, res)), vec4d_t(1, vec3d_t(1, dW_dS)))[0][0][0];
			}
	return output;
}
