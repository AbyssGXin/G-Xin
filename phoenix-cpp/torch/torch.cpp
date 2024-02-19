#include "torch.h"

bool trans = false;

// inline unsigned short float32_to_bfloat16(float value)
// {
//     // 16 : 16
//     union
//     {
//         unsigned int u;
//         float f;
//     } tmp;
//     tmp.f = value;
//     return tmp.u >> 16;
// }
// // convert brain half to float
// inline float bfloat16_to_float32(unsigned short value)
// {
//     // 16 : 16
//     union
//     {
//         unsigned int u;
//         float f;
//     } tmp;
//     tmp.u = value << 16;
//     return tmp.f;
// }

typedef int                int32_t;
#define kBFloatSignificantForQNaN 0x00410000
#define kBFloatSignificantEvenRest 0x00007fff
#define kExponentMask 0x7f800000
#define kBFloatSignificant 0x007f0000
#define kBFloatSignificantInc 0x00010000
#define kBFloatSignificantEven 0x00008000
#define kNotExponentMask 0x807fffff
#define kSignificantMask 0x007fffff
#define kMaskOffHighestBit 0xefffffff
#define kExponentInc 0x00800000
#define kHiddenBitOfSignificant 0x00800000
#define kExponentForBFloatInt 0x43000000
#define kSignMask 0x80000000
#define kIAWidthMask 0x1ffff
#define kMostTwoByteMask 0xffff0000

typedef union
{
  struct {
    int16_t int16_lo;
    int16_t int16_hi;
  };
  int32_t int_t;
  float float_t;
} TypeTranslate;
enum RoundFormat
{
  ROUND = 1, //takes the higher 16bits, and round tie to even.
  TRUNCATE = 2, // trancate.
  LOWER_ROUND = 3 // takes the lower 16bits and round tie to even.
};

void RoundTieToEven(TypeTranslate& data)
{
  // see if the data is zero or small enough
  if ((data.int_t & kExponentMask) == 0) {
    data.int_t &= kSignMask;
  }
  // see if the data is inf or NaN
  else if ((data.int_t & kExponentMask) == kExponentMask) {
    // inf
    if ((data.int_t & kSignificantMask) == 0) {
      data.int_t &= kMostTwoByteMask;
    }
    // NaN
    else {
      data.int_t &= (~kSignificantMask);
      data.int_t |= kBFloatSignificantForQNaN;
    }
  }
  // normal
  else {
    // see if the significant need to be rounded
    if (data.int_t & kBFloatSignificantEven) {
      // see if the significant should be rounded tie to even or rounded up
      if ((data.int_t & kBFloatSignificantEvenRest) == 0) {
        // see if the significant need be rounded up to even
        if (data.int_t & kBFloatSignificantInc) {
          // see if the significant will overflow
          if ((data.int_t & kBFloatSignificant) == kBFloatSignificant) {
            data.int_t += kExponentInc;
            data.int_t &= (~kBFloatSignificant);
            data.int_t &= kMostTwoByteMask;
          } else {
            data.int_t += kBFloatSignificantInc;
            data.int_t &= kMostTwoByteMask;
          }
        } else {
          data.int_t &= kMostTwoByteMask;
        }
      } else {
        // see if the significant will be overflow
        if ((data.int_t & kBFloatSignificant) == kBFloatSignificant) {
          data.int_t += kExponentInc;
          data.int_t &= (~kBFloatSignificant);
          data.int_t &= kMostTwoByteMask;
        } else {
          data.int_t += kBFloatSignificantInc;
          data.int_t &= kMostTwoByteMask;
        }
      }
    }
    else {
      data.int_t &= kMostTwoByteMask;
    }
  }
}

TypeTranslate Float32ToFloat16(float input, RoundFormat format) {
  TypeTranslate result;
  TypeTranslate data;
  data.float_t = input;
  switch (format) {
  case ROUND:
    RoundTieToEven(data);
    data.int_t &= kMostTwoByteMask;
    break;
  case TRUNCATE:
    result.int_t = data.int_t & kMostTwoByteMask;
    return result;
  case LOWER_ROUND:
    TypeTranslate immediate;
    immediate.int_t = data.int_t & kMostTwoByteMask;
    data.float_t = data.float_t - immediate.float_t;
    RoundTieToEven(data);
    data.int_t &= kMostTwoByteMask;
    break;
  default:
    break;
  }
  result = data;
  return data;
}

void transition(vec4d_t& input)
{
  for (auto &x : input)
  {
    for (auto &x_1 : x)
    {
      for (auto &x_2: x_1)
      {
		for (auto &x_3: x_2)
		{
			x_3 = Float32ToFloat16(x_3, ROUND).float_t;
      	}
	  }
    }
  }
}


void transition(vec3d_t& input)
{
  for (auto &x : input)
  {
    for (auto &x_1 : x)
    {
      for (auto &x_2: x_1)
      {
        x_2 = Float32ToFloat16(x_2, ROUND).float_t;
      }
    }
  }
}

void transition(vec2d_t& input)
{
  for (auto &x : input)
  {
    for (auto &x_1 : x)
    {
      x_1 = Float32ToFloat16(x_1, ROUND).float_t;
    }
  }
}

void transition(vec1d_t& input)
{
  for (auto &x : input)
  {
      x = Float32ToFloat16(x, ROUND).float_t;
  }
}

vec4d_t Tanh(vec4d_t input_vec) {

	for (auto& x_1 : input_vec) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				for (auto& x_4 : x_3) {
					if (x_4 > 9.5) {
						x_4 = 1.0;
					}
					else if (x_4 < -9.5) {
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

vec3d_t Tanh(vec3d_t input_vec) {

	for (auto& x_1 : input_vec) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				if (x_3 > 9.6) {
					x_3 = 1.0;
				}
				else if (x_3 < -9.6) {
					x_3 = -1.0;
				}
				else {
					x_3 = (exp(x_3) - exp(-x_3)) / (exp(x_3) + exp(-x_3));
				}
			}
		}
	}

	return input_vec;
}

vec4d_t MatMul(vec4d_t mat_1, vec4d_t mat_2) {
	int B = mat_1.size();
	int features_in = mat_1[0][0][0].size();
	assert(features_in == mat_2[0][0].size());
	int features_out = mat_2[0][0][0].size();
	vec4d_t output(B, vec3d_t(mat_1[0].size(),
	vec2d_t(mat_1[0][0].size(), vec1d_t(features_out, 0.0))));

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

vec4d_t MatMulForWeight(vec4d_t mat_1, vec4d_t mat_2) {
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

	// WriteOutput(temp_1, "tamp_1.txt");
	temp_1 = Tanh(temp_1);
	// WriteOutput(temp_1, "tamp_2.txt");
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

vec3d_t BACK_NewGELUActivation(vec4d_t x, vec4d_t g) {
	vec3d_t temp_1 = x[0];
	float temp_val = sqrt(2.0 / PI);

	vec3d_t tanh_out = temp_1;

	for (auto& x_1 : tanh_out) {
		for (auto& x_2 : x_1) {
			for (auto& x_3 : x_2) {
				x_3 = temp_val * (x_3 + 0.044715 * pow(x_3, 3));
			}
		}
	}

	tanh_out = Tanh(tanh_out);

	uint32_t dimension_0_size = tanh_out.size();
	uint32_t dimension_1_size = tanh_out[0].size();
	uint32_t dimension_2_size = tanh_out[0][0].size();

	vec3d_t output_vec(dimension_0_size, vec2d_t(dimension_1_size, vec1d_t(dimension_2_size)));
	for (uint32_t i = 0; i < dimension_0_size; i++)
	{
		for (uint32_t j = 0; j < dimension_1_size; j++)
		{
			for (uint32_t k = 0; k < dimension_2_size; k++)
			{
				float x = tanh_out[i][j][k];
				tanh_out[i][j][k] = 0.5 * temp_1[i][j][k] * ((1 - pow(x, 2)) * (temp_val + 0.1070322243 * pow(temp_1[i][j][k], 2))) + 0.5 * (1 + x);
				output_vec[i][j][k] = tanh_out[i][j][k] * g[0][i][j][k];
			}
		}
	}


	return output_vec;
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
	//float sum = 0.0;
	float sum = 0.0;
	float max = -3.4028234663852886e+38;
	int n = input.size();
	vec1d_t output(n);
	for (int i = 0; i < n; i++)
	{
		// if (input[i] == -3.4028234663852886e+38) input[i] = 0;
		if (input[i] > max) max = input[i];
	}
	for (int i = 0; i < n; i++) {
		sum += exp(input[i] - max);
	}
	for (int i = 0; i < n; i++) {
		output[i] = exp(input[i] - max) / sum;
		// std::cout << "output[" << i << "]: " << output[i] << std::endl;
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

vec1d_t logsoftmax(vec1d_t input)
{
	float sum = 0.0;
	float max = -3.4028234663852886e+38;
	int n = input.size();
	vec1d_t output(n);
	for (int i = 0; i < n; i++) if (input[i] > max) max = input[i];
	for (int i = 0; i < n; i++) sum += exp(input[i] - max);
	for (int i = 0; i < n; i++) output[i] = input[i] - max - std::log(sum);
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

vec2d_t Embedding(vec2d_t Embedding, vec1d_t indices) {
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

vec4d_t LayerNorm(vec4d_t&& input_vec, vec1d_t ln_weights, vec1d_t ln_bias, float eps) {
	vec4d_t output_vec(input_vec.size(), vec3d_t(input_vec[0].size(), vec2d_t(input_vec[0][0].size(), vec1d_t(input_vec[0][0][0].size()))));
	uint32_t B = input_vec.size();
	uint32_t C = input_vec[0].size();
	uint32_t H = input_vec[0][0].size();
	uint32_t W = input_vec[0][0][0].size();
	float mean;
	float var;
	// vec1d_t ln_weights = vec1d_t(input_vec[0][0][0].size(), 0.0);
	// vec1d_t ln_bias = vec1d_t(input_vec[0][0][0].size(), 0.0);
	// std::cout << input_vec[0][0][0].size() << std::endl;
	// LoadInput(ln_weights, path + "weight.txt");
	// LoadInput(ln_bias, path + "bias.txt");

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

vec4d_t LayerNorm(vec4d_t& input_vec, vec1d_t weights, vec1d_t bias, float eps) {
	vec4d_t output_vec(input_vec.size(), vec3d_t(input_vec[0].size(), vec2d_t(input_vec[0][0].size(), vec1d_t(input_vec[0][0][0].size()))));
	uint32_t B = input_vec.size();
	uint32_t C = input_vec[0].size();
	uint32_t H = input_vec[0][0].size();
	uint32_t W = input_vec[0][0][0].size();
	float mean;
	float var;
	// vec1d_t ln_weights = vec1d_t(input_vec[0][0][0].size(), 0.0);
	// vec1d_t ln_bias = vec1d_t(input_vec[0][0][0].size(), 0.0);
	// LoadInput(ln_weights, path + "weight.txt");
	// LoadInput(ln_bias, path + "bias.txt");

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
					output_vec[batch][c][h][w] = (input_vec[batch][c][h][w] - mean) / sqrt(var + eps) * weights[w] + bias[w];
				}
			}
		}
	}

	return output_vec;
}

vec3d_t reshape(vec4d_t input, int dim1, int dim2, int dim3) {
	vec3d_t output(dim1, vec2d_t(dim2, vec1d_t(dim3)));

	// for (int b = 0; b < input.size(); b++) {
	// 	for (int h = 0; h < input[0].size(); h++) {
	// 		for (int s = 0; s < input[0][0].size(); s++) {
	// 			for (int d = 0; d < input[0][0][0].size(); d++) {
	// 				output[b][h][s * input[0][0][0].size() + d] = input[b][h][s][d];
	// 			}
	// 		}
	// 	}
	// }

	for (int d1 = 0; d1 < dim1; d1++) {
		for (int d2 = 0; d2 < dim2; d2++) {
			for (int d3 = 0; d3 < dim3; d3++) {
				int size = d1 * dim2 * dim3 + d2 * dim3 + d3;
				int index_1 = size / (input[0].size() * input[0][0].size() * input[0][0][0].size()),
						index_2 = (size % (input[0].size() * input[0][0].size() * input[0][0][0].size())) / (input[0][0].size() * input[0][0][0].size()),
						index_3 = (size % (input[0][0].size() * input[0][0][0].size())) / input[0][0][0].size(),
						index_4 = size % input[0][0][0].size();
				output[d1][d2][d3] = input[index_1][index_2][index_3][index_4];
			}
		}
	}

	return output;
}


vec4d_t Linear(vec4d_t input, vec2d_t weight, vec1d_t bias) {
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

					output[i][j][k][l] += bias[l];
				}
			}
		}
	}
	return output;
}

vec4d_t Linear_Nobias(vec4d_t input, vec2d_t weight) {
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


vec1d_t CrossEntropyLoss(vec2d_t logits, vec1d_t_i labels) {
	vec4d_t _logits(1, vec3d_t(1, logits));
	vec2d_t probabilities = softmax(_logits)[0][0];
	vec1d_t output(logits.size(), 0);

	for (int i = 0; i < logits.size(); i++) {
		for (int j = 0; j < logits[0].size(); j++) {
			if(i == labels[i])
				output[i] = std::log(probabilities[i][j]);
		}
	}

	
	return output;
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


vec4d_t MatmulBackDx(vec4d_t forward_input, vec4d_t backward_input) {
	uint32_t B = backward_input.size();
	uint32_t dim1 = backward_input[0].size();
	uint32_t dim2 = backward_input[0][0].size();
	uint32_t ch_in = backward_input[0][0][0].size();
	assert(ch_in == forward_input[0][0].size());
	uint32_t ch_out = forward_input[0][0][0].size();
	vec4d_t X_T = transpose(forward_input, 2, 3);
	vec4d_t backward_output(1, vec3d_t(1, vec2d_t(dim2, vec1d_t(ch_out))));
	for (int i = 0; i < B; i++)
	{
		for (int j = 0; j < dim1; j++)
		{
			for (int k = 0; k < dim2; k++)
			{
				for (int l = 0; l < ch_out; l++)
				{
					for (int m = 0; m < ch_in; m++)
					{
						backward_output[i][j][k][l] += backward_input[i][j][k][m] * X_T[i][j][l][m];
					}
				}
			}
		}
	}
	return backward_output;
}


// vec4d_t MatmulBackDw(vec4d_t forward_input, vec4d_t backward_input) {
// 	assert(forward_input.size() == backward_input.size());
// 	assert(forward_input[0].size() == backward_input[0].size());
// 	assert(forward_input[0][0].size() == backward_input[0][0].size());

// 	vec2d_t output(backward_input[0][0][0].size(), vec1d_t(forward_input[0][0][0].size(), 0));

// 	for (int h = 0; h < forward_input.size(); h++) {
// 		for (int w = 0; w < forward_input[0].size(); w++) {
// 			for (int row = 0; row < backward_input[0][0][0].size(); row++) {
// 				for (int col = 0; col < forward_input[0][0][0].size(); col++) {
// 					for (int r = 0; r < forward_input[0][0].size(); r++) {
// 						output[row][col] += backward_input[h][w][r][row] * forward_input[h][w][r][col];
// 					}
// 				}
// 			}
// 		}
// 	}

// 	return vec4d_t(1, vec3d_t(1, output));
// 	// return backward_output;
// }

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

vec4d_t MatmulBackDw_yinxun(vec4d_t forward_input, vec4d_t backward_input) {
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
	backward_output = transpose(backward_output, 2, 3);
	return backward_output;
}
vec4d_t LinearBackDx(vec4d_t dx, vec4d_t weights)
{
	// vec4d_t weights(1, vec3d_t(1, vec2d_t(nf, vec1d_t(dx[0][0][0].size(), 0.0))));
	// vec4d_t weights(1, vec3d_t(1, vec2d_t(dx[0][0][0].size(), vec1d_t(nf, 0.0))));
	// std::cout << weights[0][0].size() << ' ' << weights[0][0][0].size() << std::endl;
	// LoadInput(weights, weights_path);
	dx = MatmulBackDx(weights, dx);
	return dx;
}

vec4d_t LinearBackDw(vec4d_t forward_input, vec4d_t backward_input)
{
	assert(forward_input.size() == backward_input.size());
	assert(forward_input[0].size() == backward_input[0].size());
	assert(forward_input[0][0].size() == backward_input[0][0].size());

	vec2d_t output(backward_input[0][0][0].size(), vec1d_t(forward_input[0][0][0].size(), 0));

	for (int h = 0; h < forward_input.size(); h++) {
		for (int w = 0; w < forward_input[0].size(); w++) {
			for (int row = 0; row < backward_input[0][0][0].size(); row++) {
				for (int col = 0; col < forward_input[0][0][0].size(); col++) {
					for (int r = 0; r < forward_input[0][0].size(); r++) {
						output[row][col] += backward_input[h][w][r][row] * forward_input[h][w][r][col];
					}
				}
			}
		}
	}

	return vec4d_t(1, vec3d_t(1, output));
}

vec1d_t LinearBackDb(vec4d_t dx)
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


vec4d_t BLOOMLayerNormDxBackward(vec4d_t forward_input, vec4d_t backward_input, vec1d_t weights, vec1d_t bias, float eps) 
{
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

vec1d_t BLOOMLayerNormDwBackward(vec4d_t forward_input, vec4d_t backward_input, float eps) 
{
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

vec1d_t BLOOMLayerNormDbBackward(vec4d_t backward_input) 
{
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



vec3d_t dropout(vec3d_t input, float prob, bool training) 
{
    if (training)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (auto& x_1 : input)
        {
            for (auto& x_2 : x_1)
            {
                for (auto& x_3 : x_2)
                {
                    if (dis(gen) < prob)
                    {
                        x_3 = 0.0;
                    }
                    else
                    {
                        x_3 = x_3 / (1.0 - prob);
                    }
                }
            }
        }
    }
    return input;
 }

vec3d_t dropout_add(vec3d_t input, vec3d_t residual, float prob, bool training)
{
    vec3d_t out(input.size(), vec2d_t(input[0].size(), vec1d_t(input[0][0].size(), 0)));
    out = dropout(input, prob, training);
    
    for (int i = 0; i < input.size(); i++) 
    {
        for (int j = 0; j < input[0].size(); j++)
        {
            for (int k = 0; k < input[0][0].size(); k++)
            {
                out[i][j][k] += residual[i][j][k];
            }
        }
    }
    return out;
}

vec3d_t build_alibi(vec2d_t_i& input, int num_heads)
{
	int batch_size = input.size();
	int seq_length = input[0].size();

	int closest_power_of_2 = std::pow(2, std::floor(std::log2(num_heads)));
    float base = std::pow(2, -(std::pow(2, -(std::log2(closest_power_of_2) - 3))));
	
	std::cout << base << std::endl;
    vec1d_t powers(closest_power_of_2);
	std::iota(powers.begin(), powers.end(), 1);

	vec1d_t slopes(closest_power_of_2);
	for (int i = 0; i < closest_power_of_2; i++)
	{
		slopes[i] = std::pow(base, powers[i]);
	}
	
	if (closest_power_of_2 != num_heads)
	{
		float extra_base = std::pow(2, -(std::pow(2, -(std::log2(2 * closest_power_of_2) - 3))));
	    int num_remaining_heads = std::min(closest_power_of_2, num_heads - closest_power_of_2);
		vec1d_t extra_powers(num_remaining_heads);
		for (int i = 0; i < num_remaining_heads; i++)
		{
			extra_powers[i] = 2 * i + 1;
		}
		vec1d_t extra_slopes(num_remaining_heads);
		for (int i = 0; i < num_remaining_heads; i++)
	    {
		    extra_slopes[i] = std::pow(extra_base, extra_powers[i]);
	    }
		slopes.insert(slopes.end(), extra_slopes.begin(), extra_slopes.end());
	}
    vec2d_t_i arange_tensor(input);
	for (int i = 0; i < input.size(); i++) 
	{
		for (int j = 0; j < input[0].size(); j++)
		{
			if(j == 0) arange_tensor[i][j] = arange_tensor[i][j] - 1;
            else arange_tensor[i][j] = input[i][j] + arange_tensor[i][j - 1];
		}
	}
    for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input[0].size(); j++)
		{
			arange_tensor[i][j] *= input[i][j];
		}
	}
    
	vec3d_t alibi(num_heads * batch_size, vec2d_t(1, vec1d_t(seq_length)));
	for (int i = 0; i < num_heads * batch_size; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			for (int k = 0; k < seq_length; k++)
			{
				alibi[i][j][k] = arange_tensor[0][k] * slopes[i];
			}
		}
	}

    return alibi;
}

vec4d_t reshape(vec3d_t input, int dim1, int dim2, uint32_t dim3, uint32_t dim4) {
	// std::cout << "Debug reshape!" << std::endl;
	vec4d_t output(dim1, vec3d_t(dim2, vec2d_t(dim3, vec1d_t(dim4, 0))));
	int output_H = 0, output_C = 0, output_B = 0;

	for(int h = 0; h < dim1; h++) {
		for(int w = 0; w < dim2; w++) {
			for(int row = 0; row < dim3; row++) {
				for(int col = 0; col < dim4; col++) {
					output[h][w][row][col] = input[output_B][output_C][output_H];

					// std::cout << "output.index: [" << h << ", " << w << ", " << row << ", " << col << "]" << std::endl;
					// std::cout << "input.index: [" << output_B << ", " << output_C << ", " << output_H << "]\n";
					output_H++;
					if (output_H == input[0][0].size()) {
						output_H = 0;
						output_C++;
						if (output_C == input[0].size()) {
							output_C = 0;
							output_B++;
						}
					}
				}
			}
		}
	}

	return output;
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
				// if (trans) transition(temp);
				vec2d_t y_TMuly = vec1d_mul(temp);
				vec2d_t dW_dS = SubVector(vec4d_t(1, vec3d_t(1, diag_y)), vec4d_t(1, vec3d_t(1, y_TMuly)))[0][0];
				vec2d_t res(1, backward_input[i][j][k]);
				// if (trans)
				// {
				// 	transition(res);
				// 	transition(dW_dS);
				// }
				output[i][j][k] = MatMul(vec4d_t(1, vec3d_t(1, res)), vec4d_t(1, vec3d_t(1, dW_dS)))[0][0][0];
			}
	return output;
}

vec3d_t cat(vec3d_t input1, vec3d_t input2, int dim) {
	vec3d_t output = input1;

	switch (dim)
	{
	case 0:
		for (int i = 0; i < input2.size(); i++) {
			output.push_back(input2[i]);
		}
		break;
	case 1:
		for (int i = 0; i < input2.size(); i++) {
			for (int j = 0; j < input2[0].size(); j++) {
				output[i].push_back(input2[i][j]);
			}
		}
		break;
	case 2:
		for (int i = 0; i < input2.size(); i++) {
			for (int j = 0; j < input2[0].size(); j++) {
				for (int m = 0; m < input2[0][0].size(); m++) {
					output[i][j].push_back(input2[i][j][m]);
				}
			}
		}
		break;	
	default:
		std::cerr << "cat dim out range\n";
		break;
	}

	return output;
}

vec4d_t masked_fill(vec4d_t scores, vec4d_t_i mask, float value) {
	vec4d_t output(scores.size(), vec3d_t(scores[0].size(), vec2d_t(scores[0][0].size(), vec1d_t(scores[0][0][0].size()))));

	for (int d1 = 0; d1 < scores.size(); d1++) {
		for (int d2 = 0; d2 < scores[0].size(); d2++) {
			for (int d3 = 0; d3 < scores[0][0].size(); d3++) {
				for (int d4 = 0; d4 < scores[0][0][0].size(); d4++) {
					output[d1][d2][d3][d4] = mask[0][0][d3][d4] == 1 ? value : scores[d1][d2][d3][d4];
				}
			}
		}
	}

	return output;
}

vec3d_t bmm(vec3d_t input, vec3d_t mat2) {
	vec3d_t output(input.size(), vec2d_t(input[0].size(), vec1d_t(mat2[0][0].size(), 0.0)));

	for (int d1 = 0; d1 < input.size(); d1++) {
		for (int d2 = 0; d2 < input[0].size(); d2++) {
			for (int d3 = 0; d3 < mat2[0][0].size(); d3++) {
				for(int c = 0; c < input[0][0].size(); c++) {
					output[d1][d2][d3] += input[d1][d2][c] * mat2[d1][c][d3];
				}
			}
		}
	}

	return output;
}

vec3d_t baddbmm(vec3d_t input, vec3d_t batch1, vec3d_t batch2, float beta, float alpha) {
	vec3d_t output(batch1.size(), vec2d_t(batch1[0].size(), vec1d_t(batch2[0][0].size())));

	for (int d1 = 0; d1 < batch1.size(); d1++)
	{
		for (int d2 = 0; d2 < batch1[0].size(); d2++) {
			for (int d3 = 0; d3 < batch2[0][0].size(); d3++) {
				for (int c = 0; c < batch1[0][0].size(); c++) {
					output[d1][d2][d3] += batch1[d1][d2][c] * batch2[d1][c][d3];
				}
			}
		}
	}

	for (int d1 = 0; d1 < output.size(); d1++)
	{
		for (int d2 = 0; d2 < output[0].size(); d2++) {
			for (int d3 = 0; d3 < output[0][0].size(); d3++) {
				output[d1][d2][d3] = beta * input[d1][0][d3] + alpha * output[d1][d2][d3];
			}
		}
	}

	return output;
}


vec4d_t LinearDx(vec4d_t backward_input, vec2d_t weights) {
	assert(backward_input[0][0][0].size() == weights.size());

	vec4d_t output(backward_input.size(), vec3d_t(backward_input[0].size(), vec2d_t(backward_input[0][0].size(), vec1d_t(weights[0].size(), 0))));

	for (int h = 0; h < backward_input.size(); h++) {
		for (int w = 0; w < backward_input[0].size(); w++) {
			for (int row = 0; row < backward_input[0][0].size(); row++) {
				for (int col = 0; col < weights[0].size(); col++) {
					for (int r = 0; r < backward_input[0][0][0].size(); r++) {
						output[h][w][row][col] += backward_input[h][w][row][r] * weights[r][col];
					}
				}
			}
		}
	}

	return output;
}

vec2d_t LinearDw(vec4d_t forward_input, vec4d_t backward_input) {
	assert(forward_input.size() == backward_input.size());
	assert(forward_input[0].size() == backward_input[0].size());
	assert(forward_input[0][0].size() == backward_input[0][0].size());

	vec2d_t output(backward_input[0][0][0].size(), vec1d_t(forward_input[0][0][0].size(), 0));

	for (int h = 0; h < forward_input.size(); h++) {
		for (int w = 0; w < forward_input[0].size(); w++) {
			for (int row = 0; row < backward_input[0][0][0].size(); row++) {
				for (int col = 0; col < forward_input[0][0][0].size(); col++) {
					for (int r = 0; r < forward_input[0][0].size(); r++) {
						output[row][col] += backward_input[h][w][r][row] * forward_input[h][w][r][col];
					}
				}
			}
		}
	}

	return output;
}

vec1d_t LinearDb(vec4d_t backward_input) {
	vec1d_t output(backward_input[0][0][0].size(), 0);

	for (int h = 0; h < backward_input.size(); h++) {
		for (int w = 0; w < backward_input[0].size(); w++) {
			for (int row = 0; row < backward_input[0][0].size(); row++) {
				for (int col = 0; col < backward_input[0][0][0].size(); col++) {
					output[col] += backward_input[h][w][row][col];
				}
			}
		}
	}

	return output;
}

vec3d_t bmmDx(vec3d_t backward_input, vec3d_t mat2) {
	vec3d_t output(backward_input.size(), vec2d_t(backward_input[0].size(), vec1d_t(mat2[0].size(), 0)));
	vec4d_t temp = transpose(vec4d_t(1, mat2), 2, 3);
	output = MatMul(vec4d_t(1, backward_input), temp)[0];
	return output;
}


vec3d_t bmmDw(vec3d_t forward_input, vec3d_t backward_input) {
	vec3d_t output(forward_input.size(), vec2d_t(forward_input[0][0].size(), vec1d_t(backward_input[0][0].size(), 0)));
	output = MatmulBackDw(vec4d_t(1, forward_input), vec4d_t(1, backward_input))[0];
	return output;
}

vec4d_t dropoutBackward(vec4d_t backward_input, float prob, vec4d_t mask, bool is_training) {
	
	if(is_training) {
		vec4d_t output(backward_input.size(), vec3d_t(backward_input[0].size(), vec2d_t(backward_input[0][0].size(), vec1d_t(backward_input[0][0][0].size(), 0))));

		for(int i = 0; i < mask.size(); i++) {
			for(int j = 0; j < mask[0].size(); j++) {
				for(int m = 0; m < mask[0][0].size(); m++) {
					for (int n = 0; n < mask[0][0][0].size(); n++) {
						if (mask[i][j][m][n] < prob)
						{
							output[i][j][m][n] = 0.0;
						}
						else
						{
							output[i][j][m][n] = backward_input[i][j][m][n] / (1 - prob);
						}
					}
				}
			}
		}
		
		return output;
	}
	else{
		return backward_input;
	}
}

std::tuple<vec3d_t, vec3d_t> baddbmmBackward(vec3d_t backward_input, vec3d_t batch1, vec3d_t batch2, float beta, float alpha) {
	std::tuple<vec3d_t, vec3d_t> backward_output(vec3d_t(batch1.size(), vec2d_t(batch1[0].size(), vec1d_t(batch1[0][0].size(), 0))), vec3d_t(batch2.size(), vec2d_t(batch2[0].size(), vec1d_t(batch2[0][0].size(), 0))));

	// transition(backward_input);
	for (int d1 = 0; d1 < backward_input.size(); d1++) {
		for (int d2 = 0; d2 < backward_input[0].size(); d2++) {
			for (int d3 = 0; d3 < backward_input[0][0].size(); d3++) {
				backward_input[d1][d2][d3] = alpha * backward_input[d1][d2][d3];
			}
		}
	}	
	// transition(backward_input);
	// transition(batch1);
	// transition(batch2);
	for (int d1 = 0; d1 < backward_input.size(); d1++) {
		for (int d2 = 0; d2 < backward_input[0].size(); d2++) {
			for (int d3 = 0; d3 < batch2[0].size(); d3++) {
				for (int c = 0; c < backward_input[0][0].size(); c++) {
					std::get<0>(backward_output)[d1][d2][d3] += backward_input[d1][d2][c] * batch2[d1][d3][c];
				}
			}
		}
	}

	for (int d1 = 0; d1 < batch1.size(); d1++) {
		for (int d2 = 0; d2 < batch1[0][0].size(); d2++) {
			for (int d3 = 0; d3 < backward_input[0][0].size(); d3++) {
				for (int c = 0; c < batch1[0].size(); c++) {
					std::get<1>(backward_output)[d1][d2][d3] += batch1[d1][c][d2] * backward_input[d1][c][d3];
				}
			}
		}
	}
	// std::get<0>(backward_output) = backward_input;
	return backward_output;
}

std::string get_last_checkpoint(std::string folder)
{
    std::string PREFIX_CHECKPOINT_DIR = "checkpoint";
    std::regex re_checkpoint("^" + PREFIX_CHECKPOINT_DIR + R"(-(\d+)$)");
    std::vector<std::string> checkpoints;

    for (const auto& entry : std::filesystem::directory_iterator(folder))
    {
        std::string path = entry.path().string();
        std::smatch matches;

        if (std::regex_search(path, matches, re_checkpoint) && std::filesystem::is_directory(path))
        {
            checkpoints.push_back(path);
        }
    }
    if (checkpoints.empty()) return "";

    auto result = std::max_element(checkpoints.begin(), checkpoints.end(),
                        [&re_checkpoint](const std::string& path1, const std::string& path2) {
                            std::smatch match1, match2;
                            std::regex_search(path1, match1, re_checkpoint);
                            std::regex_search(path2, match2, re_checkpoint);
                            return std::stoi(match1[1].str()) < std::stoi(match2[1].str());
                        });

    return *result;
}