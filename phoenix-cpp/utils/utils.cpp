#include"utils.h"

void PrintVector(vec4d_t vec) {
	for (auto i : vec) {
		for (auto j : i) {
			for (auto x : j) {
				for (auto y : x) {
					std::cout << y << " ";
				}
				std::cout << "\n";
			}
		}
	}
}

void LoadInput(vec5d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    for (auto& t : l) {
                        std::getline(in, line);
                        assert(line.length() > 0);
                        t = atof(line.c_str());
                    }

                }
            }
        }
    }
    std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInput(vec4d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    std::getline(in, line);
                    assert(line.length() > 0);
                    l = atof(line.c_str());
                }
            }
        }
    }
    std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInput(vec3d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                std::getline(in, line);
                //assert(line.length() > 0);
                k = atof(line.c_str());
            }
        }
    }
    std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInput(vec2d_t& input_vec, std::string input_name)
{
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            std::getline(in, line);
            assert(line.length() > 0);
            j = atof(line.c_str());
        }
    }
    std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInput(vec1d_t& input_vec, std::string input_name)
{
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        std::getline(in, line);
        assert(line.length() > 0);
        i = atof(line.c_str());
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}

void WriteOutput(vec3d_t data, std::string name) {
    std::stringstream ss;
    ss << name;
    std::cout << "Write:" << ss.str() << std::endl;
    std::ofstream out(ss.str());
    for (auto& i : data) {
        for (auto& j : i) {
            for (auto& k : j) {
                out << k << "\n";
            }
        }
    }
    out.close();
}

void WriteOutput(vec4d_t data, std::string name) {
    std::stringstream ss;
    ss << name;
    std::cout << "Write:" << ss.str() << std::endl;
    std::ofstream out(ss.str());
    for (auto& i : data) {
        for (auto& j : i) {
            for (auto& k : j) {
                for (auto& m : k) {
                    out << m << "\n";
                }
            }
        }
    }
    out.close();
}
void WriteOutput(vec5d_t data, std::string name) {
    std::stringstream ss;
    ss << name;
    std::cout << "Write:" << ss.str() << std::endl;
    std::ofstream out(ss.str());
    for (auto& i : data) {
        for (auto& j : i) {
            for (auto& k : j) {
                for (auto& m : k) {
                    for (auto& t : m) {
                        out << t << "\n";
                    }
                }
            }
        }
    }
    out.close();
}
void WriteOutput(vec6d_t data, std::string name) {
    std::stringstream ss;
    ss << name;
    std::cout << "Write:" << ss.str() << std::endl;
    std::ofstream out(ss.str());
    for (auto& i : data) {
        for (auto& j : i) {
            for (auto& k : j) {
                for (auto& m : k) {
                    for (auto& t : m) {
                        for (auto& tt : t) {
                            out << tt << "\n";
                        }
                    }
                }
            }
        }
    }
    out.close();
}

void WriteOutput(vec5d_t_i data, std::string name) {
    std::stringstream ss;
    ss << name;
    std::cout << "Write:" << ss.str() << std::endl;
    std::ofstream out(ss.str());
    for (auto& i : data) {
        for (auto& j : i) {
            for (auto& k : j) {
                for (auto& m : k) {
                    for (auto& n : m) {
                        out << n << "\n";
                    }
                }
            }
        }
    }
    out.close();
}

vec4d_t BHWC2BCHW(const vec4d_t& data)
{
    int B = data.size();
    int H = data[0].size();
    int W = data[0][0].size();
    int C = data[0][0][0].size();

    vec4d_t output(B, vec3d_t(C, vec2d_t(H, vec1d_t(W))));
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    output[b][c][h][w] = data[b][h][w][c];
            }
        }
    }
    return output;
}
vec4d_t BCHW2BHWC(const vec4d_t& data)
{
    int B = data.size();
    int C = data[0].size();
    int H = data[0][0].size();
    int W = data[0][0][0].size();

    vec4d_t output(B, vec3d_t(H, vec2d_t(W, vec1d_t(C))));
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    output[b][h][w][c] = data[b][c][h][w];
            }
        }
    }
    return output;
}

vec4d_t Rearrange(const vec4d_t& data, int dim0, int dim1, int dim2, int dim3)
{
    int D1 = data.size();
    int D2 = data[0].size();
    int D3 = data[0][0].size();
    int D4 = data[0][0][0].size();

    int dims[4] = { D1,D2,D3,D4 };
    vec4d_t output(dims[dim0], vec3d_t(dims[dim1], vec2d_t(dims[dim2], vec1d_t(dims[dim3]))));
    for (int i = 0; i < D1; i++)
    {
        for (int j = 0; j < D2; j++)
        {
            for (int k = 0; k < D3; k++)
            {
                for (int l = 0; l < D4; l++)
                {
                    int pos[4] = { i,j,k,l };
                    output[pos[dim0]][pos[dim1]][pos[dim2]][pos[dim3]] = data[i][j][k][l];
                }
            }
        }
    }
    return output;
}

vec5d_t_i getHeadMask(vec5d_t_i &head_mask, uint32_t num_hidden_layers, bool is_attention_chunked) {
    vec5d_t_i output;
    if(head_mask.size() != 0) {
        output = convertHeadMaskTo5d(head_mask, num_hidden_layers);
        if(is_attention_chunked) {
            std::cerr << "gc++ is different from python \n";
            exit(-1);
        }
    }
    else {
        for (int i = 0; i < num_hidden_layers; i++) {
            vec4d_t_i temp;
            output.emplace_back(temp);
        }
    }
    return output;
}

void LoadInputHex(vec5d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << ", " << input_vec[0][0][0].size() << ", " << input_vec[0][0][0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    for (auto& t : l) {
                        std::getline(in, line);
                        assert(line.length() > 0);
                        int temp = std::strtol(line.c_str(), 0, 0);
                        float x = *(float*)(&temp);
                        t = (float)x;
                    }

                }
            }
        }
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInputHex(vec4d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << ", " << input_vec[0][0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    std::getline(in, line);
                    assert(line.length() > 0);
                    int temp = std::strtol(line.c_str(), 0, 0);
                    float x = *(float*)(&temp);
                    l = (float)x;
                }
            }
        }
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInputHex(vec3d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                std::getline(in, line);
                assert(line.length() > 0);
                int temp = std::strtol(line.c_str(), 0, 0);
                float x = *(float*)(&temp);
                k = (float)x;
            }
        }
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInputHex(vec2d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            std::getline(in, line);
            assert(line.length() > 0);
            int temp = std::strtol(line.c_str(), 0, 0);
            float x = *(float*)(&temp);
            j = (float)x;
        }
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}
void LoadInputHex(vec1d_t& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        std::getline(in, line);
        assert(line.length() > 0);
        int temp = std::strtol(line.c_str(), 0, 0);
        float x = *(float*)(&temp);
        i = (float)x;
    }
    // std::getline(in, line);
    // assert(line.length() <= 0);
    in.close();
}

void LoadInputI(vec5d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << ", " << input_vec[0][0][0].size() << ", " << input_vec[0][0][0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    for (auto& t : l) {
                        std::getline(in, line);
                        assert(line.length() > 0);
                        t = atoi(line.c_str());
                    }

                }
            }
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputI(vec4d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << ", " << input_vec[0][0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    std::getline(in, line);
                    assert(line.length() > 0);
                    l = atoi(line.c_str());
                }
            }
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputI(vec3d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                std::getline(in, line);
                assert(line.length() > 0);
                k = atoi(line.c_str());
            }
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputI(vec2d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            std::getline(in, line);
            assert(line.length() > 0);
            j = atoi(line.c_str());
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputI(vec1d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        std::getline(in, line);
        assert(line.length() > 0);
        i = atoi(line.c_str());
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}

void LoadInputB(vec5d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << ", " << input_vec[0][0][0].size() << ", " << input_vec[0][0][0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    for (auto& t : l) {
                        std::getline(in, line);
                        assert(line.length() > 0);
                        if(line == "False")
                            t = false;
                        else
                            t = true;
                    }

                }
            }
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputB(vec4d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << ", " << input_vec[0][0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                for (auto& l : k)
                {
                    std::getline(in, line);
                    assert(line.length() > 0);
                    if(line == "False")
                        l = false;
                    else
                        l = true;
                }
            }
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputB(vec3d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << ", " << input_vec[0][0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            for (auto& k : j)
            {
                std::getline(in, line);
                assert(line.length() > 0);
                if(line == "False")
                    k = false;
                else
                    k = true;
            }
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputB(vec2d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << ", " << input_vec[0].size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        for (auto& j : i)
        {
            std::getline(in, line);
            assert(line.length() > 0);
            if(line == "False")
                j = false;
            else
                j = true;
        }
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}
void LoadInputB(vec1d_t_i& input_vec, std::string input_name) {
    std::stringstream ss;
    ss << input_name;
    std::cout << "Read:" << ss.str() << std::endl;
    std::cout << "load_size: [" << input_vec.size() << "]\n";
    std::ifstream in(ss.str());
    assert(in);

    std::string line;
    for (auto& i : input_vec)
    {
        std::getline(in, line);
        assert(line.length() > 0);
        if(line == "False")
            i = false;
        else
            i = true;
    }
    std::getline(in, line);
    assert(line.length() <= 0);
    in.close();
}


vec4d_t AddVector(vec4d_t vec1, vec4d_t vec2)
{
    int B = vec1.size();
    int C = vec1[0].size();
    int H = vec1[0][0].size();
    int W = vec1[0][0][0].size();

    vec4d_t output(B, vec3d_t(C, vec2d_t(H, vec1d_t(W))));
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    output[b][c][h][w] = vec1[b][c][h][w] + vec2[b][c][h][w];
            }
        }
    }
    return output;
}
vec4d_t SubVector(vec4d_t vec1, vec4d_t vec2)
{
    int B = vec1.size();
    int C = vec1[0].size();
    int H = vec1[0][0].size();
    int W = vec1[0][0][0].size();

    vec4d_t output(B, vec3d_t(C, vec2d_t(H, vec1d_t(W))));
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    output[b][c][h][w] = vec1[b][c][h][w] - vec2[b][c][h][w];
            }
        }
    }
    return output;
}
vec4d_t MulVector(vec4d_t& vec1, vec4d_t& vec2)
{
    int B = vec1.size();
    int C = vec1[0].size();
    int H = vec1[0][0].size();
    int W = vec1[0][0][0].size();

    vec4d_t output(B, vec3d_t(C, vec2d_t(H, vec1d_t(W))));
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    output[b][c][h][w] = vec1[b][c][h][w] * vec2[b][c][h][w];
            }
        }
    }
    return output;
}
vec4d_t AddVector(vec4d_t& vec1, vec4d_t&& vec2)
{
    int B = vec1.size();
    int C = vec1[0].size();
    int H = vec1[0][0].size();
    int W = vec1[0][0][0].size();

    vec4d_t output(B, vec3d_t(C, vec2d_t(H, vec1d_t(W))));
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    output[b][c][h][w] = vec1[b][c][h][w] + vec2[b][c][h][w];
            }
        }
    }
    return output;
}
vec4d_t MulVector(vec4d_t&& vec1, vec4d_t&& vec2) {
    int B = vec1.size();
    int C = vec1[0].size();
    int H = vec1[0][0].size();
    int W = vec1[0][0][0].size();

    vec4d_t output(B, vec3d_t(C, vec2d_t(H, vec1d_t(W))));
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    output[b][c][h][w] = vec1[b][c][h][w] * vec2[b][c][h][w];
            }
        }
    }
    return output;
}

template <typename T>
vec5d_t_i convertHeadMaskTo5d(T head_mask, int num_hidden_layers) {
    int dim_size = dimension<T>::value;
    
    // if (dim_size == 1)
    // {
    //     vec5d_t_i output(1, vec4d_t_i(1, vec3d_t_i(head_mask.size(), vec2d_t_i(1, vec1d_t_i(1)))));
    //     for (int i = 0; i < head_mask.size(); i++)
    //     {
    //         output[0][0][i][0][0] = head_mask[i];
    //     }
    //     for(int i = 0; i < num_hidden_layers; i++) {
    //         output.emplace_back(output[0]);
    //     }
    //     return output;
    // }
    // else if(dim_size == 2) {
    //     vec5d_t_i output(head_mask.size(), vec4d_t_i(1, vec3d_t_i(head_mask[0].size(), vec2d_t_i(1, vec1d_t_i(1)))));
    //     for (int i = 0; i < head_mask.size(); i++)
    //     {
    //         for (int j = 0; j < head_mask[0].size(); j++) {
    //             output[i][0][j][0][0] = head_mask[i][j];
    //         }
    //     }
    //     return output;
    // }
    // assert(dim_size == 5);
    // return head_mask;
    vec5d_t_i output;
    return output;
}

template<typename T, typename ...V> 
T operator*(float val, T x) {
  int dims_size = dimension<T>::value;
  if(dims_size == 1) {
    T temp = x;
    for (int i = 0; i < x.size(); i++)
    {
        temp[i] = val * x[i];
    }
    return temp;
  }
  else{
    T temp = x;
    for (int i = 0; i < x.size(); i++)
    {
        temp[i] = val * x[i];
    }
    return temp;
  }
}

vec2d_t update_weight(vec2d_t old_weight, vec2d_t grad, float lr) {
    vec2d_t output = old_weight;

    for (uint32_t i = 0; i < old_weight.size(); i++) {
        for (uint32_t j = 0; j < old_weight[0].size(); j++) {
            old_weight[i][j] = old_weight[i][j] - grad[i][j] * lr;
        }
    }

    return output;
}

vec1d_t update_bias(vec1d_t old_bias, vec1d_t grad, float lr) {
    vec1d_t output = old_bias;

    for(int i = 0; i < old_bias.size(); i++) {
        output[i] = old_bias[i] - grad[i] * lr;
    }

    return output;
}