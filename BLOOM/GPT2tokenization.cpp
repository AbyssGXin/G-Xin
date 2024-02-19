#include"GPT2tokenization.h"

// 接受一个text字符串作为输入，将其放入一个只包含该字符串的字符串向量result中，并返回
std::vector<std::string> GPT2Tokenizer::TrieSplit(std::string text) {
	std::vector<std::string> result{ text };
	return result;
}

/*
	该函数从指定的JSON文件中加载词汇表(vocabulary)数据。它打开文件file_path,读取JSON内容并解析为nlohmanh::json对象j.
	然后，从j中提取出词汇表数据，并将其存储_vocab中，并返回该词汇表。
*/ 
std::unordered_map<std::string, int> GPT2Tokenizer::load_vocab_from_json(const std::string file_path) {
	std::fstream fs(file_path);
	nlohmann::json j;
	fs >> j;
	auto& model_j = j["model"];
	const auto& vocab_j = model_j["vocab"];
	std::unordered_map<std::string, int> _vocab =  vocab_j.get<std::unordered_map<std::string, int>>();
	return _vocab;
}

/*
	这个函数从指定的JSON文件中加载合并列表(merges)。它打开文件file_path，读取JSON内容并解析为nlohmann::json对象j。
	然后，他从j中提取出合并列表数据，并将其存储在类型为std::vector<std::string>的变量_merges中。
*/
std::vector<std::string> GPT2Tokenizer::load_merges_form_json(const std::string file_path) {
	std::fstream fs(file_path);
	nlohmann::json j;
	fs >> j;
	auto& model_j = j["model"];
	const auto& merges_j = model_j["merges"];
	std::vector<std::string>  _merges = merges_j.get<std::vector<std::string>>();
	return _merges;
}

// 用于将给定的UTF-32编码code转换为UTF-8字符串，并将结果存储在string中
void utf32_to_utf8_string(uint32_t code, char* string) {
	if (code < 0x80) string[0] = code;
	else if (code < 0x800) {   // 00000yyy yyxxxxxx
		string[0] = (0b11000000 | (code >> 6));
		string[1] = (0b10000000 | (code & 0x3f));
	}
	else if (code < 0x10000) {  // zzzzyyyy yyxxxxxx
		string[0] = (0b11100000 | (code >> 12));         // 1110zzz
		string[1] = (0b10000000 | ((code >> 6) & 0x3f)); // 10yyyyy
		string[2] = (0b10000000 | (code & 0x3f));        // 10xxxxx
	}
	else if (code < 0x200000) { // 000uuuuu zzzzyyyy yyxxxxxx
		string[0] = (0b11110000 | (code >> 18));          // 11110uuu
		string[1] = (0b10000000 | ((code >> 12) & 0x3f)); // 10uuzzzz
		string[2] = (0b10000000 | ((code >> 6) & 0x3f)); // 10yyyyyy
		string[3] = (0b10000000 | (code & 0x3f));         // 10xxxxxx
	}
}

// 用于将给定的UTF-8字符串u8转换为UTF-32编码，并将结果存储在out向量中
void utf8_to_utf32(std::string const& u8, std::vector<uint32_t>& out) {
	int elem_len = 1;
	out.clear();
	for (size_t i = 0; i < u8.size(); i += elem_len) {
		uint32_t tmp = (uint32_t)u8[i] & 0xff;
		if (tmp < 0x80UL) {
			elem_len = 1;
			out.push_back(u8[i]);
		}
		else if (tmp < 0xe0UL) {
			elem_len = 2;
			out.push_back(
				((u8[i] & 0x1f) << 6)
				| (u8[i + 1] & 0x3f)
			);
		}
		else if (tmp < 0xf0UL) {
			elem_len = 3;
			out.push_back(
				((u8[i] & 0xf) << 12)
				| ((u8[i + 1] & 0x3f) << 6)
				| (u8[i + 2] & 0x3f)
			);
		}
		else if (tmp < 0xf8UL) {
			elem_len = 4;
			out.push_back(
				((u8[i] & 0x7) << 18)
				| ((u8[i + 1] & 0x3f) << 12)
				| ((u8[i + 2] & 0x3f) << 6)
				| (u8[i + 3] & 0x3f)
			);
		}
		else {
			return;
		}
	}
}

/*
	用于生成字节码与Unicode字符的映射关系。首先，定义了一些变量: bs时存储字节码的向量
  cs是存储Unicode字符编码的向量，_cs是存储UTF-8编码的字符串向量，n是一个计数器，
  ss是一个字符串流，temp是一个临时字符串，_map是最终的字节码与Unicode字符的映射关系

	接下来通过循环将ASCII码范围内的字符添加到bs中。然后，将bs的内容复制给cs

	接下来的的循环用于生成剩余的字符编码。通过遍历0到255之间的所有可能的字节码，如果该字节码不在
  bs中，就将其添加到bs和cs中，并增加n的计数。这样可以保证所有的字节码都有对应的Unicode字符编码

	然后，通过将Unicode字符编码转换为UTF-8字符串，将其存储在_cs中

	最后，通过将bs和_cs中的对应元素一一映射，生成字节码与Unicode字符的映射关系
*/
std::unordered_map<int, std::string> GPT2Tokenizer::byte_to_unicode() {
	std::vector<unsigned short> bs;
	std::vector<unsigned short> cs;
	std::vector<std::string> _cs;
	uint32_t n = 0;
	std::stringstream ss;
	std::string temp;
	std::unordered_map<int, std::string> _map;

	for (wchar_t i = L'!'; i <= L'~'; i++) {
		bs.emplace_back((unsigned short)i);
	}
	for (wchar_t i = L'¡'; i <= L'¬'; i++) {
		bs.emplace_back((unsigned short)i);
	}
	for (wchar_t i = L'®'; i <= L'ÿ'; i++) {
		bs.emplace_back((unsigned short)i);
	}

	cs = bs;
	for (unsigned short i = 0; i < (unsigned short)pow(2, 8); i++) {
		if (find(bs.begin(), bs.end(), i) == bs.end()) {
			bs.emplace_back((unsigned short)i);
			//不是很懂这里什么会加256
			cs.emplace_back((unsigned short)pow(2, 8) + (unsigned short)n);
			n++;
		}
	}

	for (auto& i : cs) {
		char temp[4] = {0, 0, 0, 0};
		utf32_to_utf8_string(i, temp);
		_cs.emplace_back(temp);
	}
	
	for (uint32_t i = 0; i < _cs.size(); i++) {
		_map[bs[i]] = _cs[i];
	}

	return _map;
}

GPT2Tokenizer::GPT2Tokenizer() {
}

/*
	定义了GPT2Tokenizer的构造函数。在构造函数中，首先调用byte_to_unicode函数生成字节码与Unicode字符的
  映射关系，并将结果存储在byte_encode和byte_decoder中。

	然后，它调用load_vocab_from_json函数从指定的词汇表文件中加载词汇表数据，并将结果存储在encoder和decoder中。
  encoder是将词汇表中的单词映射到整数标号，decoder是将整数编号映射回对应的单词。

	接下来，它再次调用byte_to_union函数生成字节码与Unicode字符的映射关系，并将结果存储在byte_encoder和byte_decoder中。

	最后，它调用load_merges_form_json函数从指定的合并列表文件中加载合并列表数据，，并将结果
  存储在bpe_merges中。同时，它使用bpe_ranks字典将合并列表中的元素与索引值关联起来。
*/
GPT2Tokenizer::GPT2Tokenizer(std::string vocab_file, std::string merges_file):pat(s) {
	byte_encoder = byte_to_unicode();
	for (auto& i : byte_encoder) {
		byte_decoder[i.second] = i.first;
	}

	encoder = load_vocab_from_json(vocab_file);
	for (auto& i : encoder) {
		decoder[i.second] = i.first;
	}

	byte_encoder = byte_to_unicode();
	for (auto& i : byte_encoder) {
		byte_decoder[i.second] = i.first;
	}

	std::vector<std::string> _bpe_merges = load_merges_form_json(merges_file);
	double index = 0;
	for (auto i : _bpe_merges) {
		std::tuple<std::string, std::string> temp;
		uint32_t pos = i.find(' ');
		std::get<0>(temp) = i.substr(0, pos);
		std::get<1>(temp) = i.substr(pos + 1, i.size() - pos);
		bpe_merges.emplace_back(temp);
		bpe_ranks[temp] = index++;
	}
}

/*
	接受一个单词的字符串向量word作为输入，返回一个包含所有相邻字符对的集合pairs。它通过遍历word中
  的每个单词，将当前单词和前一个单词组成一个字符对，并将字符对插入到pairs集合中。
*/
std::set<std::tuple<std::string, std::string>> GPT2Tokenizer::get_pairs(std::vector<std::string> word) {
	std::set<std::tuple<std::string, std::string>> pairs;
	std::string prev_char = "";
	prev_char = word[0];

	for (int i = 1; i < word.size(); i++) {
		std::string x = "";
		x = word[i];
		pairs.insert(std::make_tuple(prev_char, x));
		prev_char = x;
	}

	return pairs;
}

/*
	用于执行BPE(Byte Pair Encoding)算法。它接受一个单词的字符串向量"token"作为输入，并返回经过BPE处理后的单词的字符串向量。

	首先，将token赋值给word，并通过连接所有单词形成一个字符串_token

	接下来，它检查缓存中是否已经存在_token的处理结果，如果存在则直接返回缓存中的结果。

	然后，它调用get_pairs函数获取word中所有相邻字符对的集合pairs。

	接下来，进入一个循环，不断执行以下步骤，直到无法再进行合并。

	1·从pairs集合中取出排在最前面的字符对，并将其复制给bigram

	2·设置一个布尔变量is_null为true，表示是否找到了有效的字符对

	3·遍历pairs集合中的每个字符对_x，如果当前字符对bigram再bpe_ranks中存在，并且bigram是空或者_x再bpe_ranks中的值比bigram的值大，则
  更新bigram为_x，并将is_null设置为false
	4·如果bigram再bpe_ranks中不存在， 表示无法再进行合并，跳出循环
	5·从bigram中获取第一个字符复制给first，第二个字符赋值给second
	6·创建一个新的字符串向量new_word和一个整数变量i，初始值为0
	7·进入一个循环，在循环中:
		1·查找word中从索引i开始的第一个出现first的位置，并将其赋值给迭代器_j
		2·如果_j等于word.end()，表示没有找到first，则将_j之后的所有单词添加到new_word中，并跳出循环。
		3·否则，将_j之前的所有单词添加到new_word中，更新i为_j在word中的索引。
		4·如果word[i]等于first且i小于word.size()-1且word[i+1]==second，表示找到了可以合并的字符对，
	  将合并后的字符添加到new_word中，并将i增加2
	  	5·否则，将word[i]添加到new_word中，并将i增加1
	8·将word更新为new_word
	9·如果word的大小为1，表示无法再进行合并，跳出循环;否则，重新计算pairs集合
	10·将word中的单词用空格连接成一个字符串_word
*/
std::vector<std::string> GPT2Tokenizer::bpe(std::vector<std::string> token) {
	std::vector<std::string> word(token);
	std::string _token = "";
	for (auto i : token) {
		_token += i;
	}
	if (cache.find(_token) != cache.end()) {
		return cache[_token];
	}

	std::set<std::tuple<std::string, std::string>> pairs = get_pairs(word);

	while (true) {
		std::tuple<std::string, std::string> bigram = *pairs.begin();
		bool is_null = true;
		for (auto _x : pairs) { 
			if (bpe_ranks.find(_x) != bpe_ranks.end() && (is_null || bpe_ranks.at(bigram) > bpe_ranks.at(_x))) {
				bigram = _x;
				is_null = false;
			}
		}

		if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
			break;
		}

		std::string first = std::get<0>(bigram);
		std::string second = std::get<1>(bigram);
		std::vector<std::string> new_word;
		int i = 0;
		while (i < word.size()) {
			auto _j = std::find(word.begin() + i, word.end(), first);
			if (_j == word.end()) {
				for (auto it = word.begin() + i; it < word.end(); it++) {
					new_word.push_back(*it);
				}
				break;
			}
			else {
				for (auto it = word.begin() + i; it < _j; it++) {
					new_word.push_back(*it);
				}
				i =_j - word.begin();
			}

			if (word[i] == first && i < word.size() - 1&& word[i + 1] == second) {
				new_word.push_back(first + second);
				i += 2;
			}
			else {
				new_word.push_back(word[i]);
				i += 1;
			}
		}
		word = new_word;
		if (word.size() == 1) {
			break;
		}
		else{
			pairs = get_pairs(word);
		}
	}

	std::string _word = "";
	for (int i = 0; i < word.size(); i++) {
		if (i != word.size() - 1) {
			_word += (word[i] + " ");
		}
		else {
			_word += word[i];
		}
	}

	cache[_token] = word;
	return word;
}

/*
	接受一个字符串text作为输入，将输入文本进行切分，并返回切分后的字符串向量。
*/
std::vector<std::string> GPT2Tokenizer::_tokenize(std::string text) {
	// 创建一个空的字符串向量bpe_tokens用于存储切分后的片段。
	std::vector<std::string> bpe_tokens;
	// 一个正则表达式匹配结果的对象。
	std::smatch result;
	// iter_begin和iter_end分别是输入字符串text的迭代器指示了遍历范围，初始值为text的开头和结尾。
	std::string::const_iterator iter_begin = text.cbegin();
	std::string::const_iterator iter_end = text.cend();
	// 使用正则表达式pat搜索匹配iter_begin和iter_end范围内的文本，将第一个匹配结果存储在result中。
	//进入循环直到无法找到匹配结果
	while (std::regex_search(iter_begin, iter_end, result, pat)) {
		// 将result[0](第一个匹配的字符串)中的每个字符转换为对应的编码，并将编码后的字符添加到临时向量temp中。
		std::vector<std::string> temp;

		// 将临时向量temp中的编码片段作为输入传递bpe函数进行更细粒度的切分，返回切分后的向量_temp。然后将
		//_temp中的每个元素添加到bpe_token中。
		for (uint32_t i = 0; i < result[0].str().size(); i++) {
			temp.push_back([result[0].str()[i]]);
		}
		
		// 将临时变量temp中的编码片段作为输入传递给bpe函数进行更细粒度的切片，返回切分后的向量_temp。然后
		//将_temp中的每个元素添加到bpe_tokens中。
		std::vector<std::string> _temp = bpe(temp);
		for (auto i : _temp) {
			bpe_tokens.push_back(i);
		}
		// 更新iter_begin的位置，使其指向当前匹配结果的结尾，以便继续下一次匹配。
		iter_begin = result[0].second;
	}

	return bpe_tokens;
}

std::vector<std::string> GPT2Tokenizer::Token(std::string text) {
	std::vector<std::string> no_split_token{ "<|endoftext|>" };
	std::vector<std::string> tokens = TrieSplit(text);
	std::vector<std::string> tokenized_text;

	tokens = _tokenize(text);
	for (auto i : tokens) {
		tokenized_text.emplace_back(i);
	}

	return tokenized_text;
}

std::tuple<std::vector<int>, std::vector<int>> GPT2Tokenizer::Encode(std::string input) {
	// 声明一个字符串向量，用于存储分词后的文本
	std::vector<std::string>  tokenized_text;
	// 声明一个整数向量用于存储编码后的标记ID
	std::vector<int> ids;
	// 声明一个整数向量用于存储标记的掩码
	std::vector<int> mask;
	// 对输入文本进行分词，并将结果存储到tokenized_text中
	tokenized_text = Token(input);

	// 遍历分词后的文本，将每个标记对应的ID添加到ids中
	for (auto i : tokenized_text) {
		ids.emplace_back(encoder[i]);
	}

	// 为每个标记添加掩码值1，表示这些标记是有效的
	for(int i = 0; i < tokenized_text.size(); i++) {
		mask.push_back(1);
	}
	// 返回编码后的标记ID和标记掩码
	return std::make_tuple(ids, mask);
}

std::string GPT2Tokenizer::Decode(std::vector<int> token_ids) {
	// 声明一个字符串向量用于存储解码后的标记
	std::vector<std::string> tokens;
	// 声明一个32为无符号整数向量用于存储解码后的标记
	std::vector<uint32_t> tokens_decode;
	// 声明一个字符串用于存储解码后的文本
	std::string text = "";
	
	// 将每个标记ID解码为对应的字符串标记，并将结果存储到tokens中
	for (auto i : token_ids) {
		tokens.push_back(decoder[i]);
	}

	// 遍历解码后的标记，将UTF-8编码的字符串转换为UTF-32编码表示，并将结果存储到tokens_decode中
	for (auto& i : tokens) {
		std::string temp;
		utf8_to_utf32(i, tokens_decode);
		
		// 遍历UTF-32编码表示的标记，将其转换为UTF-8编码的字符串，并通过字节解码器获取原始字符
		for (auto i : tokens_decode) {
			char str[4] = { 0, 0, 0, 0 };
			utf32_to_utf8_string(i, str);
			temp += byte_decoder[str];
		}
		// 更新解码后的标记为原始字符形式
		i = temp;
		// 清空tokens_decode向量
		tokens_decode.clear();
	}

	// 讲解码后的编辑拼接成最终的文本
	for (auto i : tokens) {
		text += i;
	}

	return text;
}