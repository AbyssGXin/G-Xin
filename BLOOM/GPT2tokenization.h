#pragma once
#include"json.hpp"
#include<iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include <windows.h>
#include <regex>
#include <set>
#include <algorithm>

struct hash_tuple {
	template <class T1, class T2>
	size_t operator()(const std::tuple<T1, T2>& p) const
	{
		auto hash1 = std::hash<T1>()(std::get<0>(p));
		auto hash2 = std::hash<T2>()(std::get<1>(p));
		return hash1 ^ hash2;
	}
};

class GPT2Tokenizer {
	private:
		const std::string s= R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Z0-9]+|\s+(?!\S)|\s+)";
		std::regex pat;
		std::unordered_map<std::string, std::vector<std::string>> cache;
		std::unordered_map<std::string, int> encoder;
		std::unordered_map<int, std::string> decoder;
		std::vector<std::tuple<std::string, std::string>> bpe_merges;
		std::unordered_map<std::string, int> byte_decoder;
		std::unordered_map<int, std::string> byte_encoder;

		std::unordered_map<int, std::string> byte_to_unicode();
		std::vector<std::string> load_merges_form_json(const std::string file_path);
		std::unordered_map<std::string, int> load_vocab_from_json(const std::string file_path);
		std::vector<std::string> _tokenize(std::string text);

	public:
		std::unordered_map<std::tuple<std::string, std::string>, double, hash_tuple> bpe_ranks;

		GPT2Tokenizer();
		GPT2Tokenizer(std::string vocab_file, std::string merges_file);
		std::vector<std::string> Token(std::string text);
		std::set<std::tuple<std::string, std::string>> get_pairs(std::vector<std::string> word);
		std::vector<std::string> TrieSplit(std::string);
		std::vector<std::string> bpe(std::vector<std::string> token);
		std::vector<std::string> TrieAdd(std::string);
		std::vector<std::string> TrieCutText(std::string);
		std::tuple<std::vector<int>, std::vector<int>> Encode(std::string input);
		std::string Decode(std::vector<int> token_ids);
};