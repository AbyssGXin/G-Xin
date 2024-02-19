#include <iostream>

const std::string RED = "\033[1;31m";
const std::string GREEN = "\033[1;32m";
const std::string YELLOW = "\033[1;33m";
const std::string BLUE = "\033[1;34m";
const std::string WHITE = "\033[0;37m";
const std::string CYAN = "\033[0;96m";
const std::string ORANGE = "\033[38;2;255;165;0m";
const std::string MIKU = "\033[38;2;147;214;214m";
const std::string AYUMU = "\033[38;2;237;125;149m";
const std::string KANON = "\033[38;2;255;127;39m";
const std::string SETSUNA = "\033[38;2;216;28;47m";
const std::string SHIORI = "\033[38;2;55;180;132m";
const std::string CYARON = "\033[38;2;255;164;52m";

#define SHOW(c) std::cout << c << #c << ": " << "■■■■■■■■■■■■■■■■■■■■" << WHITE << std::endl

// int main() {
//     SHOW(RED);
//     SHOW(GREEN);
//     SHOW(YELLOW);
//     SHOW(BLUE);
//     SHOW(WHITE);
//     SHOW(CYAN);
//     SHOW(ORANGE);
//     SHOW(MIKU);
//     SHOW(AYUMU);
//     SHOW(KANON);
//     SHOW(SETSUNA);
//     SHOW(SHIORI);
//     SHOW(CYARON);
// }
