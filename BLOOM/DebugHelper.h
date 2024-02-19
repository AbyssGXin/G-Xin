#pragma once

#ifndef _DEBUG_HELPER_
#    define _DEBUG_HELPER_
#    include <string>
#    include <sstream>
#    include <unordered_map>
#    include <unordered_set>

namespace FuncHelperFeatureTest
{
constexpr int DEBUG_HELPER_VERSION = 221122;
constexpr bool COLOR_12 = true;
constexpr bool COLOR_CYARON = true;
constexpr bool GLOBAL_DIFF_TYPE_SET = true;
} // namespace FuncHelperFeatureTest

#    include <cassert>
#    include <cfloat>
#    include <cmath>
#    include <functional>
#    include <iostream>
#    include <utility>
#    include <vector>

enum class DebugLevel
{
    Silence = 0,
    Warn = 1,
    Info = 2,
    Verbose = 3
};

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

enum class DiffType
{
    RelativeChange,
    RelativeDifference
};

struct Statement
{
    DiffType diffType = DiffType::RelativeDifference;
    std::unordered_map<std::string, int> callCount{};
};

inline Statement &
GlobalState()
{
    static Statement state;
    return state;
}

template <class I, class O>
std::vector<O>
Map(const std::vector<I> &in, std::function<O(I)> func)
{
    std::vector<O> out(in.size());
    for (int i = 0; i < in.size(); i++)
    {
        out[i] = func(in[i]);
    }
    return out;
}

inline double
Diff(double test, double truth)
{
    if (test == truth)
    {
        return 0.0;
    }
    switch (GlobalState().diffType)
    {
    case DiffType::RelativeChange:
        return std::abs((test - truth) / (truth == 0.0 ? 1.0 : truth));
    case DiffType::RelativeDifference:
        return std::abs(test - truth) /
               std::max(std::abs(test), std::abs(truth));
    default:
        return 1.0;
    }
}

inline void
ShowData(const std::string &name,
         const std::vector<int32_t> &data,
         bool isFloat,
         int warpC = 8)
{
    std::clog << "=====" << name << "=====\n";
    int warp = 0;
    for (auto &v : data)
    {
        if (isFloat)
        {
            fprintf(stderr, "%.10e, ", *reinterpret_cast<const float *>(&v));
        }
        else
        {
            fprintf(stderr, "%d, ", v);
        }
        warp++;
        if (warp == warpC)
        {
            warp = 0;
            fprintf(stderr, "\n");
        }
    }
    if (warp != 0)
    {
        fprintf(stderr, "\n");
    }
}

inline void
ShowDataT(const std::string &name,
          const std::vector<int32_t> &data,
          bool isFloat,
          size_t _warpC = 8)
{
    assert(data.size() % 128 == 0);
    std::clog << "=====" << name << "^T=====\n";
    size_t warp = 0;
    const auto warpC = std::min(data.size() / 128, _warpC);
    for (size_t i = 0; i < data.size(); i++)
    {
        const auto base = (i / 1024) * 1024;
        const auto off = i % 1024;
        auto v = data[base + (off / 8) + (off % 8) * 128];
        if (isFloat)
        {
            fprintf(stderr, "%.10e, ", *reinterpret_cast<const float *>(&v));
        }
        else
        {
            fprintf(stderr, "%d, ", v);
        }
        warp++;
        if (warp == warpC)
        {
            warp = 0;
            fprintf(stderr, "\n");
        }
    }
    if (warp != 0)
    {
        fprintf(stderr, "\n");
    }
}

template <class T>
void
ShowData(const std::string &name, const std::vector<T> &data, int warpC = 8)
{
    std::clog << "=====" << name << "=====\n";
    int warp = 0;
    for (auto &v : data)
    {
        std::clog << v << ", ";
        warp++;
        if (warp == warpC)
        {
            warp = 0;
            std::clog << "\n";
        }
    }
    if (warp != 0)
    {
        std::clog << "\n";
    }
}

template <>
inline void
ShowData<float>(const std::string &name,
                const std::vector<float> &data,
                int warpC)
{
    std::clog << "=====" << name << "=====\n";
    int warp = 0;
    for (auto &v : data)
    {
        fprintf(stderr, "%.10e, ", v);
        warp++;
        if (warp == warpC)
        {
            warp = 0;
            fprintf(stderr, "\n");
        }
    }
    if (warp != 0)
    {
        fprintf(stderr, "\n");
    }
}

template <class It>
void
ShowData(const std::string &name, const It &begin, const It &end, int warpC = 8)
{
    std::clog << "=====" << name << "=====\n";
    int warp = 0;
    for (auto it = begin; it != end; ++it)
    {
        auto v = *it;
        std::clog << v << ", ";
        warp++;
        if (warp == warpC)
        {
            warp = 0;
            std::clog << "\n";
        }
    }
    if (warp != 0)
    {
        std::clog << "\n";
    }
}

inline void
ShowData(const std::string &name,
         const std::vector<float> &data,
         const std::vector<float> &should,
         double threshold = 0.05,
         int warpC = 8)
{
    std::clog << "=====" << name << "=====\n";
    int warp = 0;
    int idx = 0;
    for (auto &v : data)
    {
        if (warp == 0)
        {
            fprintf(stderr, "%6d :", idx);
        }
        const auto cmp = should[idx++];
        const double diff = Diff(v, cmp);
        if (diff > threshold || std::isnan(v))
        {
            fprintf(stderr,
                    "%s%3.5e(%3.2f%%)[%3.5e]%s, ",
                    COLOR::RED,
                    v,
                    diff * 100.0,
                    cmp,
                    COLOR::WHITE);
            // std::clog << COLOR::RED << v << COLOR::WHITE << ", ";
        }
        else if (diff < 0.000001)
        {
            fprintf(stderr,
                    "%s%3.5e(%3.2f%%)[%3.5e]%s, ",
                    COLOR::GREEN,
                    v,
                    diff * 100.0,
                    cmp,
                    COLOR::WHITE);
        }
        else
        {
            fprintf(stderr,
                    "%s%3.5e(%3.2f%%)[%3.5e]%s, ",
                    COLOR::CYAN,
                    v,
                    diff * 100.0,
                    cmp,
                    COLOR::WHITE);
            // std::clog << COLOR::GREEN << v << COLOR::WHITE << ", ";
        }
        warp++;
        if (warp == warpC)
        {
            warp = 0;
            fprintf(stderr, "\n");
        }
    }
    if (warp != 0)
    {
        printf("\n");
    }
}

struct ArgHelper
{
    std::unordered_map<std::string, size_t> opts{};
    std::unordered_map<std::string, std::string> optsInfo{};
    std::unordered_map<std::string,
                       std::function<void(const std::vector<std::string> &)>>
        optsCb{};

    ArgHelper()
    {
        reg(
            "--help",
            0,
            [this](const std::vector<std::string> &)
            {
                for (const auto &opt : opts)
                {
                    std::cout << opt.first << " accept " << opt.second
                              << " argument(s) " << optsInfo[opt.first] << "\n";
                }
                exit(0);
            },
            "Show all registered command-line options");
    }

    bool
    reg(const std::string &name,
        size_t argsCnt,
        std::function<void(const std::vector<std::string> &)> cb,
        const std::string &info = "")
    {
        if (opts.count(name) != 0)
        {
            return false;
        }
        opts[name] = argsCnt;
        optsCb[name] = std::move(cb);
        optsInfo[name] = info;
        return true;
    }

    void
    parse(int argc, char const *argv[])
    {
        for (int i = 1; i < argc; i++)
        {
            if (opts.count(argv[i]) != 0)
            {
                std::string cmd = argv[i];
                size_t argCnt = opts[cmd];
                auto &cb = optsCb[cmd];
                std::vector<std::string> args;
                size_t j = 0;
                for (; i + 1 < argc && j < argCnt; j++, i++)
                {
                    args.emplace_back(argv[i + 1]);
                }
                if (j == argCnt)
                {
                    cb(args);
                }
                else
                {
                    std::cerr << COLOR::RED << cmd << " REQUIRE " << argCnt
                              << " ARGUMENT(S) BUT GOT " << j << COLOR::WHITE
                              << std::endl;
                }
            }
            else
            {
                std::cerr << COLOR::RED << "UNKNOWN ARGUMENT: " << argv[i]
                          << COLOR::WHITE << std::endl;
            }
        }
    }
};

namespace ArgHelperUtils
{
template <class T>
T
FromStr(const std::string &s)
{
    std::cerr << "CAN'T FROM STR " << s << std::endl;
    return T{};
}

template <>
inline float
FromStr<float>(const std::string &str)
{
    return std::stof(str);
}

template <>
inline int
FromStr<int>(const std::string &str)
{
    return std::stoi(str);
}

template <>
inline uint32_t
FromStr<uint32_t>(const std::string &str)
{
    return std::stoul(str);
}

template <>
inline std::string
FromStr<std::string>(const std::string &str)
{
    return str;
}

template <class T>
std::function<void(std::vector<std::string>)>
Set(T &ref)
{
    return [&ref](const std::vector<std::string> &args)
    { ref = FromStr<T>(args[0]); };
}

inline std::function<void(std::vector<std::string>)>
Flag(bool &ref)
{
    return [&ref](const std::vector<std::string> &) { ref = true; };
}

inline std::function<void(std::vector<std::string>)>
DownFlag(bool &ref)
{
    return [&ref](const std::vector<std::string> &) { ref = false; };
}

template <class T>
std::function<void(std::vector<std::string>)>
Assign(T &ref, T value)
{
    return [&ref, value](const std::vector<std::string> &) { ref = value; };
}
} // namespace ArgHelperUtils

inline int
CallCount(const std::string &funcIdf)
{
    return GlobalState().callCount[funcIdf]++;
}

template<class T>
std::string HookCOut(T&& t) {
    /*
    --TODO: 输出
        1· 保存当前的标准输出流缓冲区
        2· 创建一个字符串输出流对象out
        3· 将标准输出流重定向到out对象，以便捕获输出
        4· 调用传递给函数的可调用对象t
        5· 恢复标准输出流的原始缓冲区
        6· 将捕获到的输出作为字符串返回
    */
    auto oldCoutRdBuf = std::cout.rdbuf();
    std::ostringstream out;
    std::cout.rdbuf(out.rdbuf());
    t();
    std::cout.rdbuf(oldCoutRdBuf);
    return out.str();
}

enum class CaseId
{
    DEFAULT,
    SCALAR_PROCESSOR
};

template <typename Base, typename T, CaseId CaseId = CaseId::DEFAULT>
struct ThiefHandle
{
    using type = T Base::*;
    friend type get(ThiefHandle);
};

template <typename Tag, typename Tag::type M>
struct Thief
{
    friend typename Tag::type
    get(Tag)
    {
        return M;
    }
};

#endif
