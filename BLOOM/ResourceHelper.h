#pragma once
#ifndef __RESOURCE_HELPER_H__
#    define __RESOURCE_HELPER_H__

namespace FuncHelperFeatureTest
{
constexpr int RESOURCE_HELPER_VERSION = 221125;
constexpr bool ALLOC_HINT = true;
constexpr bool FIX_VMEM_MERGE_ON_FREE = true;
constexpr bool HBM_OUTBOUND_CHECK = true;
constexpr bool HBM_ADDR_INIT_GUARD = true;
constexpr bool HBM_ADDR_128_ALIGNED = true;
constexpr bool STRICT_VMEM_MANGE = true;
} // namespace FuncHelperFeatureTest

#    include <array>
#    include <cassert>
#    include <iomanip>
#    include <iostream>
#    include <map>

inline bool &
StrictVMemManage()
{
    static bool val = false;
    return val;
}

struct Resource
{
    std::array<bool, 32> vReg{};
    int nextVReg = 1;
    std::array<bool, 32> sReg{};
    int nextSReg = 1;
    std::array<bool, 8> vMask{};
    int nextVMask = 1;
    std::map<uint32_t, uint32_t> vMem;
    std::map<uint32_t, std::pair<uint32_t, std::string>> allocedBlock;

    Resource()
    {
        vReg.fill(false);
        sReg.fill(false);
        vMask.fill(false);
        sReg[0] = true;
        vReg[0] = true;
        vMask[0] = true;
        vMem[0] = GlobalDeviceConfig().VMemSize;
    }

    int
    AllocVReg()
    {
        int next = nextVReg;
        do
        {
            if (vReg[next] == false)
            {
                nextVReg = (next + 1) % vReg.size();
                vReg[next] = true;
                return next;
            }
            next = (next + 1) % vReg.size();
        } while (next != nextVReg);
        assert(false && "No More Free VReg");
        return -1;
    }

    void
    FreeVReg(int idx)
    {
        vReg[idx] = false;
    }

    int
    AllocSReg()
    {
        int next = nextSReg;
        do
        {
            if (sReg[next] == false)
            {
                nextSReg = (next + 1) % sReg.size();
                sReg[next] = true;
                return next;
            }
            next = (next + 1) % sReg.size();
        } while (next != nextSReg);
        assert(false && "No More Free SReg");
        return -1;
    }

    void
    FreeSReg(int idx)
    {
        sReg[idx] = false;
    }

    int
    AllocVMask()
    {
        int next = nextVMask;
        do
        {
            if (vMask[next] == false)
            {
                nextVMask = (next + 1) % vMask.size();
                vMask[next] = true;
                return next;
            }
            next = (next + 1) % vMask.size();
        } while (next != nextVMask);
        assert(false && "No More Free VMask");
        return -1;
    }

    void
    FreeVMask(int idx)
    {
        vMask[idx] = false;
    }

    void
    PrintAllocedInfo()
    {
        for (auto &kv : allocedBlock)
        {
            std::clog << "VMem " << kv.first << "(0x" << std::hex << kv.first
                      << std::dec << ") To " << kv.first + kv.second.first
                      << "(0x" << std::hex << kv.first + kv.second.first
                      << std::dec << ") Size: " << kv.second.first << "(0x"
                      << std::hex << kv.second.first << std::dec << ")";
            if (!kv.second.second.empty())
            {
                std::clog << " For: " << kv.second.second << "\n";
            }
            else
            {
                std::clog << "\n";
            }
        }
    }

    void
    StatusCheck()
    {
        if (!StrictVMemManage())
        {
            return;
        }
        uint32_t addr = 0;
        while (addr < GlobalDeviceConfig().VMemSize)
        {
            if (vMem.count(addr) != 0)
            {
                if (addr == vMem[addr])
                {
                    vMem.erase(addr);
                }
                else
                {
                    addr = vMem[addr];
                }
            }
            else if (allocedBlock.count(addr) != 0)
            {
                addr += allocedBlock[addr].first;
            }
            else
            {
                std::cerr << COLOR::RED << "!VMEM and AllocedBlock Conflight!"
                          << COLOR::WHITE << "\n";
                PrintAllocedInfo();
                assert(false);
            }
        }
        if (addr != GlobalDeviceConfig().VMemSize)
        {
            std::cerr << COLOR::RED << "!VMEM and AllocedBlock Conflight!"
                      << COLOR::WHITE << "\n";
            PrintAllocedInfo();
            assert(false);
        }
    }

    void
    PrintInfo(const std::string &act,
              uint32_t addr,
              uint32_t size,
              const std::string &usage = "")
    {
        std::clog << act << " " << addr << "(0x" << std::hex << addr << std::dec
                  << ") Size: " << size << "(0x" << std::hex << size << std::dec
                  << ")";
        if (!usage.empty())
        {
            std::clog << " For: " << usage << "\n";
        }
        else
        {
            std::clog << "\n";
        }
    }

    uint32_t
    AllocVMem(uint32_t size, uint32_t align = 1, const std::string &usage = "")
    {
        int64_t addr = -1;
        for (auto &kv : vMem)
        {
            if (size == 0)
            {
                return kv.first;
            }
            if (kv.second - kv.first >= size)
            {
                addr = kv.first;
                auto newAddr = ((addr + align - 1) / align) * align;
                if (kv.second - newAddr < size)
                {
                    addr = -1;
                    continue;
                }
                if (newAddr + size != kv.second)
                {
                    vMem[newAddr + size] = kv.second;
                }
                if (newAddr + size != addr)
                {
                    kv.second = newAddr;
                }
                if (addr == newAddr && size != 0)
                {
                    vMem.erase(addr);
                }
                addr = newAddr;
                break;
            }
        }
        if (addr == -1)
        {
            std::clog << COLOR::SETSUNA << "OOM When Alloc " << size << "\n";
            PrintAllocedInfo();
            std::clog << COLOR::WHITE;
        }
        assert(addr != -1 && "VMem Alloc Failed, No Satisfied Free VMem");
        allocedBlock[addr] = std::make_pair(size, usage);
        StatusCheck();
        return addr;
    }

    uint32_t
    AllocVMemWithHint(uint32_t size,
                      uint32_t hint,
                      const std::string &usage = "")
    {
        int64_t addr = -1;
        for (auto &kv : vMem)
        {
            if (size == 0)
            {
                return kv.first;
            }
            if (kv.first <= hint && hint + size <= kv.second)
            {
                addr = hint;
                auto newAddr = addr;
                if (kv.second - newAddr < size)
                {
                    addr = -1;
                    continue;
                }
                if (newAddr + size != kv.second)
                {
                    vMem[newAddr + size] = kv.second;
                }
                if (newAddr + size != addr)
                {
                    kv.second = newAddr;
                }
                if (addr == newAddr && size != 0)
                {
                    vMem.erase(addr);
                }
                addr = newAddr;
                break;
            }
        }
        if (addr == -1)
        {
            std::clog << COLOR::SETSUNA << "No Enough in Addr " << hint
                      << ", Fallback to Normal Alloc\n";
            std::clog << COLOR::WHITE;
            addr = AllocVMem(size, 1, usage);
        }
        assert(addr != -1 && "VMem Alloc Failed, No Satisfied Free VMem");
        allocedBlock[addr] = std::make_pair(size, usage);
        StatusCheck();
        return addr;
    }

    void
    FreeVMem(uint32_t addr, uint32_t size)
    {
        if (StrictVMemManage())
        {
            if (allocedBlock.count(addr) == 0)
            {
                assert(size == 0);
            }
            else
            {
                assert(allocedBlock[addr].first == size || size == 0);
            }
        }
        if (size == 0)
        {
            return;
        }
        int lr = 2;
        auto lb = vMem.lower_bound(addr);
        if (!vMem.empty() && lb != vMem.begin() && --lb != vMem.end())
        {
            if (addr < lb->second)
            {
                std::clog << "Illegal State When Free " << addr
                          << ", Size: " << size << "\n";
                PrintAllocedInfo();
            }
            assert(addr >= lb->second && "Free A Free VMem, Illegal State");
            if (addr == lb->second)
            {
                lb->second += size;
                lr--;
            }
        }
        auto rb = vMem.lower_bound(addr + size);
        if (rb != vMem.end())
        {
            if (rb->first < addr + size)
            {
                std::clog << "Illegal State When Free " << addr
                          << ", Size: " << size << "\n";
                PrintAllocedInfo();
            }
            assert(rb->first >= addr + size &&
                   "Free A Free VMem, Illegal State");
            if (rb->first == addr + size)
            {
                if (lr == 1)
                {
                    lb->second += rb->second - rb->first;
                    vMem.erase(rb->first);
                }
                else
                {
                    auto end = rb->second;
                    vMem.erase(rb->first);
                    vMem[addr] = end;
                }
                lr--;
            }
        }
        if (lr == 2)
        {
            vMem[addr] = addr + size;
        }
        allocedBlock.erase(addr);
        StatusCheck();
    }

    uint32_t
    BiggestContinuousAvailableVMemSize(int align = 1) const
    {
        uint32_t size = 0;
        for (auto &kv : vMem)
        {
            auto addrStart = ((kv.first + align - 1) / align) * align;
            size = std::max(size, kv.second - addrStart);
        }
        return size;
    }
};

struct HBMAddrWarp
{
    uint32_t &val;
    bool &inited;
    HBMAddrWarp(uint32_t &v, bool &f) : val(v), inited(f) {}

    void
    RequireInited() const
    {
        if (!inited)
        {
            std::clog << COLOR::RED << "HBMAddr is uninitialized!"
                      << COLOR::WHITE << std::endl;
            getchar();
        }
    }

    operator uint32_t() const
    {
        RequireInited();
        if (val % 128 != 0)
        {
            val = ((val + 127) / 128) * 128;
        }
        return val;
    }

    HBMAddrWarp(const HBMAddrWarp &) = delete;
    HBMAddrWarp &operator=(const HBMAddrWarp &) = delete;
    HBMAddrWarp(HBMAddrWarp &&) = default;
    HBMAddrWarp &
    operator=(const uint32_t newval)
    {
        inited = true;
        val = newval;
        return *this;
    }

    HBMAddrWarp &
    operator+=(const uint32_t val)
    {
        RequireInited();
        this->val += val;
        return *this;
    }

    HBMAddrWarp &
    GetOr(uint32_t v)
    {
        if (!inited)
        {
            inited = true;
            val = v;
        }
        return *this;
    }
};

inline HBMAddrWarp
HBMAddr()
{
    static uint32_t addr = 0;
    static bool inited = false;
    if (addr >= GlobalDeviceConfig().HBMSize)
    {
        std::clog << COLOR::RED << "OUT OF HBM\n" << COLOR::WHITE;
    }
    return HBMAddrWarp(addr, inited);
}

#endif
