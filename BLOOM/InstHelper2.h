#pragma once

#ifndef _INST_HELPER2_H_
#    define _INST_HELPER2_H_

namespace FuncHelperFeatureTest
{
constexpr int INST_HELPER2_VERSION = 221129;
constexpr bool ALLOC_USAGE = true;
constexpr bool INST2_HAS_SPY_SERIES = true;
constexpr bool INST2_CO_EXEC = true;
constexpr bool INST2_RESOURCE_STACK = true;
constexpr bool IS_FLOAT_LATE_INIT = true;
constexpr bool INST2_EXEC_CHECK = true;
} // namespace FuncHelperFeatureTest

#    include "DebugHelper.h"
#    include "InstHelper.h"
#    include <cstdint>

#    include "ResourceHelper.h"

namespace InstScheduler
{
    struct Inst233;
    struct DAG;
namespace ConstrainScheduler 
{
    std::vector<Instruction *> fuse(InstScheduler::DAG& g, int upper, bool coutInfo = false);
}
}

struct Inst2;
struct VReg;
struct VRegSlice;
struct VRegSliceDyn;
struct VRegPx;
struct SReg;
struct VMem;
struct VMemDyn;
struct VMask;
struct Perm;

template <class T>
struct LateInit
{
    bool unset = true;
    T val{};

    LateInit() = default;
    LateInit(const T &t) : unset(false), val(t) {}

    operator T() const
    {
        return val;
    }

    LateInit &
    operator=(const T &v)
    {
        unset = false;
        val = v;
        return *this;
    }

    bool
    operator==(const T &v)
    {
        if (unset)
        {
            unset = false;
            val = v;
        }
        return val == v;
    }
};

struct SReg
{
    int32_t id;
    LateInit<bool> isFloat;
    Inst2 *inst2;

    SReg(int32_t id, bool isFloat, Inst2 *inst2) noexcept;

    SReg(const SReg &) = delete;
    SReg &operator=(const SReg &) noexcept;

    bool moved = false;
    SReg(SReg &&sReg) noexcept;

    SReg &operator=(SReg &&sReg) noexcept;

    ~SReg();

    void operator=(const VRegPx &vRegPx);

    void operator=(int32_t value);

    void operator=(uint32_t value);

    void operator=(float value);

    void operator+=(int32_t value);

    void operator+=(uint32_t value);

    void operator+=(float value);

    void operator+=(const SReg &value);

    void operator-=(int32_t value);

    void operator-=(uint32_t value);

    void operator-=(float value);

    void operator-=(const SReg &value);

    SReg operator+(int32_t value) const;

    SReg operator+(uint32_t value) const;

    SReg operator+(float value) const;

    SReg operator+(const SReg &value) const;

    SReg operator-(int32_t value) const;

    SReg operator-(uint32_t value) const;

    SReg operator-(float value) const;

    SReg operator-(const SReg &value) const;

    SReg operator*(int32_t value) const;

    SReg operator*(uint32_t value) const;

    SReg operator*(float value) const;

    SReg operator*(const SReg &value) const;

    void operator*=(int32_t value) const;

    void operator*=(uint32_t value) const;

    void operator*=(float value) const;

    void operator*=(const SReg &value) const;

    void operator>>=(uint32_t shr);

    SReg operator>>(uint32_t shr) const;

    void operator>>=(const SReg &shr);

    SReg operator>>(const SReg &shr) const;

    void operator<<=(uint32_t shl);

    SReg operator<<(uint32_t shl) const;

    void operator<<=(const SReg &shl);

    SReg operator<<(const SReg &shl) const;

    SReg operator&(const SReg &value) const;

    SReg operator&(uint32_t value) const;

    void operator&=(const SReg &value) const;

    void operator&=(uint32_t value) const;

    void operator|=(const SReg &value) const;

    void operator|=(uint32_t value) const;
};

struct Range
{
    uint32_t startIdx, endIdx;

    Range(uint32_t start, uint32_t end) noexcept;
};

inline Range
OffLen(uint32_t off, uint32_t len)
{
    return {off, off + len};
}

inline Range
KthBlock(uint32_t k, uint32_t blockSize)
{
    return {k * blockSize, (k + 1) * blockSize};
}

// !!USING SEGMENT ADDR
struct OffsetLenDyn
{
    uint32_t offsetSRegId;
    uint32_t len;

    Inst2 *inst2;

    // !!USING SEGMENT ADDR
    OffsetLenDyn(const SReg &sReg, uint32_t len) noexcept;
};

struct VReg
{
    int32_t id;
    LateInit<bool> isFloat;

    Inst2 *inst2;

    VReg(int32_t id, bool isFloat, Inst2 *inst2) noexcept;

    VReg(const VReg &) = delete;
    VReg &operator=(const VReg &vReg) noexcept;

    bool moved = false;
    VReg(VReg &&vReg) noexcept;

    VReg &operator=(VReg &&vReg) noexcept;

    ~VReg();

    VRegPx operator[](uint32_t idx) const;

    VRegSlice operator[](Range range) const;

    VRegSliceDyn operator[](OffsetLenDyn range) const;

    void operator=(const VMem &vMem);

    void operator=(const VMemDyn &vMem);

    void operator=(int32_t value);

    void operator=(uint32_t value);

    void operator=(float value);

    VMask operator<(const VReg &vReg);

    VMask operator>=(const VReg &vReg);

    void operator+=(int32_t value);

    void operator+=(uint32_t value);

    void operator+=(float value);

    void operator+=(const VReg &value);

    void operator&=(uint32_t value);
};

struct VRegSlice
{
    uint32_t id;
    uint32_t startIdx;
    uint32_t endIdx;
    LateInit<bool> isFloat;

    Inst2 *inst2;

    VRegSlice(uint32_t id,
              uint32_t startIdx,
              uint32_t endIdx,
              bool isFloat,
              Inst2 *inst2) noexcept;

    void operator=(const VRegSlice &vRegSlice);

    void operator=(const VMem &vMem);

    VRegPx operator[](uint32_t idx) const;

    VRegSlice operator[](Range range) const;
};

// !!USING SEGMENT ADDR
struct VRegSliceDyn
{
    uint32_t id;
    uint32_t startIdxSRegId;
    uint32_t len;

    LateInit<bool> isFloat;

    Inst2 *inst2;

    VRegSliceDyn(uint32_t id,
                 uint32_t startIdxSRegId,
                 uint32_t len,
                 bool isFloat,
                 Inst2 *inst2) noexcept;
};

struct VRegPx
{
    int32_t id;
    uint32_t idx;

    LateInit<bool> isFloat;

    Inst2 *inst2;

    VRegPx(int32_t id, uint32_t idx, bool isFloat, Inst2 *inst2) noexcept;
};

struct VMem
{
    uint32_t startAddr;
    uint32_t len;

    LateInit<bool> isFloat;

    Inst2 *inst2;

    VMem(uint32_t startAddr, uint32_t len, bool isFloat, Inst2 *inst) noexcept;

    VMem(const VMem &) = delete;

    VMem &operator=(const VMem &vMem) noexcept;

    bool owned = true;
    VMem(VMem &&vMem) noexcept;

    VMem &operator=(VMem &&vMem) noexcept;

    ~VMem();

    VMem operator[](Range range) const;

    VMemDyn operator[](OffsetLenDyn range) const;

    void operator=(const VReg &vReg);

    void operator=(const VRegSlice &vRegSlice);

    void operator=(int32_t value);

    void operator=(uint32_t value);

    void operator=(float value);

    void operator+=(const VMem &vmem);

    void operator*=(float value);

    void
    BinaryOpSelfAssign(const VMem &vmem,
                       std::function<void(VReg &ina, VReg &inb, VReg &out)> op);

    void
    BinaryOp(const VMem &vmem,
             const VMem &out,
             std::function<void(VReg &ina, VReg &inb, VReg &out)> op) const;
};

// !!USING SEGMENT ADDR
struct VMemDyn
{
    uint32_t baseAddr;
    uint32_t addrSRegId;
    uint32_t len;

    LateInit<bool> isFloat;

    Inst2 *inst2;

    VMemDyn(uint32_t baseAddr,
            uint32_t addrSRegId,
            uint32_t len,
            bool isFloat,
            Inst2 *inst2) noexcept;

    void operator=(const VReg &vReg);
};

struct VMask
{
    int32_t id;
    Inst2 *inst2;

    VMask(int32_t id, Inst2 *inst2) noexcept;

    VMask(const VMask &) = delete;
    VMask &operator=(const VMask &) = delete;

    bool moved = false;

    VMask(VMask &&vMask) noexcept;

    VMask &operator=(VMask &&vMask) noexcept;

    ~VMask();

    void operator&=(const VMask &vMask);
};

struct SpyInfo
{
    std::string name;
    uint32_t addr;
    uint32_t len;
    std::string compare_file;
};

std::vector<std::vector<Instruction *>>
InstructionsSpilt(const std::vector<Instruction *> &instruction_lists,
                  int threshold);

std::vector<std::vector<Instruction *>>
InstructionsSpilt(const std::vector<Instruction *> &instruction_lists,
                  int threshold,
                  const std::multimap<Instruction *, SpyInfo> &spiltHint);

#    ifdef INST_ALLOC_FOUCE_USAGE
#        define __ALLOC_USAGE
#    else
#        define __ALLOC_USAGE = ""
#    endif

struct Inst2
{
    Inst inst;

    Inst2() = default;

    explicit Inst2(std::vector<Instruction *> &instlist) : inst(instlist) {}

    operator std::vector<Instruction *> &()
    {
        return inst.insts;
    }

    DebugLevel logLevel = DebugLevel::Warn;

#    ifndef NDEBUG
    bool discardInst = false;
#    endif

    uint32_t curUsingVReg = 0;
    uint32_t maxUsingVReg = 0;

    uint32_t bundleInstSize = 14000;
    bool useScheduler = false;
    bool showSchedulerVerboseInfo = false;
    std::vector<std::pair<size_t, size_t>> schedulerEffectLog = {};

    std::unordered_map<int, std::pair<std::string, bool>> vRegSymbolTable;
    Resource resource;

    std::vector<Resource> resourcesStash;

    void PushResource();

    void ForkAndPushResource();

    void PopResource();

    VReg AllocVReg(const std::string &hint);

    VReg AllocVReg(const std::string &name, bool isFloat);

    SReg AllocSReg();

    VMask AllocVMask();

    void FreeVReg(VReg *vReg);

    void FreeSReg(SReg *sReg);

    void FreeVMask(VMask *vMask);

    VMem Alloc(uint32_t size, const std::string &usage __ALLOC_USAGE);

    VMem AllocF(uint32_t size, const std::string &usage __ALLOC_USAGE);

    VMem Alloc(uint32_t size,
               uint32_t align,
               const std::string &usage __ALLOC_USAGE);

    VMem AllocF(uint32_t size,
                uint32_t align,
                const std::string &usage __ALLOC_USAGE);

    void FreeVMem(VMem *vmem);

    VReg GetSelectPxMaskT();

    void operator()(std::function<void(Instruction *)> op);

    void operator()(std::function<void(Instruction *, int)> op, int p0);

    void
    operator()(std::function<void(Instruction *, int, int)> op, int p0, int p1);

    void operator()(std::function<void(Instruction *, int, int, int)> op,
                    int p0,
                    int p1,
                    int p2);

    void operator()(std::function<void(Instruction *, int, int, int, int)> op,
                    int p0,
                    int p1,
                    int p2,
                    int p3);

    void
    operator()(std::function<void(Instruction *, int, int, int, int, int)> op,
               int p0,
               int p1,
               int p2,
               int p3,
               int p4);

    void operator()(
        std::function<void(Instruction *, int, int, int, int, int, int)> op,
        int p0,
        int p1,
        int p2,
        int p3,
        int p4,
        int p5);

    void operator()(
        std::function<void(Instruction *, int, int, int, int, int, int, int)>
            op,
        int p0,
        int p1,
        int p2,
        int p3,
        int p4,
        int p5,
        int p6);

    void operator()(std::function<void(Inst &)> op);

    void operator()(std::function<void(Inst &, int)> op, int p0);

    void operator()(std::function<void(Inst &, int, int)> op, int p0, int p1);

    void operator()(std::function<void(Inst &, int, int, int)> op,
                    int p0,
                    int p1,
                    int p2);

    void operator()(std::function<void(Inst &, int, int, int, int)> op,
                    int p0,
                    int p1,
                    int p2,
                    int p3);

    void operator()(std::function<void(Inst &, int, int, int, int, int)> op,
                    int p0,
                    int p1,
                    int p2,
                    int p3,
                    int p4);

    void
    operator()(std::function<void(Inst &, int, int, int, int, int, int)> op,
               int p0,
               int p1,
               int p2,
               int p3,
               int p4,
               int p5);

    void operator()(
        std::function<void(Inst &, int, int, int, int, int, int, int)> op,
        int p0,
        int p1,
        int p2,
        int p3,
        int p4,
        int p5,
        int p6);

    // Non Pause at Break, Mock and Spy if diff less 5%
    bool nonStop = false;

    // Non Pause at Break, Mock and Spy even diff greater 5%
    bool nonHalt = false;

    // Do Nothing at Break, Mock and Spy
    bool fastSkip = false;

    bool noMock = false;

    virtual void Spy(std::string name, const VMem &vmem);

    virtual void Break(std::string name);

    std::multimap<Instruction *, SpyInfo> spies;

    std::function<bool(std::vector<Instruction *> &,
                       const std::vector<SpyInfo> &)>
        execFunc = nullptr;

    virtual void Spy(std::string name,
                     const VMem &vmem,
                     std::string filepath,
                     bool forcePrint = false);

    virtual void Exec();

    virtual void Mock(std::string name, const VMem &vmem, std::string filepath);

    template <class... Arg>
    void
    __DISCARD__(Arg &&...args)
    {
        inst(Fence);
        inst(Jmp, 0, 0);
    }

    bool __DISCARD_BOOL_VAL__ = false;
};

struct Perm
{
};

template <class T>
void
GC(T &&t)
{
    auto val = std::move(t);
}

void Memcopy(const VMem &vmsrc, VMem &&vmdest);

void Memcopy(const VMem &vmsrc, VMem &vmdest);

enum class DMA_TYPES : uint8_t
{
    LOCAL_DMA,
    CHIP_TO_HOST
};

enum class DMA_CORE_IDS : uint8_t
{
    RESERVED,
    FXC_HBM,
    XYS0,
    XYS1,
    DHB0,
    DHB1,
    DHB2,
    DHB3,
};

enum class DMA_MEM_IDS : uint8_t
{
    HBM = 0,
    XYS_VMEM = 0,
    DHB_DMEM = 0,
    RESERVED = 1,
    XYS_SMEM = 1,
    DHB_SMEM = 1,
    FXC = 2,
    XYS_IMEM = 2,
    DHB_IMEM = 2,
    DHB_VMEM = 3,
};

enum class DMA_DEST_OPCODES : uint8_t
{
    WRITE,
    RESERVED
};

enum class DMA_SRC_OPCODES : uint8_t
{
    READ,
    RESERVED,
    MEMSET
};

union DmaMisc
{
    struct
    {
        uint8_t srcOpCode : 2;
        uint8_t destOpCode : 2;
        uint8_t srcMemId : 2;
        uint8_t destMemId : 2;
        uint8_t srcCoreId : 3;
        uint8_t destCoreId : 3;
        uint8_t traceEnable : 1;
        uint8_t dmaType : 1;
    };
    uint16_t misc;
};

struct DMA_DEST
{
    static const uint16_t HBM;
    static const uint16_t FXC;
    template <uint8_t ID>
    struct XYS
    {
        static const uint16_t VMEM =
            0b0001000000000000 + ID * 0b0000100000000000;
        static const uint16_t SMEM =
            0b0001000001000000 + ID * 0b0000100000000000;
        static const uint16_t IMEM =
            0b0001000010000000 + ID * 0b0000100000000000;
    };
    template <uint8_t ID>
    struct DHB
    {
        static const uint16_t DMEM =
            0b0010000000000000 + ID * 0b0000100000000000;
        static const uint16_t SMEM =
            0b0010000001000000 + ID * 0b0000100000000000;
        static const uint16_t IMEM =
            0b0010000010000000 + ID * 0b0000100000000000;
        static const uint16_t VMEM =
            0b0010000011000000 + ID * 0b0000100000000000;
    };

    template <uint8_t ID>
    struct OUTFEED
    {
        static const uint16_t MISC = 0b1000000000000000 + ID * 0b00001000000000;
    };
};

struct DMA_SRC
{
    static const uint16_t HBM;
    static const uint16_t FXC;
    template <uint8_t ID>
    struct XYS
    {
        static const uint16_t VMEM =
            0b0000001000000000 + ID * 0b0000000100000000;
        static const uint16_t SMEM =
            0b0000001000010000 + ID * 0b0000000100000000;
        static const uint16_t IMEM =
            0b0000001000100000 + ID * 0b0000000100000000;
    };
    template <uint8_t ID>
    struct DHB
    {
        static const uint16_t DMEM =
            0b0000010000000000 + ID * 0b0000000100000000;
        static const uint16_t SMEM =
            0b0000010000010000 + ID * 0b0000000100000000;
        static const uint16_t IMEM =
            0b0000010000100000 + ID * 0b0000000100000000;
        static const uint16_t VMEM =
            0b0000010000110000 + ID * 0b0000000100000000;
    };
};

std::vector<Instruction *> schedule(const std::vector<Instruction *> &bundle,
                                    bool coutInfo = false);


namespace InstructionExports
{
std::string NormalCppConstruction(Instruction *inst);
}

namespace Visualization 
{
std::string generateDotGraph(const InstScheduler::DAG& g, const std::string& graphName);
void generateDotFile(const std::string& dotGraph, const std::string& fileName);
}

#endif
