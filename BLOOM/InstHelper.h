#pragma once

#ifndef _INST_HELPER_H_
#    define _INST_HELPER_H_

namespace FuncHelperFeatureTest
{
constexpr int INST_HELPER_VERSION = 221122;
constexpr bool TEMPLATE_PACK_EXPAND = true;
} // namespace FuncHelperFeatureTest

#    include "instruction/Instruction.h"
#    include <array>

#    include <cassert>
#    include <functional>
#    include <map>
#    include <unordered_map>
#    include <utility>

constexpr int CONST_U32_0 = 46;
constexpr int CONST_F32_0 = 46;
constexpr int CONST_U32_1 = 48;
constexpr int CONST_F32_1 = 49;
constexpr int CONST_U32_NEG_1 = 56;
constexpr int IMME0 = 32;
constexpr int IMME1 = 33;
constexpr int IMME2 = 34;
constexpr int IMME3 = 35;
constexpr int IMME1_IMME0 = 44;
constexpr int IMME3_IMME2 = 45;
const std::unordered_map<int, uint32_t> CONST_IMME_TO_VAL = {
    {CONST_U32_0, 0x00000000},
    {CONST_F32_0, 0x00000000},
    {CONST_U32_1, 0x00000001},
    {CONST_F32_1, 0xbf800000},
    {CONST_U32_NEG_1, 0xffffffff},
};

struct DeviceConfig
{
    uint32_t VMemSize = kVectorDataMemorySize;
    uint32_t HBMSize = kHbmDataMemorySize;
    uint32_t SMemSize = kScalarDataMemorySize;
};

inline DeviceConfig &
GlobalDeviceConfig()
{
    static DeviceConfig config;
    return config;
}

inline std::pair<uint16_t, uint16_t>
HelperGetAddress(uint32_t address)
{
    uint16_t upper = (address & 0xffff0000) >> 16;
    uint16_t lower = address & 0xffff;
    return std::make_pair(upper, lower);
}

inline std::pair<uint16_t, uint16_t>
HelperGetValue(int value)
{
    uint16_t upper = (value & 0xffff0000) >> 16;
    uint16_t lower = value & 0xffff;
    return std::make_pair(upper, lower);
}

inline std::pair<uint16_t, uint16_t>
HelperGetFloatingBits(float number)
{
    int32_t init_val = *reinterpret_cast<int32_t *>(&number);
    uint16_t upper = (init_val & 0xffff0000) >> 16;
    uint16_t lower = init_val & 0xffff;
    return std::make_pair(upper, lower);
}

inline int
floatAsInt(float f)
{
    // return f;
    return *reinterpret_cast<int *>(&f);
}

inline float
intAsFloat(int i)
{
    // return i;
    return *reinterpret_cast<float *>(&i);
}

inline int
LoadUImme(Instruction *inst, uint32_t Value)
{
    if (Value >= (1 << 16))
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(Value).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(Value).first);
        return IMME1_IMME0;
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE0, Value);
        return IMME0;
    }
}

inline int
LoadUImme2(Instruction *inst, uint32_t Value)
{
    if (Value >= (1 << 16))
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2,
                                HelperGetAddress(Value).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE3,
                                HelperGetAddress(Value).first);
        return IMME3_IMME2;
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, Value);
        return IMME2;
    }
}

inline int
LoadSImme(Instruction *inst, int32_t Value)
{
    if (Value >= (1 << 16))
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetValue(Value).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetValue(Value).first);
        return IMME1_IMME0;
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE0, Value);
        return IMME0;
    }
}

inline int
LoadSImme2(Instruction *inst, int32_t Value)
{
    if (Value >= (1 << 16))
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2,
                                HelperGetValue(Value).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE3,
                                HelperGetValue(Value).first);
        return IMME3_IMME2;
    }
    inst->SetImmediateValue(Instruction::IMMEDIATE2, Value);
    return IMME2;
}

inline int
LoadFImme(Instruction *inst, float Value)
{
    inst->SetImmediateValue(Instruction::IMMEDIATE0,
                            HelperGetFloatingBits(Value).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE1,
                            HelperGetFloatingBits(Value).first);
    return IMME1_IMME0;
}

inline int
LoadFImme2(Instruction *inst, float Value)
{
    inst->SetImmediateValue(Instruction::IMMEDIATE2,
                            HelperGetFloatingBits(Value).second);
    inst->SetImmediateValue(Instruction::IMMEDIATE3,
                            HelperGetFloatingBits(Value).first);
    return IMME3_IMME2;
}

inline void
Halt(Instruction *inst)
{
    ScalarOperationState halt(S_HALT, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::SCALARONE, &halt);
}

// V_U32_MOVE
inline void
VMov(Instruction *inst, int RegSrc, int VRegDest)
{
    assert(VRegDest < 32);
    VectorOperationState set(V_U32_MOVE, 0, 0, RegSrc, VRegDest);
    inst->SetOperationState(Instruction::VECTORONE, &set);
}

// S_U32_MOVE
inline void
SMov(Instruction *inst, int RegSrc, int SRegDest)
{
    assert(SRegDest < 32);
    ScalarOperationState set(S_U32_MOVE, 0, 0, RegSrc, SRegDest);
    inst->SetOperationState(Instruction::SCALARONE, &set);
}

// S_U32_MOVE
inline void
SMovImme(Instruction *inst, uint32_t Value, int SRegIdx)
{
    assert(SRegIdx < 32);
    auto Imme = LoadUImme(inst, Value);
    ScalarOperationState set(S_U32_MOVE, 0, 0, Imme, SRegIdx);
    inst->SetOperationState(Instruction::SCALARONE, &set);
}

// S_U32_MOVE
inline void
SMovImme2(Instruction *inst,
          uint32_t Value1,
          int SRegIdx1,
          uint32_t Value2,
          int SRegIdx2)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 32);
    SMovImme(inst, Value1, SRegIdx1);
    auto Imme2 = LoadUImme2(inst, Value2);
    ScalarOperationState set(S_U32_MOVE, 0, 0, Imme2, SRegIdx2);
    inst->SetOperationState(Instruction::SCALARTWO, &set);
}

// S_U32_MULTIPLICATION
inline void
SMulU(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState mul(S_U32_MULTIPLICATION,
                             0,
                             SRegIdx1,
                             SRegIdx2,
                             SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &mul);
}

// S_U32_MULTIPLICATION
inline void
SMulF(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState mul(S_F32_MULTIPLICATION,
                             0,
                             SRegIdx1,
                             SRegIdx2,
                             SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &mul);
}

// S_U32_MULTIPLICATION
inline void
SMulUImme(Instruction *inst, int SRegIdx1, uint32_t Value, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegOut < 32);
    auto Imme = LoadUImme(inst, Value);
    ScalarOperationState mul(S_U32_MULTIPLICATION, 0, SRegIdx1, Imme, SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &mul);
}

// S_U32_AND
inline void
SAndU(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState uand(S_U32_AND, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &uand);
}

// S_U32_OR
inline void
SOrU(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState uor(S_U32_OR, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &uor);
}

// S_S32_ADDITION
inline void
SAddS(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState add(S_S32_ADDITION, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &add);
}

// S_F32_ADDITION
inline void
SAddF(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState add(S_F32_ADDITION, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARTWO, &add);
}

// S_S32_ADDITION
inline void
SAddSImme(Instruction *inst, int SRegIdx1, uint32_t Value, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegOut < 32);
    auto Imme = LoadSImme(inst, Value);
    ScalarOperationState add(S_S32_ADDITION, 0, SRegIdx1, Imme, SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &add);
}

// S_S32_SUBTRACTION
inline void
SSubS(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState sub(S_S32_SUBTRACTION, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &sub);
}

// S_F32_SUBTRACTION
inline void
SSubF(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState sub(S_F32_SUBTRACTION, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARTWO, &sub);
}

// S_S32_SUBTRACTION
inline void
SSubSImme(Instruction *inst, int SRegIdx1, uint32_t Value, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegOut < 32);
    auto Imme = LoadSImme(inst, Value);
    ScalarOperationState sub(S_S32_SUBTRACTION, 0, SRegIdx1, Imme, SRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &sub);
}

// S_DMA
inline void
SDma(Instruction *inst, int SRegDesc)
{
    assert(SRegDesc < 64);

    ScalarOperationState store(S_DMA, 0, 0, SRegDesc, 0);
    inst->SetOperationState(Instruction::SCALARTWO, &store);
}

// S_PERMISSION_OR
inline void
POr(Instruction *inst, int PermX, int PermY, int PermDest)
{
    ScalarOperationState por(S_PERMISSION_OR, 0, PermX, PermY, PermDest);
    inst->SetOperationState(Instruction::SCALARONE, &por);
}

// S_S32_LESSER
inline void
SLsS(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_S32_LESSER,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_F32_LESSER
inline void
SLsF(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_F32_LESSER,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_S32_LESSER_EQUAL
inline void
SLeS(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_S32_LESSER_EQUAL,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_S32_EQUAL
inline void
SEqS(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_S32_EQUAL,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_S32_NOTEQUAL
inline void
SNeS(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_S32_NOTEQUAL,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_S32_EQUAL
inline void
SEqSImme(Instruction *inst, int SRegIdxLeft, uint32_t Value, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(PRegOut < 32);
    auto Imme = LoadSImme(inst, Value);
    ScalarOperationState cmp(S_S32_EQUAL, 0, SRegIdxLeft, Imme, PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_S32_GREATER
inline void
SGtS(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_S32_GREATER,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_F32_GREATER
inline void
SGtF(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_F32_GREATER,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_S32_GREATEREQUAL
inline void
SGeS(Instruction *inst, int SRegIdxLeft, int SRegIdxRight, int PRegOut)
{
    assert(SRegIdxLeft < 32);
    assert(SRegIdxRight < 64);
    assert(PRegOut < 32);
    ScalarOperationState cmp(S_S32_GREATEREQUAL,
                             0,
                             SRegIdxLeft,
                             SRegIdxRight,
                             PRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &cmp);
}

// S_BRANCH
inline void
Jmp(Instruction *inst, int PRegIdx, int InstOffset)
{
    inst->SetImmediateValue(Instruction::IMMEDIATE0, InstOffset);
    ScalarOperationState jmp(S_BRANCH, PRegIdx, 0, 0, 1);
    inst->SetOperationState(Instruction::SCALARONE, &jmp);
}

// V_F32_MULTIPLICATION
inline void
VMulF(Instruction *inst, int VReg1, int VReg2, int VRegOut)
{
    VectorOperationState mul(V_F32_MULTIPLICATION, 0, VReg1, VReg2, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &mul);
}

// V_F32_LOG2, Urf
inline void
VLog2(Instruction *inst, int VReg)
{
    VectorOperationState log(V_F32_LOG2, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::VECTORONE, &log);
}

// V_RNG_GENERATE_RANDOM_NUMBER
inline void
VRNG(Instruction *inst, int DestReg) 
{
    VectorOperationState vrng(V_RNG_GENERATE_RANDOM_NUMBER, 0, 0, 0, DestReg);
    inst->SetOperationState(Instruction::VECTORTWO, &vrng);
}

// V_RNG_RESEED
inline void
VReSeed(Instruction *inst, int SeedReg) 
{
    VectorOperationState vreseed(V_RNG_RESEED, 0, SeedReg, 0, 0);
    inst->SetOperationState(Instruction::VECTORTWO, &vreseed);
}

// V_F32_SUBTRACTION
inline void
VSubF(Instruction *inst, int VReg1, int VReg2, int VRegOut)
{
    VectorOperationState sub(V_F32_SUBTRACTION, 0, VReg1, VReg2, VRegOut);
    inst->SetOperationState(Instruction::VECTORTWO, &sub);
}

// V_S32_SUBTRACTION
inline void
VSubS(Instruction *inst, int VReg1, int VReg2, int VRegOut)
{
    VectorOperationState sub(V_S32_SUBTRACTION, 0, VReg1, VReg2, VRegOut);
    inst->SetOperationState(Instruction::VECTORTWO, &sub);
}

// V_F32_ADDITION
inline void
VAddF(Instruction *inst, int VReg1, int VReg2, int VRegOut)
{
    VectorOperationState add(V_F32_ADDITION, 0, VReg1, VReg2, VRegOut);
    inst->SetOperationState(Instruction::VECTORTWO, &add);
}

// V_U32_SHIFTLEFT
inline void
VShlU(Instruction *inst, int VReg1, int VReg2, int VRegOut)
{
    VectorOperationState add(V_U32_SHIFTLEFT, 0, VReg1, VReg2, VRegOut);
    inst->SetOperationState(Instruction::VECTORTWO, &add);
}

// V_S32_ADDITION
inline void
VAddS(Instruction *inst, int VReg1, int VReg2, int VRegOut)
{
    VectorOperationState add(V_S32_ADDITION, 0, VReg1, VReg2, VRegOut);
    inst->SetOperationState(Instruction::VECTORTWO, &add);
}

// V_F32_ADDITION
inline void
VAddFImme(Instruction *inst, int VRegIn, float Value, int VRegOut)
{
    auto Imme = LoadFImme(inst, Value);
    VectorOperationState add(V_F32_ADDITION, 0, VRegIn, Imme, VRegOut);
    inst->SetOperationState(Instruction::VECTORTWO, &add);
}

// V_S32_EQUAL
inline void
VEqS(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState eq(V_S32_EQUAL, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &eq);
}

// V_S32_GREATEREQUAL
inline void
VGeS(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState gE(V_S32_GREATEREQUAL, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &gE);
}

// V_S32_GREATEREQUAL
inline void
VGeSImme(Instruction *inst, int VRegX, int Value, int VMaskOut)
{
    auto Imme = LoadSImme(inst, Value);
    VectorOperationState gE(V_S32_GREATEREQUAL, 0, VRegX, Imme, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &gE);
}

// V_F32_GREATEREQUAL
inline void
VGeF(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState gE(V_F32_GREATEREQUAL, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &gE);
}

// V_S32_GREATER
inline void
VGtS(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState ge(V_S32_GREATER, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &ge);
}

// V_F32_GREATER
inline void
VGtF(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState ge(V_F32_GREATER, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &ge);
}

// V_S32_LESSER
inline void
VLsS(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState le(V_S32_LESSER, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &le);
}

// V_S32_LESSER_EQUAL
inline void
VLeS(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState le(V_S32_LESSER_EQUAL, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &le);
}

// V_F32_LESSER
inline void
VLsF(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState le(V_F32_LESSER, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &le);
}

// V_F32_EQUAL
inline void
VEqF(Instruction *inst, int VRegX, int VRegY, int VMaskOut)
{
    VectorOperationState eq(V_F32_EQUAL, 0, VRegX, VRegY, VMaskOut);
    inst->SetOperationState(Instruction::VECTORONE, &eq);
}

// V_U32_OR
inline void
VOrU(Instruction *inst, int VRegX, int VRegY, int VRegOut)
{
    VectorOperationState uor(V_U32_OR, 0, VRegX, VRegY, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &uor);
}

// V_U32_XOR
inline void
VXorU(Instruction *inst, int VRegX, int VRegY, int VRegOut)
{
    VectorOperationState _xor(V_U32_XOR, 0, VRegX, VRegY, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &_xor);
}


// V_U32_AND
inline void
VAndU(Instruction *inst, int VRegX, int VRegY, int VRegOut)
{
    VectorOperationState uand(V_U32_AND, 0, VRegX, VRegY, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &uand);
}

inline void
Noop(Instruction *inst)
{
    ScalarOperationState scalar;
    inst->SetOperationState(Instruction::SCALARONE, &scalar);
    inst->SetOperationState(Instruction::SCALARTWO, &scalar);
    VectorOperationState vector;
    inst->SetOperationState(Instruction::VECTORONE, &vector);
    inst->SetOperationState(Instruction::VECTORTWO, &vector);
    VectorLoadOperationState vectorload;
    inst->SetOperationState(Instruction::VECTORLOAD, &vectorload);
    VectorStoreOperationState vectorstore;
    inst->SetOperationState(Instruction::VECTORSTORE, &vectorstore);
    MTIOperationState mti;
    inst->SetOperationState(Instruction::MTI, &mti);
    MTROperationState mtr;
    inst->SetOperationState(Instruction::MTR, &mtr);
    MiscOperationState misc;
    inst->SetOperationState(Instruction::MISC, &misc);
}

// V_STORE_PUSH_TO_SCALAR_CORE
inline void
VPush(Instruction *inst, int VReg)
{
    VectorStoreOperationState
        push(V_STORE_PUSH_TO_SCALAR_CORE, 0, VReg, 0, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &push);
}

// S_POP
inline void
SPop(Instruction *inst, int SRegDest)
{
    ScalarOperationState pop(S_POP, 0, 0, 0, SRegDest);
    inst->SetOperationState(Instruction::SCALARONE, &pop);
}

#    define VSel(idx)                                                          \
        inline void VSel##idx(Instruction *inst,                               \
                              int VRegX,                                       \
                              int VRegY,                                       \
                              int VRegDest)                                    \
        {                                                                      \
            VectorOperationState sel(V_SELECT_VMASK##idx,                      \
                                     0,                                        \
                                     VRegX,                                    \
                                     VRegY,                                    \
                                     VRegDest);                                \
            inst->SetOperationState(Instruction::VECTORONE, &sel);             \
        }

// V_SELECT_VMASK0
VSel(0)
    // V_SELECT_VMASK1
    VSel(1)
    // V_SELECT_VMASK2
    VSel(2)
    // V_SELECT_VMASK3
    VSel(3)
    // V_SELECT_VMASK4
    VSel(4)
    // V_SELECT_VMASK5
    VSel(5)
    // V_SELECT_VMASK6
    VSel(6)
    // V_SELECT_VMASK7
    VSel(7)

#    undef VSel

    // Auto Choose V_SELECT_VMASK, False X, True Y
    inline void VSel(Instruction *inst,
                     int VMask,
                     int VRegX,
                     int VRegY,
                     int VRegDest)
{
    switch (VMask)
    {
    case 0:
        return VSel0(inst, VRegX, VRegY, VRegDest);
    case 1:
        return VSel1(inst, VRegX, VRegY, VRegDest);
    case 2:
        return VSel2(inst, VRegX, VRegY, VRegDest);
    case 3:
        return VSel3(inst, VRegX, VRegY, VRegDest);
    case 4:
        return VSel4(inst, VRegX, VRegY, VRegDest);
    case 5:
        return VSel5(inst, VRegX, VRegY, VRegDest);
    case 6:
        return VSel6(inst, VRegX, VRegY, VRegDest);
    case 7:
        return VSel7(inst, VRegX, VRegY, VRegDest);
    default:
        assert(false && "No such VMask");
        break;
    }
}

// MTI_REDUCTION_V_MIN
inline void
VMin(Instruction *inst, int MtiX)
{
    MTIOperationState vMin(MTI_REDUCTION_V_MIN, 0, MtiX, 0, 0);
    inst->SetOperationState(Instruction::MTI, &vMin);
}

// MTI_REDUCTION_V_MAX
inline void
VMax(Instruction *inst, int MtiX)
{
    MTIOperationState vMax(MTI_REDUCTION_V_MAX, 0, MtiX, 0, 0);
    inst->SetOperationState(Instruction::MTI, &vMax);
}

// MTI_REDUCTION_V_SUM
inline void
VSum(Instruction *inst, int MtiX)
{
    MTIOperationState vSum(MTI_REDUCTION_V_SUM, 0, MtiX, 0, 0);
    inst->SetOperationState(Instruction::MTI, &vSum);
}

// MTI_REDUCTION_SEGMENTED_V_MIN
inline void
SegVMin(Instruction *inst, int MtiX)
{
    MTIOperationState vMin(MTI_REDUCTION_SEGMENTED_V_MIN, 0, MtiX, 0, 0);
    inst->SetOperationState(Instruction::MTI, &vMin);
}

// MTI_REDUCTION_PACKED_V_MIN
inline void
PackVMin(Instruction *inst, int MtiX)
{
    MTIOperationState vMin(MTI_REDUCTION_PACKED_V_MIN, 0, MtiX, 0, 0);
    inst->SetOperationState(Instruction::MTI, &vMin);
}

// MTI_REDUCTION_PACKED_SEGMENTED_V_MIN
inline void
PackSegVMin(Instruction *inst, int MtiX)
{
    MTIOperationState vMin(MTI_REDUCTION_PACKED_SEGMENTED_V_MIN, 0, MtiX, 0, 0);
    inst->SetOperationState(Instruction::MTI, &vMin);
}

// MTI_ROTATE
inline void
VRotImme(Instruction *inst, int MtiX, int Shift)
{
    inst->SetImmediateValue(Instruction::IMMEDIATE2, Shift);
    MTIOperationState rot(MTI_ROTATE, 0, MtiX, 4, 0);
    inst->SetOperationState(Instruction::MTI, &rot);
}

// MTI_ROTATE
inline void
VRot(Instruction *inst, int MtiX, int SRegShift)
{
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, SRegShift);
    MTIOperationState rot(MTI_ROTATE, 0, MtiX, 1, 0);
    inst->SetOperationState(Instruction::MTI, &rot);
}

// MTI_ROTATE
inline void
VRot1(Instruction *inst, int MtiX, int SRegShift)
{
    MTIOperationState rot(MTI_ROTATE, 0, MtiX, 0, 0);
    inst->SetOperationState(Instruction::MTI, &rot);
}

// V_SUBCORE_ROTATE DOWN
inline void
VSubRotL(Instruction *inst, int VReg, int VRegOut)
{
    VectorOperationState rot(V_SUBCORE_ROTATE, 0, VReg, 0, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &rot);
}

// V_SUBCORE_ROTATE UP
inline void
VSubRotR(Instruction *inst, int VReg, int VRegOut)
{
    VectorOperationState rot(V_SUBCORE_ROTATE, 0, VReg, 1, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &rot);
}

// MTR_READ_TRANSPOSE_RESULT
inline void
TrfOut(Instruction *inst, int MtiDest)
{
    MTROperationState mat(MTR_READ_TRANSPOSE_RESULT, 0, MtiDest, 0);
    inst->SetOperationState(Instruction::MTR, &mat);
}

// MTR_READ_TRANSPOSE_RESULT
inline void
TrfOutSelect(Instruction *inst, int MtiDest, int select)
{
    MTROperationState mat(MTR_READ_TRANSPOSE_RESULT, 0, MtiDest, select);
    inst->SetOperationState(Instruction::MTR, &mat);
}

// MTR_READ_TRANSPOSE_RESULT
inline void
UrfOut(Instruction *inst, int MtiDest)
{
    MTROperationState mat(MTR_READ_UNARY_EXECUTION_RESULT, 0, MtiDest, 0);
    inst->SetOperationState(Instruction::MTR, &mat);
}

// MTI_TRANSPOSE_START_END, with mti_width = 128
inline void
VTransStartEnd(Instruction *inst, int MtiX, int MtiWidth)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, MtiX, 0, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END, 0, MtiX, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

// MTI_TRANSPOSE_START_END, with mti_width = 128
inline void
VTransStartEndSelect(Instruction *inst, int MtiX, int MtiWidth, int select)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE_START_END,
                                    0,
                                    MtiX,
                                    0,
                                    select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE_START_END,
                                    0,
                                    MtiX,
                                    4,
                                    select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

// MTI_TRANSPOSE_START, with mti_width = 128
inline void
VTransStart(Instruction *inst, int MtiX, int MtiWidth)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE_START, 0, MtiX, 0, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE_START, 0, MtiX, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

// MTI_TRANSPOSE_START, with mti_width = 128
inline void
VTransStartSelect(Instruction *inst, int MtiX, int MtiWidth, int select)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE_START, 0, MtiX, 0, select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE_START, 0, MtiX, 4, select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

// MTI_TRANSPOSE, with mti_width = 128
inline void
VTrans(Instruction *inst, int MtiX, int MtiWidth)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE, 0, MtiX, 0, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE, 0, MtiX, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

// MTI_TRANSPOSE, with mti_width = 128
inline void
VTransSelect(Instruction *inst, int MtiX, int MtiWidth, int select)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE, 0, MtiX, 0, select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE, 0, MtiX, 4, select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

// MTI_TRANSPOSE_END, with mti_width = 128
inline void
VTransEnd(Instruction *inst, int MtiX, int MtiWidth)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE_END, 0, MtiX, 0, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE_END, 0, MtiX, 4, 0);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

// MTI_TRANSPOSE_END, with mti_width = 128
inline void
VTransEndSelect(Instruction *inst, int MtiX, int MtiWidth, int select)
{
    if (MtiWidth == 128)
    {
        MTIOperationState transpose(MTI_TRANSPOSE_END, 0, MtiX, 0, select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
    else
    {
        inst->SetImmediateValue(Instruction::IMMEDIATE2, MtiWidth - 1);
        MTIOperationState transpose(MTI_TRANSPOSE_END, 0, MtiX, 4, select);
        inst->SetOperationState(Instruction::MTI, &transpose);
    }
}

inline void
VPermute(Instruction *inst, int VReg)
{
    MTIOperationState permute(MTI_PERMUTE, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::MTI, &permute);
}

// MTI_SET_PERMUTE
inline void
SetPermute(Instruction *inst, int VReg)
{
    MTIOperationState setPermute(MTI_SET_PERMUTE, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::MTI, &setPermute);
}

// MTI_SET_PERMUTE_SUBLANES
inline void
SetPermuteSub(Instruction *inst, int VReg)
{
    MTIOperationState setPermute(MTI_SET_PERMUTE_SUBLANES, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::MTI, &setPermute);
}

// MTI_SET_PERMUTE_BYTE
inline void
SetPermuteByte(Instruction *inst, int VReg)
{
    MTIOperationState setPermute(MTI_SET_PERMUTE_BYTE, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::MTI, &setPermute);
}

// V_CONVERT_F32_TO_S32
inline void
VF2S(Instruction *inst, int VRegX, int VRegY, int VRegOut)
{
    assert(VRegX < 32);
    assert(VRegY < 64);
    assert(VRegOut < 32);
    VectorOperationState f2s(V_CONVERT_F32_TO_S32, 0, VRegX, VRegY, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &f2s);
}

// S_CONVERT_F32_TO_S32
inline void
SF2S(Instruction *inst, int VRegX, int VRegY, int VRegOut)
{
    assert(VRegX < 32);
    assert(VRegY < 64);
    assert(VRegOut < 32);
    ScalarOperationState f2s(S_CONVERT_F32_TO_S32, 0, VRegX, VRegY, VRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &f2s);
}

// V_CONVERT_S32_TO_F32
inline void
VS2F(Instruction *inst, int VRegIn, int VRegOut)
{
    assert(VRegIn < 64);
    assert(VRegOut < 32);
    VectorOperationState s2f(V_CONVERT_S32_TO_F32, 0, 0, VRegIn, VRegOut);
    inst->SetOperationState(Instruction::VECTORONE, &s2f);
}

// S_CONVERT_S32_TO_F32
inline void
SS2F(Instruction *inst, int VRegIn, int VRegOut)
{
    assert(VRegIn < 64);
    assert(VRegOut < 32);
    ScalarOperationState s2f(S_CONVERT_S32_TO_F32, 0, 0, VRegIn, VRegOut);
    inst->SetOperationState(Instruction::SCALARONE, &s2f);
}

inline void
Fence(Instruction *inst)
{
    ScalarOperationState fence(S_FENCE, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::SCALARONE, &fence);
}

inline void
Delay(Instruction *inst, uint32_t cycle)
{
    ScalarOperationState fence(S_DELAY, 0, 0, cycle >> 5, cycle & 0b11111);
    inst->SetOperationState(Instruction::SCALARONE, &fence);
}

// S_U32_SHIFTRIGHT
inline void
SShrU(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState shr(S_U32_SHIFTRIGHT, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARTWO, &shr);
}

// S_U32_SHIFTLEFT
inline void
SShlU(Instruction *inst, int SRegIdx1, int SRegIdx2, int SRegOut)
{
    assert(SRegIdx1 < 32);
    assert(SRegIdx2 < 64);
    assert(SRegOut < 32);
    ScalarOperationState shr(S_U32_SHIFTLEFT, 0, SRegIdx1, SRegIdx2, SRegOut);
    inst->SetOperationState(Instruction::SCALARTWO, &shr);
}

// MISC_VMASK_OPERATION
inline void
MVMaskAnd(Instruction *inst, int VMaskA, int VMaskTarget)
{
    assert(VMaskA < 8);
    assert(VMaskTarget < 8);
    MiscOperationState vand(MISC_VMASK_OPERATION, 0, VMaskA, 2, VMaskTarget);
    inst->SetOperationState(Instruction::MISC, &vand);
}

// V_GET_V_CORE_ID
inline void
VCoreId(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    VectorOperationState cid(V_GET_V_CORE_ID, 0, 0, 0, VReg);
    inst->SetOperationState(Instruction::VECTORONE, &cid);
}

inline void
MClrUrf(Instruction *inst)
{
    MiscOperationState clr(MISC_CLEAR_RESULT_FIFO, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::MISC, &clr);
}

inline void
MClrTrf0(Instruction *inst)
{
    MiscOperationState clr(MISC_CLEAR_RESULT_FIFO, 0, 3, 0, 0);
    inst->SetOperationState(Instruction::MISC, &clr);
}

inline void
MClrTrf1(Instruction *inst)
{
    MiscOperationState clr(MISC_CLEAR_RESULT_FIFO, 0, 4, 0, 0);
    inst->SetOperationState(Instruction::MISC, &clr);
}

enum class MSyncOp
{
    Eq = 0,
    NotEq = 1,
    Gt = 2,
    Ge = 3,
    Ls = 4,
    SetDone = 5,
    ClrDone = 6
};

// MISC_SYNC
inline void
MSync(Instruction *inst, int wait, int method, int meet)
{
    inst->SetImmediateValue(Instruction::IMMEDIATE2, meet);
    inst->SetImmediateValue(Instruction::IMMEDIATE3, wait);
    MiscOperationState sync(MISC_SYNC, 0, 4, method, 5);
    inst->SetOperationState(Instruction::MISC, &sync);
}

enum class MSetSyncOp
{
    SetDone = 1,
    ClrDone = 2
};

// MISC_SET_SYNC_FLAG
inline void
MSetSyncFlag(Instruction *inst, int wait, int method, int meet)
{
    inst->SetImmediateValue(Instruction::IMMEDIATE2, meet);
    inst->SetImmediateValue(Instruction::IMMEDIATE3, wait);
    MiscOperationState sync(MISC_SET_SYNC_FLAG, 0, 4, method, 5);
    inst->SetOperationState(Instruction::MISC, &sync);
}

#    ifdef QINGSHAN
inline void
MInt(Instruction *inst, uint32_t addr, uint32_t len)
{
    inst->SetImmediateValue(Instruction::IMMEDIATE2, addr);
    MiscOperationState sync(MISC_INTERRUPT, 0, 4, len, 0);
    inst->SetOperationState(Instruction::MISC, &sync);
}
#    endif

inline void
__CompleteInstruction(Instruction *instruction)
{
    for (unsigned int i = 0; i < Instruction::NUM_OPERATIONS_INSTRUCTION; i++)
    {
        if (instruction->GetOperationState(Instruction::OperationSequence(i)) !=
            nullptr)
            continue;
        ScalarOperationState scalar;
        VectorOperationState vector;
        VectorLoadOperationState vectorload;
        VectorStoreOperationState vectorstore;
        MTIOperationState mti;
        MTROperationState mtr;
        MiscOperationState misc;
        switch (Instruction::OperationSequence(i))
        {
        case Instruction::SCALARONE:
            instruction->SetOperationState(Instruction::SCALARONE, &scalar);
            break;
        case Instruction::SCALARTWO:
            instruction->SetOperationState(Instruction::SCALARTWO, &scalar);
            break;
        case Instruction::VECTORONE:
            instruction->SetOperationState(Instruction::VECTORONE, &vector);
            break;
        case Instruction::VECTORTWO:
            instruction->SetOperationState(Instruction::VECTORTWO, &vector);
            break;
        case Instruction::VECTORLOAD:
            instruction->SetOperationState(Instruction::VECTORLOAD,
                                           &vectorload);
            break;
        case Instruction::VECTORSTORE:
            instruction->SetOperationState(Instruction::VECTORSTORE,
                                           &vectorstore);
            break;
        case Instruction::MTI:
            instruction->SetOperationState(Instruction::MTI, &mti);
            break;
        case Instruction::MTR:
            instruction->SetOperationState(Instruction::MTR, &mtr);
            break;
        case Instruction::MISC:
            instruction->SetOperationState(Instruction::MISC, &misc);
            break;
        default:
            break;
        }
    }
    return;
}

// V_F32_EXPONENT
inline void
VExp(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    VectorOperationState exp(V_F32_EXPONENT, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::VECTORONE, &exp);
}

// V_F32_RECIPROCAL
inline void
VReciprocal(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    VectorOperationState exp(V_F32_RECIPROCAL, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::VECTORONE, &exp);
}

// V_F32_SQUAREROOT_RECIPROCAL
inline void
VSqrReciprocal(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    VectorOperationState exp(V_F32_SQUAREROOT_RECIPROCAL, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::VECTORONE, &exp);
}

// MTI_PUSHGAIN_TRANSPOSE_ROUND
inline void
PushGAINTransRound(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    MTIOperationState mti(MTI_PUSHGAIN_TRANSPOSE_ROUND, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::MTI, &mti);
}

// MTI_MUL_GSTF_ROUNDED
inline void
MulGSTFRounded(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    MTIOperationState mti(MTI_MUL_GSTF_ROUNDED, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::MTI, &mti);
}

// MTI_LOAD_GSTF
inline void
LoadGSTF(Instruction *inst)
{
    MTIOperationState mti(MTI_LOAD_GSTF, 0, 0, 0, 0);
    inst->SetOperationState(Instruction::MTI, &mti);
}

// MTI_MUL_FLOAT_ROUNDED
inline void
MulFloatRounded(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    MTIOperationState mti(MTI_MUL_FLOAT_ROUNDED, 0, VReg, 0, 0);
    inst->SetOperationState(Instruction::MTI, &mti);
}

// MTR_READ_MATRIX_RESULT
inline void
MtrOut(Instruction *inst, int VReg)
{
    assert(VReg < 32);
    MTROperationState mtr(MTR_READ_MATRIX_RESULT, 0, VReg, 0);
    inst->SetOperationState(Instruction::MTR, &mtr);
}

struct Inst
{
    std::vector<Instruction *> realinsts;
    std::vector<Instruction *> &insts;

    Inst() : insts(realinsts) {}

    Inst(std::vector<Instruction *> &linkinsts) : insts(linkinsts) {}

    int
    instCount()
    {
        return insts.size();
    }
    bool requireImme = false;
    std::pair<uint16_t, uint16_t> immeSlot;

    template <class... Arg>
    Instruction *
    Ins(std::function<void(Instruction *, typename std::decay<Arg>::type...)>
            op,
        Arg... args)
    {
        Instruction *inst = new Instruction();
        if (requireImme)
        {
            inst->SetImmediateValue(Instruction::IMMEDIATE0, immeSlot.second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, immeSlot.first);
            requireImme = false;
        }
        op(inst, args...);
        __CompleteInstruction(inst);
        insts.push_back(inst);
        return inst;
    }

    void
    operator()(std::function<void(Instruction *)> op)
    {
        Ins(std::function<void(Instruction *)>(op));
    }

    void
    operator()(std::function<void(Instruction *, int)> op, int p0)
    {
        Ins(std::function<void(Instruction *, int)>(op), p0);
    }

    void
    operator()(std::function<void(Instruction *, int, int)> op, int p0, int p1)
    {
        Ins(std::function<void(Instruction *, int, int)>(op), p0, p1);
    }

    void
    operator()(std::function<void(Instruction *, int, int, int)> op,
               int p0,
               int p1,
               int p2)
    {
        Ins(std::function<void(Instruction *, int, int, int)>(op), p0, p1, p2);
    }

    void
    operator()(std::function<void(Instruction *, int, int, int, int)> op,
               int p0,
               int p1,
               int p2,
               int p3)
    {
        Ins(std::function<void(Instruction *, int, int, int, int)>(op),
            p0,
            p1,
            p2,
            p3);
    }

    void
    operator()(std::function<void(Instruction *, int, int, int, int, int)> op,
               int p0,
               int p1,
               int p2,
               int p3,
               int p4)
    {
        Ins(std::function<void(Instruction *, int, int, int, int, int)>(op),
            p0,
            p1,
            p2,
            p3,
            p4);
    }

    void
    operator()(
        std::function<void(Instruction *, int, int, int, int, int, int)> op,
        int p0,
        int p1,
        int p2,
        int p3,
        int p4,
        int p5)
    {
        Ins(std::function<void(Instruction *, int, int, int, int, int, int)>(
                op),
            p0,
            p1,
            p2,
            p3,
            p4,
            p5);
    }

    void
    operator()(std::function<
                   void(Instruction *, int, int, int, int, int, int, int)> op,
               int p0,
               int p1,
               int p2,
               int p3,
               int p4,
               int p5,
               int p6)
    {
        Ins(std::function<
                void(Instruction *, int, int, int, int, int, int, int)>(op),
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6);
    }

    template <class... Arg>
    void
    Asm(std::function<void(Inst &, typename std::decay<Arg>::type...)> op,
        Arg... args)
    {
        op(*this, args...);
    }

    void
    If(uint32_t permit, const std::function<void()> &codeBlock)
    {
        auto jmpIn = new Instruction;
        auto jmpOut = new Instruction;
        insts.push_back(jmpIn);
        insts.push_back(jmpOut);
        auto beginPc = instCount();

        codeBlock();

        auto endPc = instCount();
        Jmp(jmpIn, permit, 1);
        __CompleteInstruction(jmpIn);
        Jmp(jmpOut, 0, endPc - beginPc);
        __CompleteInstruction(jmpOut);
    }

    void
    IfNot(uint32_t permit, const std::function<void()> &codeBlock)
    {
        auto jmpOut = new Instruction;

        insts.push_back(jmpOut);
        auto beginPc = instCount();
        codeBlock();
        auto endPc = instCount();

        Jmp(jmpOut, permit, endPc - beginPc);
        __CompleteInstruction(jmpOut);
    }

    void
    WhileDo(uint32_t permit, const std::function<void()> &codeBlock)
    {
        auto loopStart = new Instruction();
        auto loopOut = new Instruction();
        auto loopBack = new Instruction();

        insts.push_back(loopStart);
        insts.push_back(loopOut);
        auto beginPc = instCount();
        codeBlock();
        auto endPc = instCount();
        insts.push_back(loopBack);

        Jmp(loopStart, permit, 1);
        __CompleteInstruction(loopStart);
        Jmp(loopOut, 0, endPc - beginPc + 1);
        __CompleteInstruction(loopOut);
        Jmp(loopBack, 0, beginPc - endPc - 3);
        __CompleteInstruction(loopBack);
    }

    void
    WhileNotDo(uint32_t permit, const std::function<void()> &codeBlock)
    {
        auto loopOut = new Instruction();
        auto loopBack = new Instruction();

        insts.push_back(loopOut);
        auto beginPc = instCount();
        codeBlock();
        auto endPc = instCount();
        insts.push_back(loopBack);

        Jmp(loopOut, permit, endPc - beginPc + 1);
        __CompleteInstruction(loopOut);
        Jmp(loopBack, 0, beginPc - endPc - 2);
        __CompleteInstruction(loopBack);
    }

    void
    DoWhile(uint32_t permit, const std::function<void()> &codeBlock)
    {
        auto loopBack = new Instruction();

        auto beginPc = instCount();
        codeBlock();
        auto endPc = instCount();
        insts.push_back(loopBack);

        Jmp(loopBack, permit, beginPc - endPc - 1);
        __CompleteInstruction(loopBack);
    }

    int
    ImmeU(uint32_t Value)
    {
        if (Value == 0)
        {
            return CONST_U32_0;
        }
        if (Value == 1)
        {
            return CONST_U32_1;
        }
        if (Value == 0xffffffff)
        {
            return CONST_U32_NEG_1;
        }
        requireImme = true;
        immeSlot = HelperGetAddress(Value);
        return IMME1_IMME0;
    }

    int
    ImmeS(int32_t Value)
    {
        if (Value == 0)
        {
            return CONST_U32_0;
        }
        if (Value == 1)
        {
            return CONST_U32_1;
        }
        if (Value == -1)
        {
            return CONST_U32_NEG_1;
        }
        requireImme = true;
        immeSlot = HelperGetValue(Value);
        return IMME1_IMME0;
    }

    int
    ImmeF(float Value)
    {
        if (Value == 0.0f)
        {
            return CONST_F32_0;
        }
        if (Value == 1.0f)
        {
            return CONST_F32_1;
        }
        requireImme = true;
        immeSlot = HelperGetFloatingBits(Value);
        return IMME1_IMME0;
    }

    std::vector<Instruction>
    GetBundle() const
    {
        std::vector<Instruction> bundle;
        for (auto i : insts)
        {
            bundle.push_back(*i);
        }
        return bundle;
    }
};

// V_LOAD_WITH_OFFSET
inline void
VLoadBySRegWithMask(Inst &inst, int SRegAddr, int VRegDest, int ColMask)
{
    assert(SRegAddr < 64);
    assert(VRegDest < 32);
    int TmpSReg = 0;
    inst(SMov, SRegAddr, TmpSReg);
    inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);

    inst(
        [TmpSReg, ColMask, VRegDest](Instruction *inst)
        {
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                    TmpSReg);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
            VectorLoadOperationState
                vload(V_LOAD_WITH_OFFSET, 0, VRegDest, 1, 0, 6, 0, 5);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        });
}

// V_LOAD_WITH_OFFSET
inline void
VLoadBySRegWithSRegMask(Inst &inst,
                        int SRegAddr,
                        int VRegDest,
                        int ColMaskSRegId)
{
    assert(SRegAddr < 64);
    assert(VRegDest < 32);
    int TmpSReg = 0;
    inst(SMov, SRegAddr, TmpSReg);
    inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);

    inst(
        [TmpSReg, ColMaskSRegId, VRegDest](Instruction *inst)
        {
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                    TmpSReg);
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1,
                                    ColMaskSRegId);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
            VectorLoadOperationState
                vload(V_LOAD_WITH_OFFSET, 0, VRegDest, 1, 0, 6, 0, 2);
            inst->SetOperationState(Instruction::VECTORLOAD, &vload);
        });
}

// V_LOAD_WITH_OFFSET
inline void
VLoad(Instruction *inst, uint32_t addr, int VRegDest)
{
    assert(VRegDest < 32);
    int TmpSReg = 0;
    SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, TmpSReg);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
    VectorLoadOperationState
        vload(V_LOAD_WITH_OFFSET, 0, VRegDest, 1, 0, 6, 0, 0);
    inst->SetOperationState(Instruction::VECTORLOAD, &vload);
}

// V_LOAD_WITH_OFFSET
inline void
VLoadWithOffset(Instruction *inst, uint32_t addr, int16_t Offset, int VRegDest)
{
    assert(VRegDest < 32);
    int TmpSReg = 0;
    SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, TmpSReg);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, Offset);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
    VectorLoadOperationState
        vload(V_LOAD_WITH_OFFSET, 0, VRegDest, 1, 0, 6, 0, 0);
    inst->SetOperationState(Instruction::VECTORLOAD, &vload);
}

// V_LOAD_WITH_OFFSET
inline void
VLoadWithMask(Instruction *inst, uint32_t addr, int VRegDest, int ColMask)
{
    assert(VRegDest < 32);
    int TmpSReg = 0;
    SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, TmpSReg);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
    inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
    VectorLoadOperationState
        vload(V_LOAD_WITH_OFFSET, 0, VRegDest, 1, 0, 6, 0, 5);
    inst->SetOperationState(Instruction::VECTORLOAD, &vload);
}

// V_LOAD_WITH_OFFSET
inline void
VLoadBySReg(Inst &inst, int SRegAddr, int VRegDest)
{
    VLoadBySRegWithMask(inst, SRegAddr, VRegDest, 255);
}

// V_STORE_WITH_OFFSET
inline void
VStoreWithMask(Instruction *inst, int VRegSrc, uint32_t addr, int ColMask)
{
    assert(VRegSrc < 32);
    int TmpSReg = 0;
    SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, TmpSReg);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
    inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
    VectorStoreOperationState
        vstore(V_STORE_WITH_OFFSET, 0, VRegSrc, 1, 0, 6, 0, 5);
    inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
}

// V_STORE_WITH_OFFSET
inline void
VStoreBySRegWithMask(Inst &inst, int VRegSrc, int SRegAddr, int ColMask)
{
    assert(SRegAddr < 64);
    assert(VRegSrc < 32);
    int TmpSReg = 0;
    inst(SMov, SRegAddr, TmpSReg);
    inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);

    inst(
        [TmpSReg, ColMask, VRegSrc](Instruction *inst)
        {
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                    TmpSReg);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
            VectorStoreOperationState
                vstore(V_STORE_WITH_OFFSET, 0, VRegSrc, 1, 0, 6, 0, 5);
            inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        });
}

// V_STORE_WITH_OFFSET
inline void
VStoreBySRegWithSRegMask(Inst &inst, int VRegSrc, int SRegAddr, int ColMaskSReg)
{
    assert(SRegAddr < 64);
    assert(VRegSrc < 32);
    int TmpSReg = 0;
    inst(SMov, SRegAddr, TmpSReg);
    inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);

    inst(
        [TmpSReg, ColMaskSReg, VRegSrc](Instruction *inst)
        {
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                    TmpSReg);
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1,
                                    ColMaskSReg);
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
            VectorStoreOperationState
                vstore(V_STORE_WITH_OFFSET, 0, VRegSrc, 1, 0, 6, 0, 2);
            inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
        });
}

// V_STORE_WITH_OFFSET
inline void
VStore(Instruction *inst, int VRegSrc, uint32_t addr)
{
    assert(VRegSrc < 32);
    int TmpSReg = 0;
    SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, TmpSReg);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
    VectorStoreOperationState
        vstore(V_STORE_WITH_OFFSET, 0, VRegSrc, 1, 0, 6, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
}

// V_STORE_WITH_OFFSET
inline void
VStoreWithOffset(Instruction *inst, int VRegSrc, uint32_t addr, uint32_t offset)
{
    assert(VRegSrc < 32);
    int TmpSReg = 0;
    SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
    inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, TmpSReg);
    inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
    inst->SetImmediateValue(Instruction::IMMEDIATE3, offset);
    inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
    VectorStoreOperationState
        vstore(V_STORE_WITH_OFFSET, 0, VRegSrc, 1, 0, 6, 0, 0);
    inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
}

// V_STORE_WITH_OFFSET
inline void
VStoreBySReg(Inst &inst, int VRegDest, int SRegAddr)
{
    VStoreBySRegWithMask(inst, VRegDest, SRegAddr, 255);
}

#    define VStM(idx)                                                                \
        inline void VStM##idx(Instruction *inst,                                     \
                              int VRegSrc,                                           \
                              int addr)                                              \
        {                                                                            \
            assert(VRegSrc < 32);                                                    \
            int TmpSReg = 0;                                                         \
            SMovImme(inst, addr * kWordToByteDivide, TmpSReg);                       \
            inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, TmpSReg);   \
            inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);                     \
            inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);                     \
            VectorStoreOperationState                                                \
            vstore(V_STORE_WITH_VMASK##idx, 0, VRegSrc, 1, 0, 6, 0, 0);              \
            inst->SetOperationState(Instruction::VECTORSTORE, &vstore);              \
        }

    // V_STORE_WITH_VMASK0
    VStM(0)
    // V_STORE_WITH_VMASK1
    VStM(1)
    // V_STORE_WITH_VMASK2
    VStM(2)
    // V_STORE_WITH_VMASK3
    VStM(3)
    // V_STORE_WITH_VMASK4
    VStM(4)
    // V_STORE_WITH_VMASK5
    VStM(5)
    // V_STORE_WITH_VMASK6
    VStM(6)
    // V_STORE_WITH_VMASK7
    VStM(7)

#    undef VStM

    // Auto Choose V_STORE_WITH_VMASK, False X, True Y
    inline void VStM(Instruction *inst,
                     int VMask,
                     int VRegSrc,
                     int addr)
{
    switch (VMask)
    {
    case 0:
        return VStM0(inst, VRegSrc, addr);
    case 1:
        return VStM1(inst, VRegSrc, addr);
    case 2:
        return VStM2(inst, VRegSrc, addr);
    case 3:
        return VStM3(inst, VRegSrc, addr);
    case 4:
        return VStM4(inst, VRegSrc, addr);
    case 5:
        return VStM5(inst, VRegSrc, addr);
    case 6:
        return VStM6(inst, VRegSrc, addr);
    case 7:
        return VStM7(inst, VRegSrc, addr);
    default:
        assert(false && "No such VMask");
        break;
    }
}

// S_SMEM_STORE
inline void
SStore(Inst &inst, int SRegVal, int SRegAddr)
{
    assert(SRegVal < 32);
    assert(SRegAddr < 64);
    int TmpSReg = 0;
    inst(SMov, SRegAddr, TmpSReg);
    inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);

    inst(
        [TmpSReg, SRegVal](Instruction *inst)
        {
            ScalarOperationState store(S_SMEM_STORE, 0, SRegVal, TmpSReg, 0);
            inst->SetOperationState(Instruction::SCALARTWO, &store);
        });
}

// S_SMEM_LOAD
inline void
SLoad(Inst &inst, int SRegAddr, int SRegVal)
{
    assert(SRegVal < 32);
    assert(SRegAddr < 64);
    int TmpSReg = 0;
    inst(SMov, SRegAddr, TmpSReg);
    inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);

    inst(
        [TmpSReg, SRegVal](Instruction *inst)
        {
            ScalarOperationState store(S_SMEM_LOAD, 0, SRegVal, TmpSReg, 0);
            inst->SetOperationState(Instruction::SCALARTWO, &store);
        });
}

// S_SMEM_STORE
inline void
SStoreDirect(Instruction *inst, int SRegVal, int SRegAddr)
{
    assert(SRegVal < 32);
    assert(SRegAddr < 64);
    ScalarOperationState store(S_SMEM_STORE, 0, SRegVal, SRegAddr, 0);
    inst->SetOperationState(Instruction::SCALARTWO, &store);
}

// S_SMEM_LOAD
inline void
SLoadDirect(Instruction *inst, int SRegAddr, int SRegVal)
{
    assert(SRegVal < 32);
    assert(SRegAddr < 64);
    ScalarOperationState store(S_SMEM_LOAD, 0, 0, SRegAddr, SRegVal);
    inst->SetOperationState(Instruction::SCALARTWO, &store);
}

inline void
VLoadEx(Inst &inst, int VMask, int SRegAddr, int VRegDest, int ColMask)
{
    assert(SRegAddr < 64);
    assert(VRegDest < 32);
    int TmpSReg = 0;

    static const int OPs[] = {
        V_LOAD_WITH_VMASK0,
        V_LOAD_WITH_VMASK1,
        V_LOAD_WITH_VMASK2,
        V_LOAD_WITH_VMASK3,
        V_LOAD_WITH_VMASK4,
        V_LOAD_WITH_VMASK5,
        V_LOAD_WITH_VMASK6,
        V_LOAD_WITH_VMASK7,
    };

    int op = OPs[VMask];

    if (inst.requireImme)
    {
        inst.requireImme = false;
        uint32_t addr = static_cast<uint32_t>(inst.immeSlot.first << 16) +
                        (inst.immeSlot.second);
        inst(
            [TmpSReg, ColMask, op, VRegDest, addr](Instruction *inst)
            {
                SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
                inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                        TmpSReg);
                inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
                inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
                inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
                VectorLoadOperationState vload(op, 0, VRegDest, 1, 0, 6, 0, 5);
                inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            });
    }
    else if (CONST_IMME_TO_VAL.count(SRegAddr) != 0)
    {
        auto addr = CONST_IMME_TO_VAL.at(SRegAddr);
        inst(
            [TmpSReg, ColMask, op, VRegDest, addr](Instruction *inst)
            {
                SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
                inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                        TmpSReg);
                inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
                inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
                inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
                VectorLoadOperationState vload(op, 0, VRegDest, 1, 0, 6, 0, 5);
                inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            });
    }
    else
    {
        inst(SMov, SRegAddr, TmpSReg);
        inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);

        inst(
            [TmpSReg, ColMask, op, VRegDest](Instruction *inst)
            {
                inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                        TmpSReg);
                inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
                inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
                inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
                VectorLoadOperationState vload(op, 0, VRegDest, 1, 0, 6, 0, 5);
                inst->SetOperationState(Instruction::VECTORLOAD, &vload);
            });
    }
}

inline void
VStoreEx(Inst &inst, int VMask, int VRegDest, int SRegAddr, int ColMask)
{
    assert(SRegAddr < 64);
    assert(VRegDest < 32);
    int TmpSReg = 0;

    static const int OPs[] = {
        V_STORE_WITH_VMASK0,
        V_STORE_WITH_VMASK1,
        V_STORE_WITH_VMASK2,
        V_STORE_WITH_VMASK3,
        V_STORE_WITH_VMASK4,
        V_STORE_WITH_VMASK5,
        V_STORE_WITH_VMASK6,
        V_STORE_WITH_VMASK7,
    };

    int op = OPs[VMask];

    if (inst.requireImme)
    {
        inst.requireImme = false;
        uint32_t addr = static_cast<uint32_t>(inst.immeSlot.first << 16) +
                        (inst.immeSlot.second);
        inst(
            [TmpSReg, ColMask, op, VRegDest, addr](Instruction *inst)
            {
                SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
                inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                        TmpSReg);
                inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
                inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
                inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
                VectorStoreOperationState
                    vstore(op, 0, VRegDest, 1, 0, 6, 0, 5);
                inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
            });
    }
    else if (CONST_IMME_TO_VAL.count(SRegAddr) != 0)
    {
        auto addr = CONST_IMME_TO_VAL.at(SRegAddr);
        inst(
            [TmpSReg, ColMask, op, VRegDest, addr](Instruction *inst)
            {
                SMovImme(inst, addr * kWordToByteDivide, TmpSReg);
                inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                        TmpSReg);
                inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
                inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
                inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
                VectorStoreOperationState
                    vstore(op, 0, VRegDest, 1, 0, 6, 0, 5);
                inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
            });
    }
    else
    {
        inst(SMov, SRegAddr, TmpSReg);
        inst(SShlU, TmpSReg, inst.ImmeU(2), TmpSReg);
        inst(
            [TmpSReg, ColMask, op, VRegDest](Instruction *inst)
            {
                inst->SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0,
                                        TmpSReg);
                inst->SetImmediateValue(Instruction::IMMEDIATE2, 0);
                inst->SetImmediateValue(Instruction::IMMEDIATE3, ColMask);
                inst->SetImmediateValue(Instruction::IMMEDIATE4, 1);
                VectorStoreOperationState
                    vstore(op, 0, VRegDest, 1, 0, 6, 0, 5);
                inst->SetOperationState(Instruction::VECTORSTORE, &vstore);
            });
    }
}

// S_READ lcclo, lcchi
inline void
ReadLcc(Instruction *inst, int SRegLo, int SRegHi)
{
    assert(SRegLo < 32);
    assert(SRegHi < 32);

    ScalarOperationState readlo(S_READ, 0, 0, 0, SRegLo);
    ScalarOperationState readhi(S_READ, 0, 0, 1, SRegHi);

    inst->SetOperationState(Instruction::SCALARONE, &readlo);
    inst->SetOperationState(Instruction::SCALARTWO, &readhi);
}

#endif
