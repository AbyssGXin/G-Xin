
#include <numeric>
#include <string>

#define INST_ALLOC_FOUCE_USAGE
#include "FuncHelper.h"
#include "InstHelper2.h"

#ifdef _DEBUG
DebugLevel globalLogLevel = DebugLevel::Warn;
#else
DebugLevel globalLogLevel = DebugLevel::Silence;
#endif

#ifdef __linux__
#    include <pwd.h>
#    include <unistd.h>

bool
InVeloce()
{
    uid_t userid;
    struct passwd *pwd;
    userid = getuid();
    pwd = getpwuid(userid);
    std::string name = pwd->pw_name;
    return name == "jack";
}
#else
bool
InVeloce()
{
    return false;
}
#endif

bool &
IKnowItIsRunningInSimulatorNotVeloceAndISureINeedFiveTransposeHack()
{
    static bool open = false;
    return open;
}

bool &
ShowFuncCallInfo()
{
    static bool flag = false;
    return flag;
}

void
DokodemoLoad(std::vector<Instruction *> &instList,
             const std::vector<uint32_t> &reservedVReg,
             const std::vector<uint32_t> &reservedSReg,
             const std::vector<uint32_t> &reservedVMask,
             const std::vector<uint32_t> &reservedPermit,
             uint32_t vMemSrcAddr,
             uint32_t dataLength,
             uint32_t vRegDest,
             uint32_t vRegOffset)
{
    assert(vRegOffset + dataLength < 1024 && "Data Range Out Of A Single VReg");

    Inst2 inst2;
    inst2.logLevel = globalLogLevel;
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }

    VMem src(vMemSrcAddr, dataLength, false, &inst2);
    VReg dest(vRegDest, false, &inst2);
    dest[Range(vRegOffset, vRegOffset + dataLength)] = src;
    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
DokodemoStore(std::vector<Instruction *> &instList,
              const std::vector<uint32_t> &reservedVReg,
              const std::vector<uint32_t> &reservedSReg,
              const std::vector<uint32_t> &reservedVMask,
              const std::vector<uint32_t> &reservedPermit,
              uint32_t vRegSrc,
              uint32_t vRegOffset,
              uint32_t dataLength,
              uint32_t vMemDestAddr)
{
    assert(vRegOffset + dataLength < 1024 && "Data Range Out Of A Single VReg");

    Inst2 inst2;
    inst2.logLevel = globalLogLevel;
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }

    VMem dest(vMemDestAddr, dataLength, false, &inst2);
    VReg src(vRegSrc, false, &inst2);
    dest = src[Range(vRegOffset, vRegOffset + dataLength)];
    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

namespace __Utils
{
uint32_t
flp2(uint32_t x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}
} // namespace __Utils

namespace __Infra
{
void
CompleteInstruction(Instruction *instruction)
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

void
AddNoop(unsigned int number, std::vector<Instruction *> &bundle)
{
    for (int i = 0; i < number; ++i)
    {
        Instruction *inst = new Instruction();
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
        bundle.push_back(inst);
    }
}

void
MemcopyEx(Inst2 &inst2,
          const VMem &src,
          uint32_t srcOffset,
          uint32_t srcLineWidth,
          const VMem &dest,
          uint32_t destOffset,
          uint32_t destLineWidth,
          uint32_t singleWidth,
          uint32_t copyTimes)
{
    assert(srcOffset + singleWidth <= srcLineWidth &&
           destOffset + singleWidth <= destLineWidth);

    for (int i = 0; i < copyTimes; i++)
    {
        Memcopy(src[Range(i * srcLineWidth + srcOffset,
                          i * srcLineWidth + srcOffset + singleWidth)],
                dest[Range(i * destLineWidth + destOffset,
                           i * destLineWidth + destOffset + singleWidth)]);
    }
}

void
MemcopyEx128(Inst2 &inst2,
             const VMem &src,
             uint32_t srcOffset,
             uint32_t srcLineWidth,
             const VMem &dest,
             uint32_t destOffset,
             uint32_t destLineWidth,
             uint32_t singleWidth,
             uint32_t copyTimes)
{
    assert(srcLineWidth % 128 == 0 && destLineWidth % 128 == 0);
    assert(srcOffset % 128 == 0 && destOffset % 128 == 0);
    assert(srcOffset + singleWidth <= srcLineWidth &&
           destOffset + singleWidth <= destLineWidth);
    assert(src.startAddr % 128 == 0 && dest.startAddr % 128 == 0);
    assert(singleWidth % 128 == 0);

    auto idx = inst2.AllocSReg();
    auto times = inst2.AllocSReg();
    auto copyCuls = singleWidth / 128;

    idx = 0;
    times = copyTimes;

    inst2(SLsS, idx.id, times.id, 1);

    inst2.inst.WhileDo(1,
                       [&]()
                       {
                           auto readAddr = idx * (srcLineWidth / 128);
                           readAddr += (srcOffset / 128);
                           readAddr += src.startAddr / 128;

                           auto writeAddr = idx * (destLineWidth / 128);
                           writeAddr += (destOffset / 128);
                           writeAddr += dest.startAddr / 128;

                           auto val = inst2.AllocVReg("Val");

                           for (int i = 0; i < copyCuls / 8; i++)
                           {
                               inst2(VLoadBySReg, readAddr.id, val.id);
                               inst2(VStoreBySReg, val.id, writeAddr.id);
                               readAddr += 8;
                               writeAddr += 8;
                           }

                           if (copyCuls % 8 != 0)
                           {
                               inst2(VLoadBySRegWithMask,
                                     readAddr.id,
                                     val.id,
                                     (1 << (copyCuls % 8)) - 1);
                               inst2(VStoreBySRegWithMask,
                                     val.id,
                                     writeAddr.id,
                                     (1 << (copyCuls % 8)) - 1);
                           }

                           idx += 1;
                           inst2(SLsS, idx.id, times.id, 1);
                       });
}

void
HBM_TO_VMEM(std::vector<Instruction *> &instruction_list,
            uint32_t input_addr,
            uint32_t dest_addr,
            uint32_t length)
{
    const int callCnt = CallCount(__FUNCTION__);

    bool safeCall = input_addr % 128 == 0 && dest_addr % 128 == 0;
    if (ShowFuncCallInfo() || (!safeCall))
    {
        std::clog << (safeCall ? COLOR::SHIORI : COLOR::SETSUNA)
                  << "FnCall: HbmToVMem#" << callCnt << "(@" << input_addr
                  << "[" << length << "]) => @" << dest_addr << COLOR::WHITE
                  << std::endl;
    }

    assert(input_addr % 128 == 0);
    assert(dest_addr % 128 == 0);
    // assert(length % 128 == 0);

    if (length % 128 != 0)
    {
        length = ((length + 127) / 128) * 128;
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON << "LENGTH UP-ALIGN TO 128: " << length
                      << COLOR::WHITE << std::endl;
        }
    }

    int sync_register = 0;
    // total_data+=length;
    Instruction *inst;
    int misc = 0b0001000100000000;
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(input_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(input_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 6);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(dest_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(dest_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 7);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(length / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(length / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 8);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
        MiscOperationState set_sync(MISC_SET_SYNC_FLAG, 0, 0, 2, 4);
        inst->SetOperationState(Instruction::MISC, &set_sync);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    if (1)
    {
        Instruction *inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE1, 16385 + sync_register);
        ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 6, 8, 7, 33, misc);
        inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    for (int i = 0; i < 1; i++)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
        MiscOperationState sync(MISC_SYNC, 0, 0, 5, 4);
        inst->SetOperationState(Instruction::MISC, &sync);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);

        inst = new Instruction();
        ScalarOperationState fence(S_FENCE, 0, 0, 0, 0);
        inst->SetOperationState(Instruction::SCALARONE, &fence);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    AddNoop(1, instruction_list);
}

void
VMEM_TO_HBM(std::vector<Instruction *> &instruction_list,
            uint32_t input_addr,
            uint32_t dest_addr,
            uint32_t length)
{
    const int callCnt = CallCount(__FUNCTION__);

    bool safeCall = input_addr % 128 == 0 && dest_addr % 128 == 0;

    if (ShowFuncCallInfo() || (!safeCall))
    {
        std::clog << (safeCall ? COLOR::SHIORI : COLOR::SETSUNA)
                  << "FnCall: VMemToHbm#" << callCnt << "(@" << input_addr
                  << "[" << length << "]) => @" << dest_addr << COLOR::WHITE
                  << std::endl;
    }

    assert(input_addr % 128 == 0);
    assert(dest_addr % 128 == 0);
    // assert(length % 128 == 0);

    if (length % 128 != 0)
    {
        length = ((length + 127) / 128) * 128;
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON << "LENGTH UP-ALIGN TO 128: " << length
                      << COLOR::WHITE << std::endl;
        }
    }

    int sync_register = 0;
    // total_data+=length;
    int misc = 0b0000101000000000;
    Instruction *inst;
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(input_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(input_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 10);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(dest_addr / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(dest_addr / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 11);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                HelperGetAddress(length / 128).second);
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                HelperGetAddress(length / 128).first);
        ScalarOperationState set_loop_num(S_U32_MOVE, 0, 0, 44, 12);
        inst->SetOperationState(Instruction::SCALARONE, &set_loop_num);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    if (1)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
        MiscOperationState set_sync(MISC_SET_SYNC_FLAG, 0, 0, 2, 4);
        inst->SetOperationState(Instruction::MISC, &set_sync);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    if (1)
    {
        Instruction *inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE1, 16385 + sync_register);
        ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 10, 12, 11, 33, misc);
        inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }

    for (int i = 0; i < 1; i++)
    {
        inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE2, 1 + sync_register);
        MiscOperationState sync(MISC_SYNC, 0, 0, 5, 4);
        inst->SetOperationState(Instruction::MISC, &sync);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);

        inst = new Instruction();
        ScalarOperationState fence(S_FENCE, 0, 0, 0, 0);
        inst->SetOperationState(Instruction::SCALARONE, &fence);
        CompleteInstruction(inst);
        instruction_list.push_back(inst);
    }
    AddNoop(1, instruction_list);
}

} // namespace __Infra

namespace __InfraRuntime
{
void
MemcopyEx128Runtime(Inst2 &inst2,
                    const SReg &srcAddr,
                    const SReg &srcOffset,
                    const SReg &srcLineWidth,
                    const SReg &destAddr,
                    const SReg &destOffset,
                    const SReg &destLineWidth,
                    const SReg &singleWidth,
                    const SReg &copyTimes)
{
    auto idx = inst2.AllocSReg();

    idx = 0;

    inst2(SLsS, idx.id, copyTimes.id, 1);

    inst2.inst.WhileDo(1,
                       [&]()
                       {
                           auto readAddr = idx * srcLineWidth;
                           readAddr += srcOffset;
                           readAddr += srcAddr;
                           readAddr >>= 7;

                           auto writeAddr = idx * destLineWidth;
                           writeAddr += destOffset;
                           writeAddr += destAddr;
                           writeAddr >>= 7;

                           auto val = inst2.AllocVReg("Val");

                           auto copyCulIdx = inst2.AllocSReg();
                           copyCulIdx = 0;

                           auto copy8Culs = singleWidth >> 8;

                           auto copyRestCul = singleWidth & 7;
                           auto copyRestMask = inst2.AllocSReg();
                           copyRestMask = 1;
                           copyRestMask << copyRestCul;
                           copyRestMask -= 1;

                           inst2(SLsS, copyCulIdx.id, copy8Culs.id, 2);

                           inst2.inst.WhileDo(
                               2,
                               [&]()
                               {
                                   inst2(VLoadBySReg, readAddr.id, val.id);
                                   inst2(VStoreBySReg, val.id, writeAddr.id);
                                   readAddr += 8;
                                   writeAddr += 8;
                                   copyCulIdx += 1;
                                   inst2(SLsS, copyCulIdx.id, copy8Culs.id, 2);
                               });

                           inst2(SGtS, copyRestCul.id, CONST_U32_0, 2);

                           inst2.inst.If(2,
                                         [&]()
                                         {
                                             inst2(VLoadBySRegWithSRegMask,
                                                   readAddr.id,
                                                   val.id,
                                                   copyRestMask.id);
                                             inst2(VStoreBySRegWithSRegMask,
                                                   val.id,
                                                   writeAddr.id,
                                                   copyRestMask.id);
                                         });

                           idx += 1;
                           inst2(SLsS, idx.id, copyTimes.id, 1);
                       });
}

void
Hbm2VMemRuntime(Inst2 &inst2,
                const SReg &srcAddr,
                const SReg &destAddr,
                const SReg &len)
{
    SReg slen = inst2.AllocSReg();
    slen = len >> 7;
    // slen |= (1u << 31);

    SReg ssrc = inst2.AllocSReg();
    ssrc = srcAddr >> 7;
    // ssrc = srcAddr;
    // ssrc |= (1u << 31);

    SReg sdest = inst2.AllocSReg();
    sdest = destAddr >> 7; 
    // sdest = destAddr;
    // sdest |= (1u << 31);

    int sync_register = 0;
    // Instruction *inst;
    int misc = 0b0001000100000000;

    inst2(MSetSyncFlag, 1 + sync_register, (int)MSetSyncOp::ClrDone, 0);
    inst2(
        [&](Instruction *inst)
        {
            inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                    16385 + sync_register);
            ScalarOperationState dma_local_1(S_LOCAL_DMA,
                                             0,
                                             ssrc.id,
                                             slen.id,
                                             sdest.id,
                                             IMME1,
                                             misc);
            inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);
        });
    inst2(MSync, 1 + sync_register, (int)MSyncOp::SetDone, 0);
    inst2(Fence);
    inst2(Noop);
}
} // namespace __InfraRuntime

namespace __Linear
{
using __Infra::CompleteInstruction;

uint32_t
AlignTo128Bytes(uint32_t v)
{
    return ((v + 127) / 128) * 128;
}

uint32_t
AlignTo8Bytes(uint32_t v)
{
    return ((v + 7) / 8) * 8;
}

/*
           +-+
           |A|
           +-+
           |B|
+-+-+-+    +-+
|A|B|C|    |C|
+-+-+-+ => +-+
|D|E|F|    |D|
+-+-+-+    +-+
           |E|
           +-+
           |F|
           +-+

*/
void
Transfer128_128(Inst2 &inst2,
                uint32_t input_addr,
                uint32_t output_addr,
                uint32_t H, // d0 * d1 * d2
                uint32_t W) // d3
{
    auto in = VMem(input_addr, H * W, true, &inst2);
    auto out = VMem(output_addr, H * W, true, &inst2);
    for (int i = 0; i < H / 128; i++)
    {
        for (int j = 0; j < W / 128; j++)
        {
            __Infra::MemcopyEx128(
                inst2,
                in[OffLen(128 * W * i, 128 * W)],
                j * 128,
                W,
                out[OffLen(128 * 128 * (i * (W / 128) + j), 128 * 128)],
                0,
                128,
                128,
                128);
        }
    }
}

/*
           +---+
           |A^T|
           +---+
           |B^T|
+-+-+-+    +---+
|A|B|C|    |C^T|
+-+-+-+ => +---+
|D|E|F|    |D^T|
+-+-+-+    +---+
           |E^T|
           +---+
           |F^T|
           +---+

*/
void
ConvolutionLoadFilter(Inst2 &inst2,
                      uint32_t input_addr,
                      uint32_t output_addr,
                      uint32_t H, // w0
                      uint32_t W) // w1
{
    for (int i = 0; i < H / 128; i++)
    {
        Transpose(inst2.inst.insts,
                  {},
                  {},
                  {},
                  {},
                  input_addr + i * 128 * AlignTo128Bytes(W),
                  128,
                  AlignTo128Bytes(W),
                  output_addr + i * 128 * AlignTo128Bytes(W));
    }
}

void
ConvolutionBase(Inst2 &inst2,
                uint32_t inputAddr,
                uint32_t weightAddr,
                uint32_t outputAddr,
                uint32_t inputHeight, // d0 * d1 * d2
                uint32_t inputWidth,  // d3 or w0
                uint32_t weightWidth) // w1
{
    for (uint32_t inputHeightBegin = 0; inputHeightBegin < inputHeight;
         inputHeightBegin += 128)
    {
        for (uint32_t inputWidthBegin = 0; inputWidthBegin < inputWidth;
             inputWidthBegin += 128)
        {
            const bool isLastInputLoad = (inputWidthBegin + 128) >= inputWidth;
            const uint32_t inputHeightEnd = inputHeightBegin + 128;
            // constexpr 128 / 8 = 16
            const uint32_t loadInputDataCount =
                AlignTo8Bytes(inputHeightEnd - inputHeightBegin) / 8;
            for (int32_t loadInputDataIdx = loadInputDataCount - 1;
                 loadInputDataIdx >= 0;
                 loadInputDataIdx--)
            {
                const uint32_t inputDataReg = 0;
                const uint32_t inputDataAddr =
                    inputAddr + inputHeightBegin * inputWidth +
                    inputWidthBegin * 128 + loadInputDataIdx * 1024;
                inst2(VLoad, inputDataAddr / 128, inputDataReg);
                inst2(PushGAINTransRound, inputDataReg);
            }

            VReg discard = inst2.AllocVReg("");
            // MulGSTFRounded do a calc than load next data,
            // we just need load next data;
            inst2(MulGSTFRounded, discard.id);
            inst2(MtrOut, discard.id);
            for (uint32_t weightWidthIdx = 0; weightWidthIdx < weightWidth;
                 weightWidthIdx += 8)
            {
                const bool isLastWeightLoad =
                    (weightWidthIdx + 8) >= weightWidth;
                VReg weightDataReg = inst2.AllocVReg("");
                VReg outputDataReg = inst2.AllocVReg("");
                VReg OldDataReg = inst2.AllocVReg("");
                uint32_t weightDataAddr =
                    weightAddr +
                    inputWidthBegin * AlignTo128Bytes(weightWidth) +
                    weightWidthIdx * 128;

                inst2(VLoad, weightDataAddr / 128, weightDataReg.id);

                inst2(MulFloatRounded, weightDataReg.id);

                inst2(MtrOut, outputDataReg.id);
                const uint32_t outputDataAddr =
                    outputAddr +
                    inputHeightBegin * AlignTo128Bytes(weightWidth) +
                    weightWidthIdx * 128;
                if (inputWidthBegin > 0)
                {
                    inst2(VLoad, outputDataAddr / 128, OldDataReg.id);
                    inst2(VAddF,
                          outputDataReg.id,
                          OldDataReg.id,
                          outputDataReg.id);
                }
                inst2(VStore, outputDataReg.id, outputDataAddr / 128);
            }
        }
    }
}

void
ConvolutionStoreOutput(Inst2 &inst2,
                       uint32_t input_addr,
                       uint32_t output_addr,
                       uint32_t H, // d0 * d1 * d2
                       uint32_t W) // w1
{
    uint32_t new_H = (H + 127) / 128;
    uint32_t offset1 = 128 * AlignTo8Bytes(W);
    uint32_t offset2 = 128 * AlignTo128Bytes(W);
    for (int i = 0; i < new_H; i++)
    {
        Transpose(inst2.inst.insts,
                  {},
                  {},
                  {},
                  {},
                  input_addr + i * offset1,
                  W,
                  128,
                  output_addr + i * offset2);
    }
}

void
convolution2d(Inst2 &inst2,
              uint32_t input_addr,       // intput
              uint32_t filter_addr,      // weight
              uint32_t output_addr,      // output
              uint32_t from_input_addr,  // temp input
              uint32_t from_filter_addr, // temp weight
              uint32_t from_output_addr, // temp output
              uint32_t input_row,        // d0 * d1
              uint32_t input_col,        // d2
              uint32_t kernal,           // 1
              uint32_t stride,           // 1
              uint32_t chan_in,          // w0 or d3
              uint32_t chan_out)         // w1
{
    uint32_t in_ch_size = kernal * kernal * chan_in; // w0 or d3
    uint32_t in_size =
        (input_row - kernal + 1) * (input_col - kernal + 1); // d0 * d1 * d2
    Transfer128_128(inst2, input_addr, from_input_addr, in_size, in_ch_size);
    ConvolutionLoadFilter(inst2,
                          filter_addr,
                          from_filter_addr,
                          in_ch_size,
                          chan_out);
    ConvolutionBase(inst2,
                    from_input_addr,
                    from_filter_addr,
                    from_output_addr,
                    in_size,
                    in_ch_size,
                    chan_out);
    ConvolutionStoreOutput(inst2,
                           from_output_addr,
                           output_addr,
                           in_size,
                           chan_out);
}

/// <summary>
/// Basic version of linear, reserve the input and weight
/// </summary>
/// <param name="inst2"></param>
/// <param name="input"></param>
/// <param name="inputSize"></param>
/// <param name="wight"></param>
/// <param name="wightSize"></param>
/// <param name="output"></param>
void
LinearInternal(Inst2 &inst2,
               const VMem &input,
               const std::array<uint32_t, 4> &inputSize,
               const VMem &weight,
               const std::array<uint32_t, 2> &weightSize,
               const VMem &output)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    assert(w0 == d3);

    assert(w0 % 128 == 0 && w1 % 128 == 0);

    const int callCnt = CallCount(__FUNCTION__);
    if (false && ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: LinearInternal#" << callCnt
                  << "(@" << input.startAddr << "[" << d0 << ", " << d1 << ", "
                  << d2 << ", " << d3 << "], @" << weight.startAddr << "[" << w0
                  << ", " << w1 << "]) => @" << output.startAddr << COLOR::WHITE
                  << std::endl;
    }

    auto vMInput =
        inst2.AllocF(d0 * d1 * AlignTo128Bytes(d2) * d3, "Internal Temp Input");
    // auto vFInput = inst2.AllocF(2 * d0 * d1 * AlignTo128Bytes(d2) * d3);

    auto vMWeight = inst2.AllocF(w0 * w1, "Internal Temp Weight");
    // auto vFWeight = inst2.AllocF(2 * w0 * w1);

    // auto vMOutput = inst2.AllocF(d0 * d1 * d2 * w1);
    auto vFOutput = inst2.AllocF(2 * d0 * d1 * AlignTo128Bytes(d2) * w1,
                                 "Internal Temp Output");

    // vMInput[Range(0, d0 * d1 * d2 * d3)] = input;

    // _TransposeAny(inst2, weight, vMWeight, w0, w1);
    // vMWeight = weight;

    convolution2d(inst2,
                  input.startAddr,
                  weight.startAddr,
                  output.startAddr,
                  vMInput.startAddr,
                  vMWeight.startAddr,
                  vFOutput.startAddr,
                  d0 * d1,
                  AlignTo128Bytes(d2),
                  1,
                  1,
                  d3,
                  w1);

    inst2.FreeVMem(&vFOutput);
    // inst2.FreeVMem(&vFWeight);
    inst2.FreeVMem(&vMWeight);
    // inst2.FreeVMem(&vFInput);
    inst2.FreeVMem(&vMInput);
}

uint32_t
LinearInternalRequiredSizeMeasure(const std::array<uint32_t, 4> &inputSize,
                                  const std::array<uint32_t, 2> &weightSize)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    auto vMInput = d0 * d1 * AlignTo128Bytes(d2) * d3;
    // auto vFInput = 2 * d0 * d1 * AlignTo128Bytes(d2) * d3;

    auto vMWeight = w0 * w1;
    // auto vFWeight = 2 * w0 * w1;

    auto vFOutput = 2 * d0 * d1 * AlignTo128Bytes(d2) * w1;

    return vMInput + vMWeight + vFOutput;
}

/*

in this fn
                        +--+
+--+--+                 |a1|
|a1|b1| will spilt into +--+
+--+--+                 |b1|
                        +--+

          +--+   +--+   +--+    +--+
          |1A|   |1A| X |a1| => |o1|
for input +--+ , +--+   +--+    +--+
          |1B|   |1B| X |b1| => |o2|
          +--+   +--+   +--+    +--+
+--+
|o1|                +--+--+
+--+ need reform to |o1|o2|
|o2|                +--+--+
+--+

*/
/// <summary>
/// linear but spilt weight in width into little piece and do linear each
/// </summary>
/// <param name="inst2"></param>
/// <param name="splitWidth"></param>
/// <param name="input"></param>
/// <param name="inputSize"></param>
/// <param name="wight"></param>
/// <param name="wightSize"></param>
/// <param name="output"></param>
void
LinearInternalSplitWidth(Inst2 &inst2,
                         uint32_t splitWidth,
                         const VMem &input,
                         const std::array<uint32_t, 4> inputSize,
                         const VMem &weight,
                         const std::array<uint32_t, 2> weightSize,
                         const VMem &output)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    VMem splitWeight = inst2.AllocF(w0 * splitWidth, "Splited Weight");
    VMem splitOutput =
        inst2.AllocF(d0 * d1 * d2 * splitWidth, "Splited Output");

    for (int i = 0; i < w1 / splitWidth; i++)
    {
        __Infra::MemcopyEx128(inst2,
                              weight,
                              i * splitWidth,
                              w1,
                              splitWeight,
                              0,
                              splitWidth,
                              splitWidth,
                              w0);

        __Linear::LinearInternal(inst2,
                                 input,
                                 inputSize,
                                 splitWeight,
                                 {w0, splitWidth},
                                 splitOutput);

        for (int j = 0; j < d0 * d1; j++)
        {
            __Infra::MemcopyEx128(inst2,
                                  splitOutput,
                                  j * splitWidth,
                                  splitWidth * d0 * d1,
                                  output,
                                  j * w1 + i * splitWidth,
                                  w1 * d0 * d1,
                                  splitWidth,
                                  d2);
        }
    }

    inst2.FreeVMem(&splitOutput);
    inst2.FreeVMem(&splitWeight);
}

uint32_t
LinearInternalSplitWidthRequiredSizeMeasure(
    uint32_t splitWidth,
    const std::array<uint32_t, 4> inputSize,
    const std::array<uint32_t, 2> weightSize)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    auto splitWeight = w0 * splitWidth;
    auto splitOutput = d0 * d1 * d2 * splitWidth;

    return __Linear::LinearInternalRequiredSizeMeasure(inputSize,
                                                       {w0, splitWidth}) +
           splitOutput + splitWeight;
}

/*

+--+--+ < input        +--+         height-splited weight
|1A|2A|                |1A|                +--+--+
+--+--+ it will choose +--+ to matmul with |a1|b1| and sum up
|1B|2B|                |1B|                +--+--+
+--+--+                +--+ < each column
        +--+     +--+
because |1A| and |1B| is not continuous in vmem so we need full size input
        +--+     +--+
*/
/// <summary>
/// linear split weight both height and width, this need to be call height
/// pieces times for get the finally result, sum of each calc. Caution,
/// weight[0] need equal to splitHeight
/// </summary>
/// <param name="inst2"></param>
/// <param name="splitHeight"></param>
/// <param name="splitWidth"></param>
/// <param name="input"></param>
/// <param name="inputSize"></param>
/// <param name="wight"></param>
/// <param name="wightSize"></param>
/// <param name="output"></param>
/// <param name="blockCount"></param>
void
LinearInternalSplit(Inst2 &inst2,
                    uint32_t splitHeight,
                    uint32_t splitWidth,
                    const VMem &input,
                    const std::array<uint32_t, 4> inputSize,
                    const VMem &weight,
                    const std::array<uint32_t, 2> weightSize,
                    const VMem &output,
                    uint32_t blockCount)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    VMem splitInput = inst2.AllocF(d0 * d1 * d2 * w0, "Splited Input");

    for (int i = 0; i < d0 * d1; i++)
    {
        __Infra::MemcopyEx128(inst2,
                              input,
                              blockCount * splitHeight + i * d3,
                              d0 * d1 * d3,
                              splitInput,
                              i * splitHeight,
                              d0 * d1 * splitHeight,
                              splitHeight,
                              d2);
    }

    if (blockCount != 0)
    {
        VMem tmpOutput = inst2.AllocF(d0 * d1 * d2 * w1, "Temp Output");

        __Linear::LinearInternalSplitWidth(inst2,
                                           splitWidth,
                                           splitInput,
                                           {d0, d1, d2, w0},
                                           weight,
                                           weightSize,
                                           tmpOutput);

        auto copyTimes = d0 * d1 * d2 * w1 / 1024;
        auto copyRestCol = d0 * d1 * d2 * w1 % 1024;
        auto val = inst2.AllocVReg("Val");
        auto valOld = inst2.AllocVReg("ValOld");

        for (int i = 0; i < copyTimes; i++)
        {
            val = tmpOutput[Range(i * 1024, (i + 1) * 1024)];
            valOld = output[Range(i * 1024, (i + 1) * 1024)];
            val += valOld;
            output[Range(i * 1024, (i + 1) * 1024)] = val;
        }
        if (copyRestCol != 0)
        {
            val = tmpOutput[Range(copyTimes * 1024,
                                  copyTimes * 1024 + copyRestCol)];
            valOld =
                output[Range(copyTimes * 1024, copyTimes * 1024 + copyRestCol)];
            val += valOld;
            output[Range(copyTimes * 1024, copyTimes * 1024 + copyRestCol)] =
                val;
        }

        inst2.FreeVMem(&tmpOutput);
    }
    else
    {
        __Linear::LinearInternalSplitWidth(inst2,
                                           splitWidth,
                                           splitInput,
                                           {d0, d1, d2, w0},
                                           weight,
                                           weightSize,
                                           output);
    }

    inst2.FreeVMem(&splitInput);
}

uint32_t
LinearInternalSplitRequiredSizeMeasure(uint32_t splitHeight,
                                       uint32_t splitWidth,
                                       const std::array<uint32_t, 4> inputSize,
                                       const std::array<uint32_t, 2> weightSize)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    auto splitInput = d0 * d1 * d2 * w0;

    auto tmpOutput = d0 * d1 * d2 * w1;

    return __Linear::LinearInternalSplitWidthRequiredSizeMeasure(
               splitWidth,
               {d0, d1, d2, w0},
               weightSize) +
           tmpOutput + splitInput;
}

/*

+--+--+ input          +--+ each column
|a1|b1|                |?1|                +--+--+
+--+--+ it will choose +--+ to matmul with |1A|2A|
|a2|b2|                |?2|                +--+--+
+--+--+                +--+
        +--+     +--+
because |a1| and |a2| is not continuous in vmem so we need full size input
        +--+     +--+

in this fn: input has been splited width
so weight would spilt height,
in caller, spilt height in input no need split weight.
for splitting, we should sum the output up.

the internal call is __Linear::LinearInternalSplit

*/
void
LinearInternalSplit_OwO_(Inst2 &inst2,
                         uint32_t splitHeight,
                         uint32_t splitWidth,
                         const VMem &input,
                         const std::array<uint32_t, 4> inputSize,
                         const VMem &weight,
                         const std::array<uint32_t, 2> weightSize,
                         const VMem &output)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    int weightSplitWidth = w1;
    auto availableVMem = inst2.resource.BiggestContinuousAvailableVMemSize(128);
    for (int i = __Utils::flp2(w1); i >= 128; i /= 2)
    {
        auto requiredVMem =
            __Linear::LinearInternalSplitRequiredSizeMeasure(splitWidth,
                                                             i,
                                                             inputSize,
                                                             {splitWidth, w1});
        if (requiredVMem < availableVMem)
        {
            weightSplitWidth = std::min(weightSplitWidth, i);
            break;
        }
    }

    if (weightSplitWidth != w1)
    {
        std::clog << COLOR::AYUMU << "CHOOSE " << weightSplitWidth
                  << " AS WEIGHT SPLIT WIDTH" << COLOR::WHITE << "\n";
    }

    for (int i = 0; i < w0 / splitWidth; i++)
    {
        __Linear::LinearInternalSplit(
            inst2,
            splitWidth,
            weightSplitWidth,
            input,
            inputSize,
            weight[OffLen(i * splitWidth * w1, splitWidth * w1)],
            {splitWidth, w1},
            output,
            i);
    }
}

/*
input is full size;
weight in hbm

+--+--+
|a1|b1| < load each height-spilted
+--+--+
|a2|b2|
+--+--+

inner call __Linear::LinearInternalSplit

in __Linear::LinearInternalSplit:

input

+--+--+                +--+
|1A|2A|                |1A|                +--+--+
+--+--+ it will choose +--+ to matmul with |a1|b1|
|1B|2B|                |1B|                +--+--+
+--+--+                +--+
        +--+     +--+
because |1A| and |1B| is not continuous in vmem so we need full size input
        +--+     +--+
*/

void
LinearInternalSplitHBMWeight(Inst2 &inst2,
                             uint32_t splitHeight,
                             uint32_t splitWidth,
                             const VMem &input,
                             const std::array<uint32_t, 4> inputSize,
                             uint32_t hbmWeightAddr,
                             const std::array<uint32_t, 2> weightSize,
                             const VMem &output)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    VMem weight = inst2.AllocF(splitHeight * w1, "Loaded HBM Weight Data");

    for (int i = 0; i < w0 / splitHeight; i++)
    {

        __Infra::HBM_TO_VMEM(inst2.inst.insts,
                             hbmWeightAddr + i * splitHeight * w1,
                             weight.startAddr,
                             splitHeight * w1);

        __Linear::LinearInternalSplit(inst2,
                                      splitHeight,
                                      splitWidth,
                                      input,
                                      inputSize,
                                      weight,
                                      {splitHeight, w1},
                                      output,
                                      i);
    }
}

/*
input in hbm
weight in vmem full size

in this fn, input is splited height,
and we will split input weight for __Linear::LinearInternalSplit_OwO_

in this case, input is huge, we need split height for input first

+--+--+
|1A|2A| < load each height-spilted
+--+--+
|1B|2B|
+--+--+

inner call __Linear::LinearInternalSplit_OwO_

in __Linear::LinearInternalSplit_OwO_:

input

+--+--+                +--+ each column
|a1|b1|                |?1|                +--+--+
+--+--+ it will choose +--+ to matmul with |1A|2A|
|a2|b2|                |?2|                +--+--+
+--+--+                +--+
        +--+     +--+
because |a1| and |a2| is not continuous in vmem so we need full size input
        +--+     +--+

this output is continuous in mem, no need sum up

because input has batches, so it can't just do one dma as same as weight
it need dma each strip of each batch, than concat them up

*/
void
LinearInternalSplitHbmInput(Inst2 &inst2,
                            uint32_t splitHeight,
                            uint32_t splitWidth,
                            uint32_t hbmInputAddr,
                            const std::array<uint32_t, 4> inputSize,
                            const VMem &weight,
                            const std::array<uint32_t, 2> weightSize,
                            const VMem &output)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    auto inSize = d0 * d1 * d2 * d3;
    auto stripSize = splitHeight * d3;
    auto stripsOutSize = d0 * d1 * splitHeight * w1;

    VMem input = inst2.AllocF(d0 * d1 * stripSize, "Loaded HBM Input Data");

    for (int i = 0; i < d2 / splitHeight; i++)
    {
        for (int j = 0; j < d0 * d1; j++)
        {
            __Infra::HBM_TO_VMEM(inst2.inst.insts,
                                 hbmInputAddr + j * inSize + i * stripSize,
                                 input.startAddr + j * stripSize,
                                 stripSize);
        }

        __Linear::LinearInternalSplit_OwO_(
            inst2,
            splitHeight,
            splitWidth,
            input,
            {d0, d1, splitHeight, d3},
            weight,
            weightSize,
            output[OffLen(i * stripsOutSize, stripsOutSize)]);
    }
}

} // namespace __Linear

namespace __InternalImpl
{

// Assume height and width less-equal 128
void
Transpose(Inst2 &inst,
          const VMem &data,
          const VMem &output,
          uint32_t height,
          uint32_t width)
{
    static int nwsSelect = 0;
    // nwsSelect = 1 - nwsSelect;
    //  输入需要的 VReg 数
    const auto inputVRegCnt = (height + 7) / 8;
    // 输出需要的 VReg 数
    const auto outputVRegCnt = (width + 7) / 8;

    std::vector<VReg> vRegs;
    const auto size = std::max(inputVRegCnt, outputVRegCnt);
    for (int i = 0; i < size; i++)
    {
        vRegs.push_back(inst.AllocVReg("Val#" + std::to_string(i), false));
    }

    for (auto i = 0; i < height; i += 8)
    {
        if (width != 128)
        {
            for (int j = i; j < ((i + 8) < height ? (i + 8) : height); j++)
            {
                vRegs[i / 8][Range((j % 8) * 128, (j % 8) * 128 + width)] =
                    data[Range(j * width, (j + 1) * width)];
            }
        }
        else
        {
            int l = (i + 8) <= height ? 8 : (height % 8);
            vRegs[i / 8][Range(0, l * 128)] =
                data[Range(i * 128, (i + l) * 128)];
        }
    }

    if (inputVRegCnt == 1)
    {
        inst(VTransStartEndSelect, vRegs[0].id, outputVRegCnt * 8, nwsSelect);
    }
    else
    {
        inst(VTransStartSelect, vRegs[0].id, outputVRegCnt * 8, nwsSelect);
        for (auto i = 1; i < inputVRegCnt - 1; i++)
        {
            inst(VTransSelect, vRegs[i].id, outputVRegCnt * 8, nwsSelect);
        }
        if (IKnowItIsRunningInSimulatorNotVeloceAndISureINeedFiveTransposeHack())
        {
            if (InVeloce())
            {
                std::cerr << COLOR::RED
                          << "\tWARNING!\nYOU SEEMS ENABLE TRANSPOSE BUG HACK "
                             "IN VELOCE!\n"
                          << COLOR::RED;
                exit(-1);
            }
            if (inputVRegCnt > 5)
            {
                inst(VTransEndSelect,
                     vRegs[inputVRegCnt - 1].id,
                     outputVRegCnt * 8,
                     nwsSelect);
                auto const0 = inst.AllocVReg("Const0");
                const0 = 0;
                for (int j = 0; j < 4; j++)
                {
                    inst(VTransSelect, const0.id, outputVRegCnt * 8, nwsSelect);
                }
                inst(VTransEndSelect, const0.id, outputVRegCnt * 8, nwsSelect);
            }
            else
            {
                inst(VTransEndSelect,
                     vRegs[inputVRegCnt - 1].id,
                     outputVRegCnt * 8,
                     nwsSelect);
            }
        }
        else
        {
            inst(VTransEndSelect,
                 vRegs[inputVRegCnt - 1].id,
                 outputVRegCnt * 8,
                 nwsSelect);
        }
    }

    auto i = 0;
    for (; i < outputVRegCnt; i++)
    {
        inst(TrfOutSelect, vRegs[i].id, nwsSelect);
    }

    // VReg _ = inst.AllocVReg("Discard", false);
    // for (; i < 16; i++)
    //{
    //     inst(TrfOut, _.id);
    // }

    for (i = 0; i < width; i += 8)
    {
        VReg tmp = inst.AllocVReg("Tmp");
        tmp = vRegs[i / 8];
        if (height != 128)
        {
            for (int j = i; j < ((i + 8) < width ? (i + 8) : width); j++)
            {
                output[Range(j * height, (j + 1) * height)] =
                    tmp[Range((j % 8) * 128, (j % 8) * 128 + height)];
            }
        }
        else
        {
            const int l = (i + 8) <= width ? 8 : (width % 8);
            output[Range(i * 128, (i + l) * 128)] = tmp[Range(0, l * 128)];
        }
    }
}

void
TransposeWithStride(Inst2 &inst,
                    const VMem &data,
                    uint32_t offsetIn,
                    uint32_t strideIn,
                    const VMem &output,
                    uint32_t offsetOut,
                    uint32_t strideOut,
                    uint32_t height,
                    uint32_t width)
{
    static int nwsSelect = 0;
    // nwsSelect = 1 - nwsSelect;
    //  输入需要的 VReg 数
    const auto inputVRegCnt = (height + 7) / 8;
    // 输出需要的 VReg 数
    const auto outputVRegCnt = (width + 7) / 8;

    std::vector<VReg> vRegs;
    const auto size = std::max(inputVRegCnt, outputVRegCnt);
    for (int i = 0; i < size; i++)
    {
        vRegs.push_back(inst.AllocVReg("Val#" + std::to_string(i), false));
    }
    auto curInAddr = offsetIn;
    for (auto i = 0; i < height; i++)
    {
        vRegs[i / 8][Range((i % 8) * 128, (i % 8) * 128 + width)] =
            data[Range(curInAddr, curInAddr + width)];
        curInAddr += strideIn;
    }

    if (inputVRegCnt == 1)
    {
        inst(VTransStartEndSelect, vRegs[0].id, outputVRegCnt * 8, nwsSelect);
    }

    else
    {
        inst(VTransStartSelect, vRegs[0].id, outputVRegCnt * 8, nwsSelect);
        for (auto i = 1; i < inputVRegCnt - 1; i++)
        {
            inst(VTransSelect, vRegs[i].id, outputVRegCnt * 8, nwsSelect);
        }
        if (IKnowItIsRunningInSimulatorNotVeloceAndISureINeedFiveTransposeHack())
        {
            if (InVeloce())
            {
                std::cerr << COLOR::RED
                          << "\tWARNING!\nYOU SEEMS ENABLE TRANSPOSE BUG HACK "
                             "IN VELOCE!\n"
                          << COLOR::RED;
                exit(-1);
            }
            if (inputVRegCnt > 5)
            {
                inst(VTransEndSelect,
                     vRegs[inputVRegCnt - 1].id,
                     outputVRegCnt * 8,
                     nwsSelect);
                auto const0 = inst.AllocVReg("Const0");
                const0 = 0;
                for (int j = 0; j < 4; j++)
                {
                    inst(VTransSelect, const0.id, outputVRegCnt * 8, nwsSelect);
                }
                inst(VTransEndSelect, const0.id, outputVRegCnt * 8, nwsSelect);
            }
            else
            {
                inst(VTransEndSelect,
                     vRegs[inputVRegCnt - 1].id,
                     outputVRegCnt * 8,
                     nwsSelect);
            }
        }
        else
        {
            inst(VTransEndSelect,
                 vRegs[inputVRegCnt - 1].id,
                 outputVRegCnt * 8,
                 nwsSelect);
        }
    }

    auto i = 0;
    for (; i < outputVRegCnt; i++)
    {
        inst(TrfOutSelect, vRegs[i].id, nwsSelect);
    }

    // VReg _ = inst.AllocVReg("Discard");
    // for (; i < 16; i++) {
    //	inst(TrfOut, _.id);
    // }

    auto curOutAddr = offsetOut;
    for (i = 0; i < width; i += 8)
    {
        VReg tmp = inst.AllocVReg("Tmp");
        tmp = vRegs[i / 8];
        for (int j = i; j < ((i + 8) < width ? (i + 8) : width); j++)
        {
            output[Range(curOutAddr, curOutAddr + height)] =
                tmp[Range((j % 8) * 128, (j % 8) * 128 + height)];
            curOutAddr += strideOut;
        }
    }
}

// Any height and width
void
TransposeAny(Inst2 &inst,
             const VMem &data,
             const VMem &output,
             uint32_t height,
             uint32_t width)
{
    if (height <= 128 && width <= 128)
    {
        Transpose(inst, data, output, height, width);
    }
    else
    {
        if (data.startAddr == output.startAddr)
        {
            auto tmp = inst.Alloc(output.len, "TransposeTempExchange");
            for (int i = 0; i < (height + 127) / 128; i++)
            {
                for (int j = 0; j < (width + 127) / 128; j++)
                {
                    const auto curH =
                        (i + 1) * 128 <= height ? 128 : height % 128;
                    const auto curW =
                        (j + 1) * 128 <= width ? 128 : width % 128;
                    const auto inputAddr = width * i * 128 + j * 128;
                    const auto outputAddr = i * 128 + height * j * 128;
                    TransposeWithStride(inst,
                                        data,
                                        inputAddr,
                                        width,
                                        tmp,
                                        outputAddr,
                                        height,
                                        curH,
                                        curW);
                }
            }
            Memcopy(tmp,
                    VMem(output.startAddr, output.len, output.isFloat, &inst));
            inst.FreeVMem(&tmp);
        }
        else
        {
            for (int i = 0; i < (height + 127) / 128; i++)
            {
                for (int j = 0; j < (width + 127) / 128; j++)
                {
                    const auto curH =
                        (i + 1) * 128 <= height ? 128 : height % 128;
                    const auto curW =
                        (j + 1) * 128 <= width ? 128 : width % 128;
                    const auto inputAddr = width * i * 128 + j * 128;
                    const auto outputAddr = i * 128 + height * j * 128;
                    TransposeWithStride(inst,
                                        data,
                                        inputAddr,
                                        width,
                                        output,
                                        outputAddr,
                                        height,
                                        curH,
                                        curW);
                }
            }
        }
    }
}

void
Permute(Inst2 &inst,
        const VMem &data,
        const VMem &output,
        const std::vector<uint32_t> &dimSize,
        const std::vector<uint32_t> &newDimFrom)
{
    auto dimCnt = dimSize.size();
    auto size = dimSize;
    auto to = newDimFrom;
    std::vector<uint32_t> cur(dimCnt, 0);
    for (int i = 0; i < dimCnt; i++)
    {
        cur[i] = i;
    }
    bool first = true;
    for (int i = 0; i < dimCnt - 1; i++)
    {
        for (int j = i + 1; j < dimCnt; j++)
        {
            if (cur[j] == to[i])
            {
                auto ncur = cur;
                auto loopSize = 1;
                for (int k = 0; k < i; k++)
                {
                    loopSize *= size[cur[k]];
                }
                auto dim1Size = 1;
                auto dim2Size = 1;
                if (inst.logLevel >= DebugLevel::Warn)
                {
                    std::clog << "TRANSPOSE: [";
                }
                for (int k = i; k < j; k++)
                {
                    if (inst.logLevel >= DebugLevel::Warn)
                    {
                        if (k != i)
                        {
                            std::clog << ", ";
                        }
                        std::clog << k;
                    }
                    dim1Size *= size[cur[k]];
                }
                if (inst.logLevel >= DebugLevel::Warn)
                {
                    std::clog << "] X [";
                }
                for (int k = j; k < dimCnt; k++)
                {
                    if (inst.logLevel >= DebugLevel::Warn)
                    {
                        if (k != j)
                        {
                            std::clog << ", ";
                        }
                        std::clog << k;
                    }
                    dim2Size *= size[cur[k]];
                }
                if (inst.logLevel >= DebugLevel::Warn)
                {
                    std::clog << "]\n";
                }
                auto dimSize = dim1Size * dim2Size;

                for (int k = 0; k < loopSize; k++)
                {
                    if (first)
                    {
                        TransposeAny(
                            inst,
                            data[Range(k * dimSize, (k + 1) * dimSize)],
                            output[Range(k * dimSize, (k + 1) * dimSize)],
                            dim1Size,
                            dim2Size);
                    }
                    else
                    {
                        TransposeAny(
                            inst,
                            output[Range(k * dimSize, (k + 1) * dimSize)],
                            output[Range(k * dimSize, (k + 1) * dimSize)],
                            dim1Size,
                            dim2Size);
                    }
                }

                first = false;
                for (int k = i; k < dimCnt; k++)
                {
                    ncur[k] =
                        cur[((k - i + j - i + dimCnt - i) % (dimCnt - i)) + i];
                }
                cur = ncur;
                if (inst.logLevel >= DebugLevel::Warn)
                {
                    std::clog << "CUR ORDER: ";
                    bool first = true;
                    for (auto c : cur)
                    {
                        if (!first)
                        {
                            std::clog << ", ";
                        }
                        first = false;
                        std::clog << c;
                    }
                    std::clog << "\n";
                }
                break;
            }
        }
    }
}
// use MTI_REDUCTION_V_MAX to get max value and expand to all vector
void
MaxFExpand(Inst2 &inst, const VReg &vreg)
{
    // VMax 会计算所有128个核心数据的最大值，会放到转置fifo中
    inst(VMax, vreg.id);
    inst(TrfOut, vreg.id);
    inst(VTransStartEnd, vreg.id, 8);
    inst(TrfOut, vreg.id);
    VMask selInf = inst.AllocVMask();
    VReg id = inst.AllocVReg("");
    inst(VCoreId, id.id);
    id &= 127;
    inst(VGeS, id.id, inst.inst.ImmeU(8), selInf.id);
    inst(VSel, selInf.id, vreg.id, inst.inst.ImmeF(-INFINITY), vreg.id);
    inst(VMax, vreg.id);
    inst(TrfOut, vreg.id);
}

// use MTI_REDUCTION_V_MIN to get min value and expand to all vector
void
MinFExpand(Inst2 &inst, const VReg &v_reg)
{
    inst(VMin, v_reg.id);
    inst(TrfOut, v_reg.id);
    inst(VTransStartEnd, v_reg.id, 8);
    inst(TrfOut, v_reg.id);
    VMask selInf = inst.AllocVMask();
    VReg id = inst.AllocVReg("");
    inst(VCoreId, id.id);
    id &= 127;
    inst(VGeS, id.id, inst.inst.ImmeU(8), selInf.id);
    inst(VSel, selInf.id, v_reg.id, inst.inst.ImmeF(INFINITY), v_reg.id);
    inst(VMin, v_reg.id);
    inst(TrfOut, v_reg.id);
}

// 地址都对 128 对齐
// EmbeddingWidth 是 128 的倍速
void
Embedding2D(Inst2 &inst,
            const VMem &embeddingData,
            int embeddingWidth,
            const VMem &indicesData,
            int indicesSize,
            const VMem &output)
{
    auto selectPx = inst.GetSelectPxMaskT();
    auto const0 = inst.AllocVReg("Const0");
    const0 = 0;

    auto sCurIdx = inst.AllocSReg();
    inst(SMov, CONST_U32_0, sCurIdx.id);
    auto vCurIdx = inst.AllocVReg("CurIdx");
    vCurIdx = 0;

    auto indices = inst.AllocVReg("Indices");

    // 准备 indices 前 1024 数据，每次只能处理 1024 个数据
    inst(VLoad, indicesData.startAddr / 128, indices.id);

    auto writeAddr = inst.AllocSReg();
    inst(SMov, inst.inst.ImmeU(output.startAddr / 128), writeAddr.id);

    const int PRegOutLoop = 1;
    inst(SLsS, sCurIdx.id, inst.inst.ImmeS(indicesSize), PRegOutLoop);
    inst.inst.DoWhile(
        PRegOutLoop,
        [&]()
        {
            auto sIdxVal = inst.AllocSReg();
            auto vIdxVal = inst.AllocVReg("IdxVal");
            auto vMask = inst.AllocVMask();

            inst(VEqS, selectPx.id, vCurIdx.id, vMask.id);
            inst(VSel, vMask.id, const0.id, indices.id, vIdxVal.id);
            MaxFExpand(inst, vIdxVal);
            inst(VF2S, vIdxVal.id, CONST_U32_NEG_1, vIdxVal.id);
            inst(VPush, vIdxVal.id);
            inst(SPop, sIdxVal.id);

            // 读取偏移
            auto readStart = inst.AllocSReg();
            inst(SMulU,
                 sIdxVal.id,
                 inst.inst.ImmeU(embeddingWidth / 128),
                 readStart.id);

            // 读取用的地址
            auto readAddr = inst.AllocSReg();
            inst(SAddS,
                 readStart.id,
                 inst.inst.ImmeU(embeddingData.startAddr / 128),
                 readAddr.id);

            auto embedding = inst.AllocVReg("Embedding", false);
            // 把从 embeddingAddr + ReadStart  的 [0, embeddingWidth) 的数据
            // 写到 outputAddr    + WriteStart 的 [0, embeddingWidth) 位置

            int copyTime = (embeddingWidth + 1023) / 1024;
            for (int i = 1; i <= copyTime; i++)
            {
                int copyColCnt = (i * 8) > (embeddingWidth / 128)
                                     ? ((embeddingWidth / 128) % 8)
                                     : 8;
                inst(VLoadBySRegWithMask,
                     readAddr.id,
                     embedding.id,
                     (1 << copyColCnt) - 1);
                inst(VStoreBySRegWithMask,
                     embedding.id,
                     writeAddr.id,
                     (1 << copyColCnt) - 1);
                inst(SAddS,
                     readAddr.id,
                     inst.inst.ImmeS(copyColCnt),
                     readAddr.id);
                inst(SAddS,
                     writeAddr.id,
                     inst.inst.ImmeS(copyColCnt),
                     writeAddr.id);
            }

            inst(SAddS, sCurIdx.id, CONST_U32_1, sCurIdx.id);
            inst(VAddS, vCurIdx.id, CONST_U32_1, vCurIdx.id);

            inst(SLsS, sCurIdx.id, inst.inst.ImmeS(indicesSize), PRegOutLoop);
        });
}

void
EmbeddingAny(Inst2 &inst,
             const VMem &embeddingData,
             int embeddingWidth,
             const VMem &indicesData,
             int indicesSize,
             const VMem &output)
{
    auto selectPx = inst.GetSelectPxMaskT();
    auto const0 = inst.AllocVReg("Const0");
    const0 = 0;

    auto sCurIdx = inst.AllocSReg();
    inst(SMov, CONST_U32_0, sCurIdx.id);
    auto vCurIdx = inst.AllocVReg("CurIdx");
    vCurIdx = 0;

    auto indices = inst.AllocVReg("Indices");

    // 准备 indices 前 1024 数据，每次只能处理 1024 个数据
    inst(VLoad, indicesData.startAddr / 128, indices.id);

    auto sIdxVal = inst.AllocSReg();
    auto vIdxVal = inst.AllocVReg("IdxVal");
    auto vMask = inst.AllocVMask();

    std::map<int, int> xx;
    xx[1] = 0;
    xx[2] = 1;
    xx[4] = 2;
    xx[8] = 3;
    xx[16] = 4;
    xx[32] = 5;
    xx[64] = 6;

    for (int i = 0; i < indicesSize; i++)
    {
        inst(VEqS, selectPx.id, vCurIdx.id, vMask.id);
        // in veloce, will cause infinite 0x4004 in second loop if VF2S's in and
        // out use same vreg.
        VReg vIdxTemp = inst.AllocVReg("");
        inst(VSel, vMask.id, const0.id, indices.id, vIdxTemp.id);
        MaxFExpand(inst, vIdxTemp);
        inst(VF2S, vIdxTemp.id, CONST_U32_NEG_1, vIdxVal.id);
        inst(VPush, vIdxVal.id);
        inst(SPop, sIdxVal.id);

        inst(SMulU, sIdxVal.id, inst.inst.ImmeU(embeddingWidth), sIdxVal.id);
        // addr align to 128 and div 128
        inst(SShrU, sIdxVal.id, inst.inst.ImmeU(7), sIdxVal.id);
        sIdxVal += embeddingData.startAddr / 128;

        // data offset to 128-aligned addr

        inst(VShlU,
             vIdxVal.id,
             inst.inst.ImmeU(xx[embeddingWidth]),
             vIdxVal.id);
        vIdxVal &= 127;

        auto itoa = inst.GetSelectPxMaskT();
        auto embbData = inst.AllocVReg("Embb data");
        inst(VLoadBySReg, sIdxVal.id, embbData.id);

        itoa += vIdxVal;
        itoa &= 127;

        inst(SetPermute, itoa.id);
        inst(VPermute, embbData.id);
        inst(TrfOut, embbData.id);

        output[Range(i * embeddingWidth, (i + 1) * embeddingWidth)] =
            embbData[Range(0, embeddingWidth)];

        inst(SAddS, sCurIdx.id, CONST_U32_1, sCurIdx.id);
        inst(VAddS, vCurIdx.id, CONST_U32_1, vCurIdx.id);
    }
}

void Embeddings(Inst2 &inst,
                uint32_t hbmEmbeddingDataAddr,
                uint32_t embeddingHeight,
                uint32_t embeddingWidth,
                int input_addr,
                int inputSize,
                int output_addr,
                int weight_input_addr)
{
    assert(embeddingWidth % 128 == 0);
    
    // uint32_t coutinuousDataSize = inst.resource.BiggestContinuousAvailableVMemSize(128);

    // assert(coutinuousDataSize >= embeddingWidth);

    uint32_t coutinuousDataSize = inst.resource.BiggestContinuousAvailableVMemSize(128) - weight_input_addr;

    uint32_t embeddingDataHeight = 1;
    for (; embeddingDataHeight * 2 <= (coutinuousDataSize / embeddingWidth) && 
           embeddingDataHeight * 2 <= embeddingHeight;
        embeddingDataHeight *= 2)
    {
    }

    std::clog << "Segment Embeddings Data Height: " << embeddingDataHeight << "\n";

    uint32_t embeddingDataSize = embeddingDataHeight * embeddingWidth;

    auto selectPx = inst.GetSelectPxMaskT();
    auto const0 = inst.AllocVReg("Const0");
    const0 = 0;

    auto sCurIdx = inst.AllocSReg();
    inst(SMov, CONST_F32_0, sCurIdx.id);
    auto vCurIdx = inst.AllocVReg("CurIdx");
    vCurIdx = 0;

    SReg sEmbeddingHeight = inst.AllocSReg();
    sEmbeddingHeight = embeddingDataHeight;
    SReg currentAvailableEmbeddingIdx = inst.AllocSReg();
    currentAvailableEmbeddingIdx = -(int32_t)embeddingDataHeight;

    auto indices = inst.AllocVReg("Indices");

    inst(VLoad, input_addr / 128, indices.id);
    std::clog << "embeddingDataSize: " << embeddingDataSize << std::endl;
    std::clog << "input_addr: " << input_addr << std::endl;

    auto writeAddr = inst.AllocSReg();
    inst(SMov, inst.inst.ImmeU(output_addr / 128), writeAddr.id);

    std::clog << "output_addr: " << output_addr << std::endl;

    const int PRegOutLoop = 1;
    // 判断inputsize是否为0
    inst(SLsS, sCurIdx.id, inst.inst.ImmeS(inputSize), PRegOutLoop);

    inst.inst.DoWhile(
        PRegOutLoop,
        [&]()
        {
            auto sIdxVal = inst.AllocSReg();
            auto vIdxVal = inst.AllocVReg("IdxVal");
            auto vMask = inst.AllocVMask();
            
            /*
                vCurIdx.id是input当前下标
                selectPx.id是coreid的号码，
                这样判断会让vMask中排除当前下标的值之外都为false，
                比如当前下标为3.
                vmask:
                F F F T F F F F F F F F F F F F F F F F
                F F F F F F F F F F F F F F F F F F F F
                const0中间都为0,
                indices中就是我们输入的input
                则vIdxVal反映出的情况为
                0 0 0 input[3] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                ........
                maxFExpand会扩充到vIdxVal全部的
                input[3] input[3] input[3] input[3] input[3] input[3] input[3]
                input[3] input[3] input[3] input[3] input[3] input[3] input[3]
            */
            inst(VEqS, selectPx.id, vCurIdx.id, vMask.id);
            inst(VSel, vMask.id, const0.id, indices.id, vIdxVal.id);
            MaxFExpand(inst, vIdxVal);
            // 不是很清楚这里为什么会需要用float to int，且用-1去比较?
            inst(VF2S, vIdxVal.id, CONST_U32_NEG_1, vIdxVal.id);
            //VPush, 从vIdxVal.id中的core0发送到Vector-to-Scalar这个FIFO中
            inst(VPush, vIdxVal.id);
            //SPop 从Vector-to-Scalar这个FIFO中弹出一个值到sIdxVal.id, sIdxVal.id得到是vIdxVal.id中的第一个值input[3]
            inst(SPop, sIdxVal.id);

            auto diff = sIdxVal - currentAvailableEmbeddingIdx;
            inst(SLsS, diff.id, CONST_U32_0, 2);

            inst.inst.If(2, 
                            [&]()
                            {
                                SReg hbmAddr = inst.AllocSReg();
                                hbmAddr = sIdxVal;
                                // 与embeddingDataHeight保持对齐
                                hbmAddr &= ~(embeddingDataHeight - 1);
                                currentAvailableEmbeddingIdx = hbmAddr;
                                hbmAddr *= embeddingWidth;
                                hbmAddr += hbmEmbeddingDataAddr;
                                SReg vmemAddr = inst.AllocSReg();
                                vmemAddr = weight_input_addr;
                                SReg len = inst.AllocSReg();
                                len = embeddingDataSize;
                               __InfraRuntime::Hbm2VMemRuntime(inst, hbmAddr, vmemAddr, len);
                            });
            inst.inst.IfNot(
                2,
                [&]()
                {
                    inst(SGeS, diff.id, sEmbeddingHeight.id, 3);
                    inst.inst.If(3,
                                [&]()
                                {
                                    SReg hbmAddr = inst.AllocSReg();
                                    hbmAddr = sIdxVal;
                                    // 与embeddingDataHeight保持对齐
                                    hbmAddr &= ~(embeddingDataHeight - 1);
                                    currentAvailableEmbeddingIdx = hbmAddr;
                                    hbmAddr *= embeddingWidth;
                                    hbmAddr += hbmEmbeddingDataAddr;
                                    SReg vmemAddr = inst.AllocSReg();
                                    vmemAddr = weight_input_addr;
                                    SReg len = inst.AllocSReg();
                                    len = embeddingDataSize;
                                    __InfraRuntime::Hbm2VMemRuntime(inst, hbmAddr, vmemAddr, len);
                                }
                    );
                }
            );
            sIdxVal &= (embeddingDataHeight - 1);

            // 读取偏移
            auto readStart = inst.AllocSReg();
            inst(SMulU, sIdxVal.id, inst.inst.ImmeU(embeddingWidth / 128), readStart.id);

            // 读取用的地址
            auto readAddr = inst.AllocSReg();
            inst(SAddS, readStart.id, inst.inst.ImmeU(weight_input_addr / 128), readAddr.id);

            auto embedding = inst.AllocVReg("Embedding", false);

            int copyTime = (embeddingWidth + 1023) / 1024;
            for (int i = 1; i <= copyTime; i++)
            {
                int copyColCnt = (i * 8) > (embeddingWidth / 128) ? ((embeddingWidth / 128) % 8) : 8;
                inst(VLoadBySRegWithMask, readAddr.id, embedding.id, (1 << copyColCnt) - 1);
                inst(VStoreBySRegWithMask, embedding.id, writeAddr.id, (1 << copyColCnt) - 1);
                inst(SAddS, readAddr.id, inst.inst.ImmeS(copyColCnt), readAddr.id);
                inst(SAddS, writeAddr.id, inst.inst.ImmeS(copyColCnt), writeAddr.id);
            }
            inst(SAddS, sCurIdx.id, CONST_U32_1, sCurIdx.id);
            inst(VAddS, vCurIdx.id, CONST_U32_1, vCurIdx.id);

            inst(SLsS, sCurIdx.id, inst.inst.ImmeS(inputSize), PRegOutLoop);
        }
    );
}

void
EmbeddingEx(Inst2 &inst,
            uint32_t hbmEmbeddingDataAddr,
            int embeddingHeight,
            int embeddingWidth,
            const VMem &indicesData,
            int indicesSize,
            const VMem &output,
            const int weight_addr)
{
    assert(embeddingWidth % 128 == 0);

    // uint32_t continuousDataSize =
    //     inst.resource.BiggestContinuousAvailableVMemSize(128);

    uint32_t continuousDataSize =
        inst.resource.BiggestContinuousAvailableVMemSize(128) - weight_addr;

    assert(continuousDataSize >= embeddingWidth);

    uint32_t embeddingDataHeight = 1;
    for (; embeddingDataHeight * 2 <= (continuousDataSize / embeddingWidth) &&
           embeddingDataHeight * 2 <= embeddingHeight;
         embeddingDataHeight *= 2)
    {
    }

    std::clog << "Segment Embedding Data Height: " << embeddingDataHeight
              << "\n";

    uint32_t embeddingDataSize = embeddingDataHeight * embeddingWidth;

    VMem embeddingData = inst.Alloc(embeddingDataSize, 128, "EmbeddingData");
    std::clog << "embeddingData.len: " << embeddingData.len << std::endl;

    auto selectPx = inst.GetSelectPxMaskT();
    auto const0 = inst.AllocVReg("Const0");
    const0 = 0;

    auto sCurIdx = inst.AllocSReg();
    inst(SMov, CONST_U32_0, sCurIdx.id);
    auto vCurIdx = inst.AllocVReg("CurIdx");
    vCurIdx = 0;

    SReg sEmbeddingHeight = inst.AllocSReg();
    sEmbeddingHeight = embeddingDataHeight;
    SReg currentAvailableEmbeddingIdx = inst.AllocSReg();
    currentAvailableEmbeddingIdx = -(int32_t)embeddingDataHeight;

    auto indices = inst.AllocVReg("Indices");

    // 准备 indices 前 1024 数据，每次只能处理 1024 个数据
    inst(VLoad, indicesData.startAddr / 128, indices.id);

    auto writeAddr = inst.AllocSReg();
    inst(SMov, inst.inst.ImmeU(1024 / 128), writeAddr.id);
    std::clog << "output.startAddr: " << output.startAddr << std::endl;

    embeddingData.startAddr = weight_addr;

    const int PRegOutLoop = 1;
    inst(SLsS, sCurIdx.id, inst.inst.ImmeS(indicesSize), PRegOutLoop);
    std::clog << "embeddingData.startAddr: " << embeddingData.startAddr << std::endl;
    inst.inst.DoWhile(
        PRegOutLoop,
        [&]()
        {
            auto sIdxVal = inst.AllocSReg();
            auto vIdxVal = inst.AllocVReg("IdxVal");
            auto vMask = inst.AllocVMask();

            inst(VEqS, selectPx.id, vCurIdx.id, vMask.id);
            inst(VSel, vMask.id, const0.id, indices.id, vIdxVal.id);
            MaxFExpand(inst, vIdxVal);
            inst(VF2S, vIdxVal.id, CONST_U32_NEG_1, vIdxVal.id);
            inst(VPush, vIdxVal.id);
            inst(SPop, sIdxVal.id);

            // Is current read embeddingData available?
            auto diff = sIdxVal - currentAvailableEmbeddingIdx;
            inst(SLsS, diff.id, CONST_U32_0, 2);
            // inst(SEqS, vCurIdx.id, CONST_U32_0, 2);

            // inst(POr, 2, 3, 4);
            inst.inst.If(2,
                         [&]()
                         {
                             SReg hbmAddr = inst.AllocSReg();
                             hbmAddr = sIdxVal;
                             hbmAddr &= ~(embeddingDataHeight - 1);
                             currentAvailableEmbeddingIdx = hbmAddr;
                             hbmAddr *= embeddingWidth;
                             hbmAddr += hbmEmbeddingDataAddr;
                             SReg vmemAddr = inst.AllocSReg();
                             vmemAddr = embeddingData.startAddr;
                             SReg len = inst.AllocSReg();
                             len = embeddingData.len;
                             __InfraRuntime::Hbm2VMemRuntime(inst,
                                                             hbmAddr,
                                                             vmemAddr,
                                                             len);
                         });
            inst.inst.IfNot(
                2,
                [&]()
                {
                    inst(SGeS, diff.id, sEmbeddingHeight.id, 3);
                    inst.inst.If(3,
                                 [&]()
                                 {
                                     SReg hbmAddr = inst.AllocSReg();
                                     hbmAddr = sIdxVal;
                                     hbmAddr &= ~(embeddingDataHeight - 1);
                                     currentAvailableEmbeddingIdx = hbmAddr;
                                     hbmAddr *= embeddingWidth;
                                     hbmAddr += hbmEmbeddingDataAddr;
                                     SReg vmemAddr = inst.AllocSReg();
                                     vmemAddr = embeddingData.startAddr;
                                     SReg len = inst.AllocSReg();
                                     len = embeddingData.len;
                                    //  len = 1024;
                                     __InfraRuntime::Hbm2VMemRuntime(inst,
                                                                     hbmAddr,
                                                                     vmemAddr,
                                                                     len);
                                 });
                });

            sIdxVal &= (embeddingDataHeight - 1);

            // 读取偏移
            auto readStart = inst.AllocSReg();
            inst(SMulU,
                 sIdxVal.id,
                 inst.inst.ImmeU(embeddingWidth / 128),
                 readStart.id);

            // 读取用的地址
            auto readAddr = inst.AllocSReg();
            inst(SAddS,
                 readStart.id,
                 inst.inst.ImmeU(embeddingData.startAddr / 128),
                 readAddr.id);

            auto embedding = inst.AllocVReg("Embedding", false);
            // 把从 embeddingAddr + ReadStart  的 [0, embeddingWidth) 的数据
            // 写到 outputAddr    + WriteStart 的 [0, embeddingWidth) 位置

            int copyTime = (embeddingWidth + 1023) / 1024;
            for (int i = 1; i <= copyTime; i++)
            {
                int copyColCnt = (i * 8) > (embeddingWidth / 128)
                                     ? ((embeddingWidth / 128) % 8)
                                     : 8;
                inst(VLoadBySRegWithMask,
                     readAddr.id,
                     embedding.id,
                     (1 << copyColCnt) - 1);
                inst(VStoreBySRegWithMask,
                     embedding.id,
                     writeAddr.id,
                     (1 << copyColCnt) - 1);
                inst(SAddS,
                     readAddr.id,
                     inst.inst.ImmeS(copyColCnt),
                     readAddr.id);
                inst(SAddS,
                     writeAddr.id,
                     inst.inst.ImmeS(copyColCnt),
                     writeAddr.id);
            }

            inst(SAddS, sCurIdx.id, CONST_U32_1, sCurIdx.id);
            inst(VAddS, vCurIdx.id, CONST_U32_1, vCurIdx.id);

            inst(SLsS, sCurIdx.id, inst.inst.ImmeS(indicesSize), PRegOutLoop);
        });

    inst.FreeVMem(&embeddingData);
}

#define SOFTMAX_SUB

void
SoftmaxSingle(const VMem &vmem, const VMem &out)
{
    Inst2 &inst2 = *vmem.inst2;
    VReg totSum = inst2.AllocVReg("Total Exp Sum");
    totSum = 0.0f;

    int times = (vmem.len + 1023) / 1024;

#ifdef SOFTMAX_SUB

    VReg vmax = inst2.AllocVReg("");
    vmax = -INFINITY;
    for (int i = 0; i < times; i++)
    {
        int curLen = i == times - 1
                         ? (vmem.len % 1024 == 0 ? 1024 : vmem.len % 1024)
                         : 1024;
        VReg r = inst2.AllocVReg("InputVal");
        r[Range(0, curLen)] = vmem[OffLen(i * 1024, curLen)];
        r.isFloat = vmem.isFloat;
        VMask big = vmax < r;
        if (curLen != 1024)
        {
            VMask sel = inst2.AllocVMask();
            VReg id = inst2.GetSelectPxMaskT();
            inst2(VLsS, id.id, inst2.inst.ImmeS(curLen), sel.id);
            big &= sel;
        }
        inst2(VSel, big.id, vmax.id, r.id, vmax.id);
        MaxFExpand(inst2, vmax);
    }

#endif

    for (int i = 0; i < times; i++)
    {
        int curLen = i == times - 1
                         ? (vmem.len % 1024 == 0 ? 1024 : vmem.len % 1024)
                         : 1024;
        VReg r = inst2.AllocVReg("SoftmaxVal");
        r[Range(0, curLen)] = vmem[OffLen(i * 1024, curLen)];

#ifdef SOFTMAX_SUB
        inst2(VSubF, r.id, vmax.id, r.id);
#endif

        inst2(VExp, r.id);
        VReg rexpabs = inst2.AllocVReg("ExpAbsVal");
        inst2(UrfOut, rexpabs.id);
        VMask neg = inst2.AllocVMask();
        inst2(VLsF, r.id, CONST_F32_0, neg.id);
        inst2(VReciprocal, rexpabs.id);
        VReg rexpabsp = inst2.AllocVReg("ExpAbsValReciprocal");
        inst2(UrfOut, rexpabsp.id);
        VReg rexp = inst2.AllocVReg("ExpVal");
        inst2(VSel, neg.id, rexpabs.id, rexpabsp.id, rexp.id);
        GC(std::move(rexpabs));
        GC(std::move(rexpabsp));
        GC(std::move(neg));

        VMask hasVal = inst2.AllocVMask();
        VReg coreId = inst2.GetSelectPxMaskT();
        inst2(VLsS, coreId.id, inst2.inst.ImmeS(curLen), hasVal.id);
        VReg zero = inst2.AllocVReg("ZeroVal");
        zero = 0;
        inst2(VSel, hasVal.id, zero.id, rexp.id, rexp.id);
        GC(std::move(hasVal));
        GC(std::move(coreId));
        GC(std::move(zero));

        VReg expsum = inst2.AllocVReg("ExpSumVal");
        inst2(VSum, rexp.id);
        inst2(TrfOut, expsum.id);

        out[OffLen(i * 1024, curLen)] = rexp[Range(0, curLen)];

        VReg tmp = inst2.AllocVReg("RotVal");
        tmp = expsum;
        expsum.isFloat = true;
        tmp.isFloat = true;
        for (int i = 1; i < 8; i++)
        {
            inst2(VSubRotL, tmp.id, tmp.id);
            expsum += tmp;
        }

        totSum += expsum;
    }

    inst2(VReciprocal, totSum.id);
    inst2(UrfOut, totSum.id);

    for (int i = 0; i < times; i++)
    {
        int curLen = i == times - 1
                         ? (vmem.len % 1024 == 0 ? 1024 : vmem.len % 1024)
                         : 1024;
        VReg r = inst2.AllocVReg("ExpRVal");
        r[Range(0, curLen)] = out[OffLen(i * 1024, curLen)];
        inst2(VMulF, r.id, totSum.id, r.id);
        out[OffLen(i * 1024, curLen)] = r[Range(0, curLen)];
    }
}
} // namespace __InternalImpl

void
Transpose(std::vector<Instruction *> &instList,
          const std::vector<uint32_t> &reservedVReg,
          const std::vector<uint32_t> &reservedSReg,
          const std::vector<uint32_t> &reservedVMask,
          const std::vector<uint32_t> &reservedPermit,
          uint32_t vMemSrcAddr,
          uint32_t height,
          uint32_t width,
          uint32_t vMemDestAddr)
{
    Inst2 inst2;
    inst2.logLevel = globalLogLevel;
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }
    inst2.Alloc(vMemDestAddr + height * width, "Transpose Reserved VMem");
    VMem src(vMemSrcAddr, height * width, false, &inst2);
    VMem dest(vMemDestAddr, height * width, false, &inst2);
    __InternalImpl::TransposeAny(inst2, src, dest, height, width);
    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
InstVer::PermuteInst(Inst2 &inst2,
                     uint32_t vMemSrcAddr,
                     const std::vector<uint32_t> &dimSize,
                     const std::vector<uint32_t> &newDims,
                     uint32_t vMemDestAddr)
{
    const int callCnt = CallCount(__FUNCTION__);
    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: Permute#" << callCnt << "(@"
                  << vMemSrcAddr << "[";

        bool first = true;
        for (auto s : dimSize)
        {
            if (!first)
            {
                std::clog << ", ";
            }
            std::clog << s;
            first = false;
        }
        std::clog << "]) => @" << vMemDestAddr << "[";
        first = true;
        for (auto s : newDims)
        {
            if (!first)
            {
                std::clog << ", ";
            }
            std::clog << dimSize[s];
            first = false;
        }
        std::clog << "]" << COLOR::WHITE << "\n";
    }
    auto dataSize = std::accumulate(dimSize.begin(),
                                    dimSize.end(),
                                    1,
                                    std::multiplies<uint32_t>());
    VMem src(vMemSrcAddr, dataSize, false, &inst2);
    VMem dest(vMemDestAddr, dataSize, false, &inst2);

    bool needPermute = false;
    for (int i = 0; i < newDims.size(); i++)
    {
        if (i != newDims[i])
        {
            needPermute = true;
            break;
        }
    }

    if (needPermute)
    {
        __InternalImpl::Permute(inst2, src, dest, dimSize, newDims);
    }
    else
    {
        int i = 0;
        auto val = inst2.AllocVReg("Val");
        for (; i < src.len / 1024; i++)
        {
            val = src[Range(i * 1024, i * 1024 + 1024)];
            dest[Range(i * 1024, i * 1024 + 1024)] = val;
        }
        val[Range(0, src.len % 1024)] = src[Range(i * 1024, src.len)];
        dest[Range(i * 1024, dest.len)] = val[Range(0, dest.len % 1024)];
    }
}

void
Permute(std::vector<Instruction *> &instList,
        const std::vector<uint32_t> &reservedVReg,
        const std::vector<uint32_t> &reservedSReg,
        const std::vector<uint32_t> &reservedVMask,
        const std::vector<uint32_t> &reservedPermit,
        uint32_t vMemSrcAddr,
        const std::vector<uint32_t> &dimSize,
        const std::vector<uint32_t> &newDims,
        uint32_t vMemDestAddr)
{
    Inst2 inst2;
    inst2.logLevel = globalLogLevel;
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }
    auto dataSize = std::accumulate(dimSize.begin(),
                                    dimSize.end(),
                                    1,
                                    std::multiplies<uint32_t>());
    inst2.Alloc(vMemDestAddr + dataSize, "Permute Reserved VMem");
    InstVer::PermuteInst(inst2, vMemSrcAddr, dimSize, newDims, vMemDestAddr);
    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
Embedding(std::vector<Instruction *> &instList,
          const std::vector<uint32_t> &reservedVReg,
          const std::vector<uint32_t> &reservedSReg,
          const std::vector<uint32_t> &reservedVMask,
          const std::vector<uint32_t> &reservedPermit,
          uint32_t embeddingAddr,
          uint32_t embeddingHeight,
          uint32_t embeddingWidth,
          uint32_t indicesAddr,
          uint32_t indicesSize,
          uint32_t outputAddr)
{
    const int callCnt = CallCount(__FUNCTION__);
    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: Embedding#" << callCnt << "(@"
                  << embeddingAddr << "[" << embeddingHeight << " x "
                  << embeddingWidth << "], @" << indicesAddr << "["
                  << indicesSize << "]) => @" << outputAddr << COLOR::WHITE
                  << std::endl;
    }

    Inst2 inst2;
    inst2.logLevel = globalLogLevel;
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }
    auto size = embeddingHeight * embeddingWidth;
    VMem embedding(embeddingAddr, size, false, &inst2);
    VMem indices(indicesAddr, indicesSize, false, &inst2);
    VMem output(outputAddr, indicesSize * embeddingWidth, false, &inst2);

    int i = 0;
    for (; i < indicesSize / 1024; i++)
    {
        __InternalImpl::Embedding2D(
            inst2,
            embedding,
            embeddingWidth,
            indices[Range(i * 1024, (i + 1) * 1024)],
            1024,
            output[Range(i * embeddingWidth * 1024,
                         (i + 1) * embeddingWidth * 1024)]);
    }

    if (indicesSize % 1024 != 0)
    {
        __InternalImpl::Embedding2D(
            inst2,
            embedding,
            embeddingWidth,
            indices[Range(i * 1024, indicesSize)],
            indicesSize % 1024,
            output[Range(i * embeddingWidth * 1024,
                         indicesSize * embeddingWidth)]);
    }

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
Linear(std::vector<Instruction *> &instList,
       const std::vector<uint32_t> &reservedVReg,
       const std::vector<uint32_t> &reservedSReg,
       const std::vector<uint32_t> &reservedVMask,
       const std::vector<uint32_t> &reservedPermit,
       uint32_t inputAddr,
       const std::array<uint32_t, 4> &inputSize,
       uint32_t weightAddr,
       const std::array<uint32_t, 2> &weightSize,
       uint32_t outputAddr)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    assert(w0 == d3);

    Inst2 inst2;
    inst2.AllocF(outputAddr + d0 * d1 * __Linear::AlignTo128Bytes(d2) * w1,
                 "Linear Reserved VMem");
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }

    auto inputDataSize = std::accumulate(inputSize.begin(),
                                         inputSize.end(),
                                         1u,
                                         std::multiplies<uint32_t>());

    VMem input(inputAddr, inputDataSize, true, &inst2);
    VMem weight(weightAddr, w0 * w1, true, &inst2);

    VMem output(outputAddr, d0 * d1 * d2 * w1, true, &inst2);

    __Linear::LinearInternal(inst2,
                             input,
                             inputSize,
                             weight,
                             weightSize,
                             output);

    // std::clog << "INTERNAL MEM: " << inst2.availableAddr << "\n";

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
LinearSplitWidth(std::vector<Instruction *> &instList,
                 const std::vector<uint32_t> &reservedVReg,
                 const std::vector<uint32_t> &reservedSReg,
                 const std::vector<uint32_t> &reservedVMask,
                 const std::vector<uint32_t> &reservedPermit,
                 uint32_t splitWidth,
                 uint32_t inputAddr,
                 const std::array<uint32_t, 4> &inputSize,
                 uint32_t weightAddr,
                 const std::array<uint32_t, 2> &weightSize,
                 uint32_t outputAddr)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    assert(w0 == d3);
    // assert(d0 == 1 && d1 == 1);

    // assert(d2 % 128 == 0);

    assert(w1 % splitWidth == 0);
    assert(splitWidth % 128 == 0);
    assert(w0 % 128 == 0);

    Inst2 inst2;
    inst2.AllocF(outputAddr + d0 * d1 * __Linear::AlignTo128Bytes(d2) * w1,
                 "LinearSplitWidth Reserved VMem");
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }

    auto inputDataSize = std::accumulate(inputSize.begin(),
                                         inputSize.end(),
                                         1u,
                                         std::multiplies<uint32_t>());

    VMem input(inputAddr, inputDataSize, true, &inst2);
    VMem weight(weightAddr, w0 * w1, true, &inst2);

    VMem output(outputAddr, d0 * d1 * d2 * w1, true, &inst2);

    __Linear::LinearInternalSplitWidth(inst2,
                                       splitWidth,
                                       input,
                                       inputSize,
                                       weight,
                                       weightSize,
                                       output);

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
LinearSplit(std::vector<Instruction *> &instList,
            const std::vector<uint32_t> &reservedVReg,
            const std::vector<uint32_t> &reservedSReg,
            const std::vector<uint32_t> &reservedVMask,
            const std::vector<uint32_t> &reservedPermit,
            uint32_t splitHeight,
            uint32_t splitWidth,
            uint32_t inputAddr,
            const std::array<uint32_t, 4> &inputSize,
            uint32_t weightAddr,
            const std::array<uint32_t, 2> &weightSize,
            uint32_t outputAddr,
            uint32_t blockCount)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    assert(w1 % splitWidth == 0);
    assert(splitWidth % 128 == 0);
    assert(w0 % 128 == 0);

    assert(w0 == splitHeight);
    assert(d3 % splitHeight == 0);

    Inst2 inst2;
    inst2.AllocF(outputAddr + d0 * d1 * __Linear::AlignTo128Bytes(d2) * w1,
                 "LinearSplit Reserved VMem");
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }

    auto inputDataSize = std::accumulate(inputSize.begin(),
                                         inputSize.end(),
                                         1u,
                                         std::multiplies<uint32_t>());

    VMem input(inputAddr, inputDataSize, true, &inst2);
    VMem weight(weightAddr, w0 * w1, true, &inst2);

    VMem output(outputAddr, d0 * d1 * d2 * w1, true, &inst2);

    __Linear::LinearInternalSplit(inst2,
                                  splitHeight,
                                  splitWidth,
                                  input,
                                  inputSize,
                                  weight,
                                  weightSize,
                                  output,
                                  blockCount);

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
EmbeddingAny(std::vector<Instruction *> &instList,
             const std::vector<uint32_t> &reservedVReg,
             const std::vector<uint32_t> &reservedSReg,
             const std::vector<uint32_t> &reservedVMask,
             const std::vector<uint32_t> &reservedPermit,
             uint32_t embeddingAddr,
             uint32_t embeddingHeight,
             uint32_t embeddingWidth,
             uint32_t indicesAddr,
             uint32_t indicesSize,
             uint32_t outputAddr)
{
    const int callCnt = CallCount(__FUNCTION__);
    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: EmbeddingAny#" << callCnt << "(@"
                  << embeddingAddr << "[" << embeddingHeight << " x "
                  << embeddingWidth << "], @" << indicesAddr << "["
                  << indicesSize << "]) => @" << outputAddr << COLOR::WHITE
                  << std::endl;
    }

    Inst2 inst2;
    inst2.logLevel = globalLogLevel;
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }
    auto size = embeddingHeight * embeddingWidth;
    VMem embedding(embeddingAddr, size, false, &inst2);
    VMem indices(indicesAddr, indicesSize, false, &inst2);
    VMem output(outputAddr, indicesSize * embeddingWidth, false, &inst2);
    int i = 0;
    for (; i < indicesSize / 1024; i++)
    {
        __InternalImpl::EmbeddingAny(
            inst2,
            embedding,
            embeddingWidth,
            indices[Range(i * 1024, (i + 1) * 1024)],
            1024,
            output[Range(i * embeddingWidth * 1024,
                         (i + 1) * embeddingWidth * 1024)]);
    }

    if (indicesSize % 1024 != 0)
    {
        __InternalImpl::EmbeddingAny(
            inst2,
            embedding,
            embeddingWidth,
            indices[Range(i * 1024, indicesSize)],
            indicesSize % 1024,
            output[Range(i * embeddingWidth * 1024,
                         indicesSize * embeddingWidth)]);
    }

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

/*
it will measure the available vmem size and required vmem size
as available vmem, it's unimplememted to measure the each vmem block
it only choose the biggest continuous avavilable vmem block
use __Linear::LinearInternalSplitRequiredSizeMeasure + splitHeight * w1
as required vmem size

prespilt weight:

load each height-splited weight
in graph, mem order is left to right, up do down;

                   +--+
                   |  |  it will split in width
                   +--+  while it not need full block size vmem requirement
+--+--+--+--+      |  |  each block spilted will store into hbm soon
|  |  |  |  | =>   +--+  0th block will reserve and spilt as last
+--+--+--+--+      |  |  so no need hbm load and store for 0th block
                   +--+
                   |  |
                   +--+

as result mem layout

+--+ < after linear
|1A|
+--+
|1B|    +--+--+ < output required
+--+    |1A|2A|
|1C|    +--+--+
+--+ => |1B|2B|
|2A|    +--+--+
+--+    |1C|2C|
|2B|    +--+--+
+--+
|2C|
+--+

how split:

+--+--+                 +--+--+--+
|1A|2A|   +--+--+--+    |o1|o2|o3|
+--+--+   |a1|b1|c1|    +--+--+--+
|1B|2B| X +--+--+--+ => |o4|o5|o6|
+--+--+   |a2|b2|c2|    +--+--+--+
|1C|2C|   +--+--+--+    |o7|o8|o9|
+--+--+                 +--+--+--+

          +--+
+--+--+   |a1|    +--+
|1A|2A| X +--+ => |o1|
+--+--+   |a2|    +--+
          +--+

+--+   +--+   +--+   +--+    +--+
|1A| X |a1| + |2A| X |a2| => |o1|
+--+   +--+   +--+   +--+    +--+

*/

void
InstVer::LinearExVMemIOInst(Inst2 &inst2,
                            uint32_t splitHeight,
                            uint32_t splitWidth,
                            uint32_t vmemInputAddr,
                            const std::array<uint32_t, 4> &inputSize,
                            uint32_t hbmWeightAddr,
                            const std::array<uint32_t, 2> &weightSize,
                            uint32_t vmemOutputAddr)
{
    const int callCnt = CallCount(__FUNCTION__);

    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    assert(w1 % splitWidth == 0);
    assert(splitWidth % 128 == 0);
    assert(w0 % 128 == 0);

    assert(w0 == d3);
    assert(d3 % splitHeight == 0);

    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: LinearExVMemIO#" << callCnt
                  << "(@" << vmemInputAddr << "[" << d0 << ", " << d1 << ", "
                  << d2 << ", " << d3 << "], @" << hbmWeightAddr << "[" << w0
                  << ", " << w1 << "]) => @" << vmemOutputAddr << COLOR::WHITE
                  << std::endl;
    }

    const auto outputDataSize = d0 * d1 * d2 * w1;
    const auto oldOutputVMemAddr = vmemOutputAddr;
    const auto oldInputVMemAddr = vmemInputAddr;
    const auto savedVMemSize = kVectorDataMemorySize;

    auto inputDataSize = std::accumulate(inputSize.begin(),
                                         inputSize.end(),
                                         1u,
                                         std::multiplies<uint32_t>());

    // How much vmem do I need
    auto requiredSize =
        __Linear::LinearInternalSplitRequiredSizeMeasure(splitHeight,
                                                         splitWidth,
                                                         inputSize,
                                                         {splitHeight, w1}) +
        splitHeight * w1;

    auto availableContinuousVMemSize =
        inst2.resource.BiggestContinuousAvailableVMemSize(128);

    bool needReserveVMemToHbm = availableContinuousVMemSize < requiredSize;

    uint32_t hbmAvailableAddr = HBMAddr();
    uint32_t vmemHbmAddr = hbmAvailableAddr;

    inst2.ForkAndPushResource();

    if (needReserveVMemToHbm)
    {
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON << "RESERVE VMEM TO HBM\n"
                      << COLOR::WHITE;
        }
        __Infra::VMEM_TO_HBM(inst2.inst.insts, 0, vmemHbmAddr, savedVMemSize);
        hbmAvailableAddr += savedVMemSize;
        Memcopy(VMem(vmemInputAddr, inputDataSize, true, &inst2),
                VMem(0, inputDataSize, true, &inst2));
        inst2.PushResource();
        vmemInputAddr = 0;
        vmemOutputAddr = inputDataSize;
        inst2.Alloc(inputDataSize, "InputData");
        inst2.Alloc(outputDataSize, "OutputData");
    }

    VMem input(vmemInputAddr, inputDataSize, true, &inst2);

    VMem output(vmemOutputAddr, d0 * d1 * d2 * w1, true, &inst2);

    if (requiredSize > inst2.resource.BiggestContinuousAvailableVMemSize(128))
    {
        auto splitIndex =
            inst2.resource.BiggestContinuousAvailableVMemSize(128) / 8 /
            splitHeight;
        splitIndex = __Utils::flp2(splitIndex);
        while (d3 % splitIndex != 0)
        {
            splitIndex /= 2;
        }
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON
                      << "SPLIT HBM WEIGHT WIDTH: " << splitIndex << "\n"
                      << COLOR::WHITE;
        }
        uint32_t widthSplit = splitIndex;
        int widthCnt = ((w1 + widthSplit - 1) / widthSplit);

        // if no enough vmem to spilt weight, save output to hbm
        bool needSaveOutput =
            inst2.resource.BiggestContinuousAvailableVMemSize(128) <
            splitHeight * w1 + splitHeight * widthSplit;

        uint32_t outputHbmAddr = hbmAvailableAddr;
        if (needSaveOutput)
        {
            hbmAvailableAddr += output.len;
            if (ShowFuncCallInfo())
            {
                std::clog << COLOR::KANON << "RESERVE OUTPUT TO HBM\n"
                          << COLOR::WHITE;
            }
            inst2.FreeVMem(&output);
        }

        bool needFreeInput =
            inst2.resource.BiggestContinuousAvailableVMemSize(128) <
            splitHeight * w1 + splitHeight * widthSplit;

        if (needFreeInput)
        {
            if (ShowFuncCallInfo())
            {
                std::clog << COLOR::KANON << "FREE INPUT DURING SPLIT WEIGHT\n"
                          << COLOR::WHITE;
            }
            inst2.FreeVMem(&input);
        }

        if (inst2.resource.BiggestContinuousAvailableVMemSize(128) <
            splitHeight * w1 + splitHeight * widthSplit)
        {
            splitIndex = 128;
            if (ShowFuncCallInfo())
            {
                std::clog << COLOR::KANON
                          << "RE-CHOOSE SPLIT HBM WEIGHT WIDTH: 128\n"
                          << COLOR::WHITE;
            }
            widthSplit = splitIndex;
            widthCnt = ((w1 + widthSplit - 1) / widthSplit);
        }

        for (int hi = 0; hi < w0 / splitHeight; hi++)
        {
            if (needSaveOutput)
            {
                // no need save at first
                if (hi != 0)
                {
                    __Infra::VMEM_TO_HBM(inst2.inst.insts,
                                         output.startAddr,
                                         outputHbmAddr,
                                         output.len);
                    inst2.FreeVMem(&output);
                }
            }

            if (needFreeInput && hi != 0)
            {
                inst2.FreeVMem(&input);
            }

            // alloc the weigth block first, so after free allWeight we can get
            // a bigger continous vmem
            VMem weight =
                inst2.Alloc(splitHeight * widthSplit, "A Weight Block");

            VMem allWeight = inst2.Alloc(splitHeight * w1, "A Weight Stripe");

            __Infra::HBM_TO_VMEM(inst2.inst.insts,
                                 hbmWeightAddr + hi * splitHeight * w1,
                                 allWeight.startAddr,
                                 splitHeight * w1);

            // reserve the 0th block for linear directly, no need save to hbm
            for (int i = 1; i < widthCnt; i++)
            {
                __Infra::MemcopyEx128(inst2,
                                      allWeight,
                                      i * widthSplit,
                                      w1,
                                      weight,
                                      0,
                                      widthSplit,
                                      widthSplit,
                                      splitHeight);

                __Infra::VMEM_TO_HBM(inst2.inst.insts,
                                     weight.startAddr,
                                     hbmAvailableAddr +
                                         i * splitHeight * widthSplit,
                                     splitHeight * widthSplit);
            }

            // first block
            __Infra::MemcopyEx128(inst2,
                                  allWeight,
                                  0,
                                  w1,
                                  weight,
                                  0,
                                  widthSplit,
                                  widthSplit,
                                  splitHeight);

            // free all weight
            inst2.FreeVMem(&allWeight);

            if (needSaveOutput)
            {
                // realloc the output vmem
                // if there no enough to alloc reform output's vmem
                // we need reserve input into hbm, then alloc the reform
                // output's vmem
                // we can reduce memcopy if alloc the output at bottom
                output.startAddr = inst2.resource.AllocVMemWithHint(
                    output.len,
                    kVectorDataMemorySize - output.len);
                __Infra::HBM_TO_VMEM(inst2.inst.insts,
                                     outputHbmAddr,
                                     output.startAddr,
                                     output.len);
            }

            if (needFreeInput)
            {
                input = inst2.AllocF(input.len, "Linear Input Data");
                __Infra::HBM_TO_VMEM(inst2.inst.insts,
                                     vmemHbmAddr + oldInputVMemAddr,
                                     input.startAddr,
                                     input.len);
            }

            for (int i = 0; i < widthCnt; i++)
            {

                uint32_t curWidth = widthSplit;
                if (i == widthCnt - 1 && w1 % widthSplit != 0)
                {
                    curWidth = w1 % widthSplit;
                }

                __Linear::LinearInternalSplit(
                    inst2,
                    splitHeight,
                    splitWidth,
                    input,
                    inputSize,
                    weight,
                    {splitHeight, curWidth},
                    output[OffLen(i * (d0 * d1 * d2 * widthSplit),
                                  d0 * d1 * d2 * curWidth)],
                    hi);

                // load next block from hbm
                if (i != widthCnt - 1)
                {
                    __Infra::HBM_TO_VMEM(inst2.inst.insts,
                                         hbmAvailableAddr +
                                             (i + 1) * splitHeight * widthSplit,
                                         weight.startAddr,
                                         splitHeight * widthSplit);
                }
            }

            inst2.FreeVMem(&weight);
        }

        // input is already saved into hbm during reserve vmem
        inst2.FreeVMem(&input);

        bool moveOutputToVMemStart =
            inst2.resource.BiggestContinuousAvailableVMemSize(128) <
            d0 * d1 * d2 * w1;

        if (moveOutputToVMemStart)
        {
            inst2.PushResource();
            Memcopy(output, VMem(0, output.len, true, &inst2));
            inst2.Alloc(output.len, "Moved Output");
            auto oldOutputAddr = output.startAddr;
            output.startAddr = 0;
        }

        // reform output
        VMem Toutput =
            inst2.AllocF(d0 * d1 * d2 * w1, 128, "Output For Reform");

        for (int i = 0; i < widthCnt; i++)
        {
            for (int d = 0; d < d0 * d1; d++)
            {
                uint32_t curWidth = widthSplit;
                if (i == widthCnt - 1 && w1 % widthSplit != 0)
                {
                    curWidth = w1 % widthSplit;
                }
                auto blockSize = d2 * widthSplit;
                auto curBlockSize = d2 * curWidth;
                auto batch = d0 * d1;
                __Infra::MemcopyEx128(
                    inst2,
                    output[OffLen(i * batch * blockSize + d * curBlockSize,
                                  curBlockSize)],
                    0,
                    curWidth,
                    Toutput[OffLen(d * d2 * w1, d2 * w1)],
                    i * widthSplit,
                    w1,
                    curWidth,
                    d2);
            }
        }

        if (moveOutputToVMemStart)
        {
            inst2.PopResource();
        }

        if (needReserveVMemToHbm)
        {
            vmemOutputAddr = Toutput.startAddr;
        }
        else
        {
            Memcopy(Toutput, output);
        }
    }
    else
    {
        __Linear::LinearInternalSplitHBMWeight(inst2,
                                               splitHeight,
                                               splitWidth,
                                               input,
                                               inputSize,
                                               hbmWeightAddr,
                                               weightSize,
                                               output);
    }

    if (needReserveVMemToHbm)
    {
        __Infra::VMEM_TO_HBM(inst2.inst.insts,
                             vmemOutputAddr,
                             vmemHbmAddr + oldOutputVMemAddr,
                             outputDataSize);
        __Infra::HBM_TO_VMEM(inst2.inst.insts, vmemHbmAddr, 0, savedVMemSize);
        inst2.PopResource();
    }
    inst2.PopResource();
}

void
LinearExVMemIO(std::vector<Instruction *> &instList,
               const std::vector<uint32_t> &reservedVReg,
               const std::vector<uint32_t> &reservedSReg,
               const std::vector<uint32_t> &reservedVMask,
               const std::vector<uint32_t> &reservedPermit,
               uint32_t splitHeight,
               uint32_t splitWidth,
               uint32_t vmemInputAddr,
               const std::array<uint32_t, 4> &inputSize,
               uint32_t hbmWeightAddr,
               const std::array<uint32_t, 2> &weightSize,
               uint32_t vmemOutputAddr)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    assert(w1 % splitWidth == 0);
    assert(splitWidth % 128 == 0);
    assert(w0 % 128 == 0);

    assert(w0 == d3);
    assert(d3 % splitHeight == 0);

    const auto outputDataSize = d0 * d1 * d2 * w1;

    Inst2 inst2;
    inst2.resource.AllocVMem(vmemOutputAddr + outputDataSize);
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }

    InstVer::LinearExVMemIOInst(inst2,
                                splitHeight,
                                splitWidth,
                                vmemInputAddr,
                                inputSize,
                                hbmWeightAddr,
                                weightSize,
                                vmemOutputAddr);

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

/*

because input has batch, so first

+--+ < input batch 0
+--+ < input batch 0 strip 1     +--+
|  |                          => +--+ < input batch 0 strip 1
+--+ < input batch 1             +--+ < input batch 1 strip 1
+--+ < input batch 1 strip 1
|  |
+--+

v-d3--v    v--v--widthSplit
+--+--+    +--+<+
|00|01|    |00| |
+--+--+ => +--+ + d0 * d1 * spiltHeight
|10|11|    |10| |
+--+--+    +--+<+

number is block, alpha is batch
!!Attention, this not mean out of this comment it's still this meaning

batch = d0 * d1
block = d2 / splitHeight

   v after linear
+--+    +--+ < output required
|1A|    |1A|
+--+    +--+
|1B|    |2A|
+--+    +--+
|1C|    |1B|
+--+ => +--+
|2A|    |2B|
+--+    +--+
|2B|    |1C|
+--+    +--+
|2C|    |2C| < a block size = splitHeight * w1
+--+    +--+

*/

void
LinearExHiVwVo(std::vector<Instruction *> &instList,
               const std::vector<uint32_t> &reservedVReg,
               const std::vector<uint32_t> &reservedSReg,
               const std::vector<uint32_t> &reservedVMask,
               const std::vector<uint32_t> &reservedPermit,
               uint32_t splitHeight,
               uint32_t splitWidth,
               uint32_t hbmInputAddr,
               const std::array<uint32_t, 4> &inputSize,
               uint32_t vmemWeightAddr,
               const std::array<uint32_t, 2> &weightSize,
               uint32_t vmemOutputAddr)
{
    auto d0 = inputSize[0];
    auto d1 = inputSize[1];
    auto d2 = inputSize[2];
    auto d3 = inputSize[3];
    auto w0 = weightSize[0];
    auto w1 = weightSize[1];

    assert(d3 % splitWidth == 0);
    assert(splitWidth % 128 == 0);
    // assert(w0 % 128 == 0);

    assert(w0 == d3);
    assert(d2 % splitHeight == 0);
    assert(splitHeight % 128 == 0);

    const int callCnt = CallCount(__FUNCTION__);

    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: LinearExHiVwVo#" << callCnt
                  << "(@" << hbmInputAddr << "[" << d0 << ", " << d1 << ", "
                  << d2 << ", " << d3 << "], @" << vmemWeightAddr << "[" << w0
                  << ", " << w1 << "]) => @" << vmemOutputAddr << COLOR::WHITE
                  << std::endl;
    }

    Inst2 inst2;

    const auto w1_128 = ((w1 + 127) / 128) * 128;

    // output size with 128-aligned w1
    const auto outputDataSize = d0 * d1 * d2 * w1_128;
    inst2.resource.AllocVMem(vmemOutputAddr + outputDataSize);

    auto requiredVMem =
        __Linear::LinearInternalSplitRequiredSizeMeasure(splitWidth,
                                                         w1_128,
                                                         inputSize,
                                                         {splitWidth, w1_128}) +
        d0 * d1 * splitHeight * d3;

    auto availableVMem = inst2.resource.BiggestContinuousAvailableVMemSize(128);

    bool needReserveVMemToHbm = availableVMem < requiredVMem;

    uint32_t hbmAvailableAddr = HBMAddr();

    const auto oldOutputVMemAddr = vmemOutputAddr;
    const auto oldWeightVMemAddr = vmemWeightAddr;
    const auto savedVMemSize = vmemOutputAddr;
    uint32_t hbmVMemAddr = 0;

    bool storeWeight = false;
    uint32_t hbmWeightAddr = 0;

    if (needReserveVMemToHbm)
    {
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::AYUMU << "RESERVE VMEM TO HBM\n"
                      << COLOR::WHITE;
        }
        __Infra::VMEM_TO_HBM(inst2.inst.insts,
                             0,
                             hbmAvailableAddr,
                             savedVMemSize);
        hbmVMemAddr = hbmAvailableAddr;
        hbmAvailableAddr += savedVMemSize;
        inst2.resource.FreeVMem(0, vmemOutputAddr + outputDataSize);
        inst2.resource.AllocVMem(w0 * w1 + outputDataSize);

        auto requiredVMemLeast =
            __Linear::LinearInternalSplitRequiredSizeMeasure(
                splitWidth,
                128,
                {d0, d1, splitHeight, 128},
                {128, w1_128}) +
            d0 * d1 * splitHeight * d3;

        storeWeight = requiredVMemLeast >
                      inst2.resource.BiggestContinuousAvailableVMemSize(128);

        if (storeWeight)
        {
            if (ShowFuncCallInfo())
            {
                std::clog << COLOR::AYUMU << "STORE WEIGHT INTO HBM\n"
                          << COLOR::WHITE;
            }
            // weight is already saved when saving vmem
            hbmWeightAddr = hbmVMemAddr + vmemWeightAddr;

            vmemWeightAddr = 0;
            vmemOutputAddr = 0;
            inst2.resource.FreeVMem(0, w0 * w1 + outputDataSize);
            inst2.resource.AllocVMem(outputDataSize);
        }
        else
        {
            Memcopy(VMem(vmemWeightAddr, w0 * w1, true, &inst2),
                    VMem(0, w0 * w1, true, &inst2));
            vmemWeightAddr = 0;
            vmemOutputAddr = w0 * w1;
        }
    }

    VMem weight(vmemWeightAddr, w0 * w1, true, &inst2);
    VMem output(vmemOutputAddr, outputDataSize, true, &inst2);

    if (requiredVMem > inst2.resource.BiggestContinuousAvailableVMemSize(128) ||
        storeWeight || w1 != w1_128)
    {
        auto splitIndex =
            inst2.resource.BiggestContinuousAvailableVMemSize(128) / 4 /
            (d0 * d1 * splitHeight);
        splitIndex = std::min(splitIndex, d3);
        splitIndex = __Utils::flp2(splitIndex);
        while (d3 % splitIndex != 0)
        {
            splitIndex /= 2;
        }
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::AYUMU << "SPLIT HBM INPUT WIDTH: " << splitIndex
                      << "\n"
                      << COLOR::WHITE;
        }
        uint32_t widthSplit = splitIndex;
        int widthCnt = ((d3 + widthSplit - 1) / widthSplit);

        const uint32_t inputSplitRequiredVMem =
            d0 * d1 * splitHeight * widthSplit + d0 * d1 * splitHeight * d3;

        if (inputSplitRequiredVMem >
            inst2.resource.BiggestContinuousAvailableVMemSize(128))
        {
            splitIndex = 128;
            if (ShowFuncCallInfo())
            {
                std::clog << COLOR::AYUMU
                          << "RE-CHOOSE SPLIT HBM INPUT WIDTH: " << splitIndex
                          << "\n"
                          << COLOR::WHITE;
            }
            widthSplit = splitIndex;
            widthCnt = ((d3 + widthSplit - 1) / widthSplit);
        }

        for (int hi = 0; hi < d2 / splitHeight; hi++)
        {
            // this func is for calc W^T * I^T , so d0 * d1 actually always 0
            // for max case [512, 32128] x [32128, 128]
            // output alloced 512 * 128 = 65536, rest 4128768 > 128 * 32128 +
            // 128 * 128 = 4112512 so we no need deal with save output temporary
            VMem inputBlock = inst2.Alloc(d0 * d1 * splitHeight * widthSplit,
                                          "LExHiVwVo InputBlock");

            // full strip is free first, so alloc last to get biggest continous
            // vmem
            VMem inputFullStrip = inst2.AllocF(d0 * d1 * splitHeight * d3,
                                               "LExHiVwVo InputFullStrip");

            for (int b = 0; b < d0 * d1; b++)
            {
                __Infra::HBM_TO_VMEM(
                    inst2.inst.insts,
                    hbmInputAddr + b * d2 * d3 + hi * splitHeight * d3,
                    inputFullStrip.startAddr + b * splitHeight * d3,
                    splitHeight * d3);
            }

            for (int i = 1; i < widthCnt; i++)
            {
                __Infra::MemcopyEx128(inst2,
                                      inputFullStrip,
                                      i * widthSplit,
                                      d3,
                                      inputBlock,
                                      0,
                                      widthSplit,
                                      widthSplit,
                                      d0 * d1 * splitHeight);

                __Infra::VMEM_TO_HBM(inst2.inst.insts,
                                     inputBlock.startAddr,
                                     hbmAvailableAddr + i * inputBlock.len,
                                     inputBlock.len);
            }

            // first block
            __Infra::MemcopyEx128(inst2,
                                  inputFullStrip,
                                  0,
                                  d3,
                                  inputBlock,
                                  0,
                                  widthSplit,
                                  widthSplit,
                                  d0 * d1 * splitHeight);

            // free full strip
            inst2.FreeVMem(&inputFullStrip);

            VMem tempOut = inst2.AllocF(d0 * d1 * splitHeight * w1_128,
                                        "LExHiVwVo TempOutput");
            VMem curOut = output[OffLen(hi * (d0 * d1 * splitHeight * w1_128),
                                        d0 * d1 * splitHeight * w1_128)];

            for (int i = 0; i < widthCnt; i++)
            {

                uint32_t curWidth = widthSplit;
                if (i == widthCnt - 1 && d3 % widthSplit != 0)
                {
                    curWidth = d3 % widthSplit;
                }

                VMem curWeight =
                    weight[OffLen(i * widthSplit * w1, curWidth * w1)];
                if (storeWeight)
                {
                    curWeight =
                        inst2.AllocF(curWidth * w1, "LExHiVwVo WeightBlock");
                    __Infra::HBM_TO_VMEM(inst2.inst.insts,
                                         hbmWeightAddr + i * widthSplit * w1,
                                         curWeight.startAddr,
                                         curWidth * w1);
                }

                if (w1 != w1_128)
                {
                    VMem calcWeight =
                        inst2.AllocF(curWidth * w1_128,
                                     "LExHiVwVo WeightBlockPadding");
                    calcWeight = 0.0f;
                    __Infra::MemcopyEx(inst2,
                                       curWeight,
                                       0,
                                       w1,
                                       calcWeight,
                                       0,
                                       w1_128,
                                       w1,
                                       curWidth);
                    if (storeWeight)
                    {
                        inst2.FreeVMem(&curWeight);
                    }
                    curWeight = std::move(calcWeight);
                }

                __Linear::LinearInternalSplit_OwO_(
                    inst2,
                    splitHeight,
                    splitWidth,
                    inputBlock,
                    {d0, d1, splitHeight, curWidth},
                    curWeight,
                    {curWidth, w1_128},
                    i == 0 ? curOut : tempOut);

                if (storeWeight || w1 != w1_128)
                {
                    inst2.FreeVMem(&curWeight);
                }

                if (i != 0)
                {
                    curOut += tempOut;
                }

                // load next block from hbm
                if (i != widthCnt - 1)
                {
                    __Infra::HBM_TO_VMEM(inst2.inst.insts,
                                         hbmAvailableAddr +
                                             (i + 1) * inputBlock.len,
                                         inputBlock.startAddr,
                                         inputBlock.len);
                }
            }

            inst2.FreeVMem(&tempOut);
            inst2.FreeVMem(&inputBlock);
        }

        // reform output
        VMem Toutput =
            inst2.AllocF(d0 * d1 * d2 * w1_128, 128, "LExHiVwVo Reform Output");

        // Memcopy(output, Toutput);

        for (int i = 0; i < d2 / splitHeight; i++)
        {
            for (int d = 0; d < d0 * d1; d++)
            {
                Memcopy(output[OffLen((i * d0 * d1 + d) * splitHeight * w1_128,
                                      splitHeight * w1_128)],
                        Toutput[OffLen((d * d2 / splitHeight + i) *
                                           splitHeight * w1_128,
                                       splitHeight * w1_128)]);
            }
        }
        if (w1 != w1_128)
        {
            // we need padding, so it always do depadding
            __Infra::MemcopyEx(inst2,
                               Toutput,
                               0,
                               w1_128,
                               output,
                               0,
                               w1,
                               w1,
                               d0 * d1 * d2);
        }
        else
        {
            if (needReserveVMemToHbm)
            {
                vmemOutputAddr = Toutput.startAddr;
            }
            else
            {
                Memcopy(Toutput, output);
            }
        }
    }
    else
    {
        __Linear::LinearInternalSplitHbmInput(inst2,
                                              splitHeight,
                                              splitWidth,
                                              hbmInputAddr,
                                              inputSize,
                                              weight,
                                              weightSize,
                                              output);
    }

    if (needReserveVMemToHbm)
    {
        Memcopy(VMem(vmemOutputAddr, d0 * d1 * d2 * w1, true, &inst2),
                VMem(oldOutputVMemAddr, d0 * d1 * d2 * w1, true, &inst2));

        if (storeWeight && w1 != w1_128)
        {
            // by test, in this case some data will lost after dma
            // so add noop to delay dma to assure memcopy finish as much as we
            // can
            for (int i = 0; i < 1000; i++)
            {
                inst2(Noop);
            }
        }

        __Infra::HBM_TO_VMEM(inst2.inst.insts, hbmVMemAddr, 0, savedVMemSize);
        // no need to load, weight is already in reserved vmem
    }

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
InstVer::MatMulInst(Inst2 &inst2,
                    uint32_t leftMatAddr,
                    const std::array<uint32_t, 4> &leftMatSize,
                    uint32_t rightMatAddr,
                    const std::array<uint32_t, 4> &rightMatSize,
                    uint32_t outputAddr)
{
    auto d0 = leftMatSize[0];
    auto d1 = leftMatSize[1];
    auto d2 = leftMatSize[2];
    auto d3 = leftMatSize[3];
    auto w0 = rightMatSize[0];
    auto w1 = rightMatSize[1];
    auto w2 = rightMatSize[2];
    auto w3 = rightMatSize[3];

    const int callCnt = CallCount(__FUNCTION__);
    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: MatMul#" << callCnt << "(@"
                  << leftMatAddr << "[" << d0 << ", " << d1 << ", " << d2
                  << ", " << d3 << "], @" << rightMatAddr << "[" << w0 << ", "
                  << w1 << ", " << w2 << ", " << w3 << "]) => @" << outputAddr
                  << COLOR::WHITE << std::endl;
    }

    uint32_t inputDataSize = std::accumulate(leftMatSize.begin(),
                                             leftMatSize.end(),
                                             1u,
                                             std::multiplies<uint32_t>());
    uint32_t weightDataSize = std::accumulate(rightMatSize.begin(),
                                              rightMatSize.end(),
                                              1u,
                                              std::multiplies<uint32_t>());

    VMem input(leftMatAddr, inputDataSize, true, &inst2);
    VMem weight(rightMatAddr, weightDataSize, true, &inst2);
    VMem output(outputAddr, d0 * d1 * d2 * w3, true, &inst2);

    auto alignedD3 = ((d3 + 127) / 128) * 128;
    auto alignedW3 = ((w3 + 127) / 128) * 128;

    auto requireVMem =
        __Linear::LinearInternalRequiredSizeMeasure({1, 1, d2, alignedD3},
                                                    {alignedD3, alignedW3}) +
        d2 * alignedD3 + alignedD3 * alignedW3 + d2 * alignedW3;

    bool saveVmem =
        inst2.resource.BiggestContinuousAvailableVMemSize(128) < requireVMem;

    uint32_t vmemHbmAddr = HBMAddr();
    if (saveVmem)
    {
        __Infra::VMEM_TO_HBM(inst2.inst.insts,
                             0,
                             vmemHbmAddr,
                             GlobalDeviceConfig().VMemSize);
        inst2.PushResource();
        HBMAddr() += GlobalDeviceConfig().VMemSize;
        input = inst2.AllocF(inputDataSize, "MatMul Replaced Input");
        weight = inst2.AllocF(weightDataSize, "MatMul Replaced Weight");
        output = inst2.AllocF(d0 * d1 * d2 * w3, "MatMul Replaced Output");
        __Infra::HBM_TO_VMEM(inst2.inst.insts,
                             vmemHbmAddr + leftMatAddr,
                             input.startAddr,
                             input.len);
        __Infra::HBM_TO_VMEM(inst2.inst.insts,
                             vmemHbmAddr + rightMatAddr,
                             weight.startAddr,
                             weight.len);
    }

    VMem vInput = inst2.Alloc(d2 * alignedD3, "MatMul Temp Input");
    VMem vWeight = inst2.Alloc(alignedD3 * alignedW3, "MatMul Temp Weight");
    VMem vOutput = inst2.Alloc(d2 * alignedW3, "MatMul Temp Output");

    vInput = 0.0f;
    vWeight = 0.0f;

    requireVMem =
        __Linear::LinearInternalRequiredSizeMeasure({1, 1, d2, alignedD3},
                                                    {alignedD3, alignedW3});

    auto availableVMem = inst2.resource.BiggestContinuousAvailableVMemSize(128);

    bool requireSplit = requireVMem > availableVMem;

    unsigned int spH = 128;
    unsigned int spW = 128;
    bool sw = true;

    while (__Linear::LinearInternalSplitRequiredSizeMeasure(
               spH,
               spW,
               {1, 1, d2, alignedD3},
               {spW, alignedW3}) < availableVMem)
    {
        if (sw)
        {
            spW *= 2;
        }
        else
        {
            spH *= 2;
        }
        sw = !sw;
    }

    if (spH != 128 && spW != 128)
    {
        if (!sw)
        {
            spW /= 2;
        }
        else
        {
            spH /= 2;
        }
    }

    spH = std::min(alignedD3, spH);
    spW = std::min(alignedW3, spW);

    if (ShowFuncCallInfo() && requireSplit)
    {
        std::clog << COLOR::YELLOW << "SPLIT CHOOSE: " << spH << " X " << spW
                  << "\n"
                  << COLOR::WHITE;
    }

    for (int i = 0; i < d0 * d1; i++)
    {
        __Infra::MemcopyEx(inst2,
                           input[Range(i * d2 * d3, (i + 1) * d2 * d3)],
                           0,
                           d3,
                           vInput,
                           0,
                           alignedD3,
                           d3,
                           d2);
        __Infra::MemcopyEx(inst2,
                           weight[Range(i * w2 * w3, (i + 1) * w2 * w3)],
                           0,
                           w3,
                           vWeight,
                           0,
                           alignedW3,
                           w3,
                           w2);
        if (requireSplit)
        {
            bool useLinearEx =
                __Linear::LinearInternalSplitRequiredSizeMeasure(
                    spH,
                    spW,
                    {1, 1, d2, alignedD3},
                    {spW, alignedW3}) >
                inst2.resource.BiggestContinuousAvailableVMemSize(128);
            if (!useLinearEx)
            {
                for (int i = 0; i < alignedD3 / spH; i++)
                {
                    __Linear::LinearInternalSplit(
                        inst2,
                        spH,
                        spW,
                        vInput,
                        {1, 1, d2, alignedD3},
                        vWeight[OffLen(i * spH * alignedW3, spH * alignedW3)],
                        {spH, alignedW3},
                        vOutput,
                        i);
                }
            }
            else
            {
                uint64_t weightHbmAddr = HBMAddr();
                HBMAddr() += vWeight.len;
                __Infra::VMEM_TO_HBM(inst2.inst.insts,
                                     vWeight.startAddr,
                                     weightHbmAddr,
                                     vWeight.len);
                LinearExVMemIOInst(inst2,
                                   spH,
                                   spW,
                                   vInput.startAddr,
                                   {1, 1, d2, alignedD3},
                                   weightHbmAddr,
                                   {alignedD3, alignedW3},
                                   vOutput.startAddr);
                HBMAddr() = weightHbmAddr;
            }
        }
        else
        {
            __Linear::LinearInternal(inst2,
                                     vInput,
                                     {1, 1, d2, alignedD3},
                                     vWeight,
                                     {alignedD3, alignedW3},
                                     vOutput);
        }
        __Infra::MemcopyEx(inst2,
                           vOutput,
                           0,
                           alignedW3,
                           output[Range(i * d2 * w3, (i + 1) * d2 * w3)],
                           0,
                           w3,
                           w3,
                           d2);
    }

    inst2.FreeVMem(&vInput);
    inst2.FreeVMem(&vWeight);
    inst2.FreeVMem(&vOutput);

    if (saveVmem)
    {
        __Infra::VMEM_TO_HBM(inst2.inst.insts,
                             output.startAddr,
                             vmemHbmAddr + outputAddr,
                             output.len);
        __Infra::HBM_TO_VMEM(inst2.inst.insts,
                             vmemHbmAddr,
                             0,
                             kVectorDataMemorySize);
        inst2.PopResource();
        HBMAddr() = vmemHbmAddr;
    }
}

void
MatMul(std::vector<Instruction *> &instList,
       const std::vector<uint32_t> &reservedVReg,
       const std::vector<uint32_t> &reservedSReg,
       const std::vector<uint32_t> &reservedVMask,
       const std::vector<uint32_t> &reservedPermit,
       uint32_t leftMatAddr,
       const std::array<uint32_t, 4> &leftMatSize,
       uint32_t rightMatAddr,
       const std::array<uint32_t, 4> &rightMatSize,
       uint32_t outputAddr)
{
    auto d0 = leftMatSize[0];
    auto d1 = leftMatSize[1];
    auto d2 = leftMatSize[2];
    auto d3 = leftMatSize[3];
    auto w0 = rightMatSize[0];
    auto w1 = rightMatSize[1];
    auto w2 = rightMatSize[2];
    auto w3 = rightMatSize[3];

    assert(d0 == w0);
    assert(d1 == w1);

    assert(w2 == d3);

    // assert(d3 <= 128);
    // assert(w2 <= 128);
    // assert(w3 <= 128);

    Inst2 inst2;
    inst2.resource.AllocVMem(outputAddr + d0 * d1 *
                                              __Linear::AlignTo128Bytes(d2) *
                                              __Linear::AlignTo128Bytes(w3));
    for (const auto &vReg : reservedVReg)
    {
        assert(vReg < 32);
        inst2.resource.vReg[vReg] = true;
    }
    for (const auto &sReg : reservedSReg)
    {
        assert(sReg < 32);
        inst2.resource.sReg[sReg] = true;
    }
    
    for (const auto &vMask : reservedVMask)
    {
        assert(vMask < 8);
        inst2.resource.vMask[vMask] = true;
    }

    InstVer::MatMulInst(inst2,
                        leftMatAddr,
                        leftMatSize,
                        rightMatAddr,
                        rightMatSize,
                        outputAddr);

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}




void
InstVer::EmbeddingExInst(Inst2 &inst2,
                         uint32_t hbmEmbeddingAddr,
                         uint32_t embeddingHeight,
                         uint32_t embeddingWidth,
                         uint32_t indicesAddr,
                         uint32_t indicesSize,
                         uint32_t outputAddr,
                         uint32_t weight_addr)
{
    const int callCnt = CallCount(__FUNCTION__);

    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: EmbeddingEx#" << callCnt << "(@"
                  << hbmEmbeddingAddr << "[" << embeddingHeight << " x "
                  << embeddingWidth << "], @" << indicesAddr << "["
                  << indicesSize << "]) => @" << outputAddr << COLOR::WHITE
                  << std::endl;
    }

    auto size = embeddingHeight * embeddingWidth;
    VMem indices(indicesAddr, indicesSize, false, &inst2);
    VMem output(outputAddr, indicesSize * embeddingWidth, false, &inst2);

    int i = 0;
    // for (; i < indicesSize / 1024; i++)
    // {

    //     __InternalImpl::EmbeddingEx(
    //         inst2,
    //         hbmEmbeddingAddr,
    //         embeddingHeight,
    //         embeddingWidth,
    //         indices[Range(i * 1024, (i + 1) * 1024)],
    //         1024,
    //         output[Range(i * 1024 * embeddingWidth,
    //                      (i + 1) * 1024 * embeddingWidth)],
    //         weight_addr);
    // }

    if (indicesSize % 1024 != 0)
    {
        __InternalImpl::EmbeddingEx(
            inst2,
            hbmEmbeddingAddr,
            embeddingHeight,
            embeddingWidth,
            indices[Range(i * 1024, indicesSize)],
            indicesSize % 1024,
            // output[Range(i * 1024 * embeddingWidth,
            //              indicesSize * embeddingWidth)]);
            output[Range(outputAddr, indicesSize * embeddingWidth)],
            weight_addr);
    }
}

void
InstVer::Embeddings(Inst2 &inst2,
                    uint32_t hbmEmbeddingAddr,
                    uint32_t embeddingHeight,
                    uint32_t embeddingWidth,
                    int input_addr,
                    int inputSize,
                    int output_addr,
                    int weight_input_addr)
{
    const int callCnt = CallCount(__FUNCTION__);

    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: Embeddings#" << callCnt << "(@"
                  << hbmEmbeddingAddr << "[" << embeddingHeight << " x "
                  << embeddingWidth << "], @" << input_addr << "["
                  << inputSize << "]) => @" << output_addr << COLOR::WHITE
                  << std::endl;
    }

    __InternalImpl::Embeddings(inst2,
                        hbmEmbeddingAddr,
                        embeddingHeight,
                        embeddingWidth,
                        input_addr,
                        inputSize,
                        output_addr,
                        weight_input_addr);
}

// void
// EmbeddingEx(std::vector<Instruction *> &instList,
//             const std::vector<uint32_t> &reservedVReg,
//             const std::vector<uint32_t> &reservedSReg,
//             const std::vector<uint32_t> &reservedVMask,
//             const std::vector<uint32_t> &reservedPermit,
//             uint32_t hbmEmbeddingAddr,
//             uint32_t embeddingHeight,
//             uint32_t embeddingWidth,
//             uint32_t indicesAddr,
//             uint32_t indicesSize,
//             uint32_t outputAddr)
// {
//     Inst2 inst2;
//     for (const auto &vReg : reservedVReg)
//     {
//         assert(vReg < 32);
//         inst2.resource.vReg[vReg] = true;
//     }
//     for (const auto &sReg : reservedSReg)
//     {
//         assert(sReg < 32);
//         inst2.resource.sReg[sReg] = true;
//     }
//     for (const auto &vMask : reservedVMask)
//     {
//         assert(vMask < 8);
//         inst2.resource.vMask[vMask] = true;
//     }

//     inst2.Alloc(outputAddr + indicesSize * embeddingWidth,
//                 "Reserved Output Data");

//     InstVer::EmbeddingExInst(inst2,
//                              hbmEmbeddingAddr,
//                              embeddingHeight,
//                              embeddingWidth,
//                              indicesAddr,
//                              indicesSize,
//                              outputAddr);

//     std::copy(inst2.inst.insts.begin(),
//               inst2.inst.insts.end(),
//               std::back_inserter(instList));
// }

void
Softmax(std::vector<Instruction *> &instList,
        uint32_t input_addr,
        uint32_t output_addr,
        uint32_t num,
        uint32_t size)
{
    const int callCnt = CallCount(__FUNCTION__);

    if (ShowFuncCallInfo())
    {
        std::clog << COLOR::SHIORI << "FnCall: Softmax#" << callCnt << "(@"
                  << input_addr << "[" << num << " x " << size << "]) => @"
                  << output_addr << COLOR::WHITE << std::endl;
    }

    Inst2 inst2;

    inst2.Alloc(output_addr + num * size, "Softmax Reserved VMem");

    VMem in(input_addr, num * size, true, &inst2);
    VMem out(output_addr, num * size, true, &inst2);

    for (int i = 0; i < num; i++)
    {
        __InternalImpl::SoftmaxSingle(in[OffLen(i * size, size)],
                                      out[OffLen(i * size, size)]);
    }

    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
Dma(std::vector<Instruction *> &instList,
    uint16_t misc,
    uint32_t src_addr,
    uint32_t dest_addr,
    uint32_t length)
{
    const int callCnt = CallCount(__FUNCTION__);

    bool safeCall = src_addr % 128 == 0 && dest_addr % 128 == 0;
    if (ShowFuncCallInfo() || (!safeCall))
    {
        std::clog << (safeCall ? COLOR::SHIORI : COLOR::SETSUNA)
                  << "FnCall: Dma#" << callCnt << "(@" << src_addr << "["
                  << length << "]) => @" << dest_addr << COLOR::WHITE
                  << std::endl;
    }

    assert(src_addr % 128 == 0);
    assert(dest_addr % 128 == 0);
    // assert(length % 128 == 0);

    if (length % 128 != 0)
    {
        length = ((length + 127) / 128) * 128;
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON << "LENGTH UP-ALIGN TO 128: " << length
                      << COLOR::WHITE << std::endl;
        }
    }
    else if (length == 0)
    {
        std::clog << COLOR::ORANGE << "SKIP ZERO-LENGTH DMA" << COLOR::WHITE
                  << std::endl;
        return;
    }

    int sync_register = 0;
    Inst2 inst2(instList);

    inst2(SMov, inst2.inst.ImmeU(src_addr / 128), 6);
    inst2(SMov, inst2.inst.ImmeU(dest_addr / 128), 7);
    inst2(SMov, inst2.inst.ImmeU(length / 128), 8);
    inst2(MSetSyncFlag, 1 + sync_register, (int)MSetSyncOp::ClrDone, 0);

    if (1)
    {
        Instruction *inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                16384 + 1 + sync_register);
        ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 6, 8, 7, 33, misc);
        inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);
        __Infra::CompleteInstruction(inst);
        instList.push_back(inst);
    }

    inst2(MSync, 1 + sync_register, (int)MSyncOp::SetDone, 0);
    inst2(Fence);
    inst2(Noop);
}

void
DmaNonBlock(std::vector<Instruction *> &instList,
            uint16_t misc,
            uint32_t src_addr,
            uint32_t dest_addr,
            uint32_t length)
{
    const int callCnt = CallCount(__FUNCTION__);

    bool safeCall = src_addr % 128 == 0 && dest_addr % 128 == 0;
    if (ShowFuncCallInfo() || (!safeCall))
    {
        std::clog << (safeCall ? COLOR::SHIORI : COLOR::SETSUNA)
                  << "FnCall: Dma#" << callCnt << "(@" << src_addr << "["
                  << length << "]) => @" << dest_addr << COLOR::WHITE
                  << std::endl;
    }

    assert(src_addr % 128 == 0);
    assert(dest_addr % 128 == 0);
    // assert(length % 128 == 0);

    if (length % 128 != 0)
    {
        length = ((length + 127) / 128) * 128;
        if (ShowFuncCallInfo())
        {
            std::clog << COLOR::KANON << "LENGTH UP-ALIGN TO 128: " << length
                      << COLOR::WHITE << std::endl;
        }
    }
    else if (length == 0)
    {
        std::clog << COLOR::ORANGE << "SKIP ZERO-LENGTH DMA" << COLOR::WHITE
                  << std::endl;
        return;
    }

    int sync_register = 0;
    Inst2 inst2(instList);

    inst2(SMov, inst2.inst.ImmeU(src_addr / 128), 6);
    inst2(SMov, inst2.inst.ImmeU(dest_addr / 128), 7);
    inst2(SMov, inst2.inst.ImmeU(length / 128), 8);

    if (1)
    {
        Instruction *inst = new Instruction();
        inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                16384 + 1 + sync_register);
        ScalarOperationState dma_local_1(S_LOCAL_DMA, 0, 6, 8, 7, 33, misc);
        inst->SetOperationState(Instruction::SCALARONE, &dma_local_1);
        __Infra::CompleteInstruction(inst);
        instList.push_back(inst);
    }

    inst2(Noop);
}

void
Padding(std::vector<Instruction *> &instList,
        uint32_t srcAddr,
        uint32_t width,
        uint32_t destAddr,
        uint32_t newWidth,
        uint32_t height)
{
    Inst2 inst2;
    __Infra::MemcopyEx(inst2,
                       VMem(srcAddr, width * height, true, &inst2),
                       0,
                       width,
                       VMem(destAddr, newWidth * height, true, &inst2),
                       0,
                       newWidth,
                       std::min(width, newWidth),
                       height);
    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}

void
DePadding(std::vector<Instruction *> &instList,
          uint32_t srcAddr,
          uint32_t width,
          uint32_t destAddr,
          uint32_t newWidth,
          uint32_t height)
{
    Inst2 inst2;
    __Infra::MemcopyEx(inst2,
                       VMem(srcAddr, width * height, true, &inst2),
                       0,
                       width,
                       VMem(destAddr, newWidth * height, true, &inst2),
                       0,
                       newWidth,
                       std::min(width, newWidth),
                       height);
    std::copy(inst2.inst.insts.begin(),
              inst2.inst.insts.end(),
              std::back_inserter(instList));
}