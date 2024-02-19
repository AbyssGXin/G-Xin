
#ifndef __EXEC_HELPER_H__
#define __EXEC_HELPER_H__

#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifdef __linux__
#    include <dirent.h>
#    define VELOCE_AVAILABLE
#endif
#include <ctime>
#ifdef _WIN32
#    include <filesystem>
#endif

#include "../device/Devices/Device_Simulator.h"
#ifdef VELOCE_AVAILABLE
#    include "../device/Devices/Veloce.h"
#    include "../models_bringup/Testing_Cases.h"
#endif
#include "simple_test/test_helper.h"

#include "FuncHelper.h"
#include "InstHelper2.h"

// ret % k == 0 && ret >= v
template <class T, class K>
T
UpRound(T v, K k)
{
    return ((v + k - 1) / k) * k;
}

// ret % k == 0 && ret <= v
template <class T, class K>
T
DownRound(T v, K k)
{
    return (v / k) * k;
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

template struct Thief<ThiefHandle<Device_Simulator, Xys *>,
                      &Device_Simulator::_xys>;
template struct Thief<ThiefHandle<Device_Simulator, std::shared_ptr<Hbm>>,
                      &Device_Simulator::_hbm>;
template struct Thief<ThiefHandle<Xys, Processor>, &Xys::_vector_processor>;
template struct Thief<ThiefHandle<Xys, Processor, CaseId::SCALAR_PROCESSOR>,
                      &Xys::_scalar_processor>;
template struct Thief<ThiefHandle<Processor, RegisterFile *>,
                      &Processor::_data_registerfile>;
template struct Thief<ThiefHandle<Processor, std::vector<ShadowBuffer>>,
                      &Processor::_transpose_buffer>;

namespace VeloceCSR
{
constexpr uint64_t PC = 0x02000010;
constexpr uint64_t RUN_STATUS = 0x02000008;
constexpr uint64_t SCALAR_ERROR = 0x020001b0;
constexpr uint64_t VECTOR_PROGRAM_ERROR = 0x02010048;
constexpr uint64_t VECTOR_FATAL_ERROR = 0x02010060;

inline uint64_t
VECTOR_CORE_ERROR(int8_t idx)
{
    return 0x02800000ull + (idx * 0x10000ull) + 0x0228ull;
}
const std::string SCALAR_ERRORS[] = {
    "bad_opcode_in_s0",
    "bad_opcode_in_s1",
    "bad_opcode_in_v0",
    "bad_opcode_in_v1",
    "bad_opcode_in_st",
    "bad_opcode_in_ld",
    "bad_opcode_in_mti",
    "bad_opcode_in_mtr",
    "bad_opcode_in_misc",
    "bad_opcode_in_s0s1",
    "bad_permission_dest",
    "bad_vmask_dest",
    "bad_imem_addr",
    "bad_smem_addr",
    "bad_smem_dma_addr",
    "sync_flag_overflow",
    "pop_v2s_empty",
    "push_v2s_full",
    "imem_0_ecc",
    "imem_0_ecc_fatal",
    "imem_1_ecc",
    "imem_1_ecc_fatal",
    "smem_0_ecc",
    "smem_0_ecc_fatal",
    "smem_1_ecc",
    "smem_1_ecc_fatal",
    "smem_2_ecc",
    "smem_2_ecc_fatal",
    "smem_3_ecc",
    "smem_3_ecc_fatal",
    "smem_4_ecc",
    "smem_4_ecc_fatal",
    "smem_5_ecc",
    "smem_5_ecc_fatal",
    "smem_6_ecc",
    "smem_6_ecc_fatal",
    "smem_7_ecc",
    "smem_7_ecc_fatal",
    "sync_mem_ecc",
    "sync_mem_ecc_fatal",
    "ins_log_ecc",
    "ins_log_ecc_fatal",
    "p_cnt_ecc",
    "p_cnt_ecc_fatal",
    "remote_sync_flag_overflow",
    "imem_dma_illegal_read",
    "bad_imem_dma_addr",
    "smem_segmentation_fault",
    "imem_segmentation_fault",
    "sfm_segmentation_fault",
};
constexpr size_t SCALAR_ERRORS_CNT =
    sizeof(SCALAR_ERRORS) / sizeof(std::string);
const std::string VECTOR_PROGRAM_ERRORS[] = {
    "bad_vld_base",
    "bad_vld_stride",
    "bad_vst_base",
    "bad_vst_stride",
    "bad_cld_base",
    "bad_cld_stride",
    "bad_cld_addr",
    "cld_bank_conf",
    "bad_cst_base",
    "bad_cst_stride",
    "bad_cst_addr",
    "cst_bank_conf",
};
constexpr size_t VECTOR_PROGRAM_ERRORS_CNT =
    sizeof(VECTOR_PROGRAM_ERRORS) / sizeof(std::string);
const std::string VECTOR_FATAL_ERRORS[] = {
    "mrf0_uf",
    "mrf0_of",
    "mrf1_uf",
    "mrf1_of",
    "trf0_uf",
    "trf0_of",
    "trf1_uf",
    "trf1_of",
    "urf_uf",
    "urf_of",
    "crf_uf",
    "crf_of",
    "transpose_interrupted",
    "transpose_missing_start",
    "transpose_missing_end",
    "transpose_wrong_start",
    "transpose_wrong_end",
    "transpose_bad_width",
    "transpose_type_change",
};
constexpr size_t VECTOR_FATAL_ERRORS_CNT =
    sizeof(VECTOR_FATAL_ERRORS) / sizeof(std::string);
const std::string VECTOR_CORE_ERRORS[] = {
    "mrf0_ecc",
    "mrf0_ecc_fatal",
    "mrf1_ecc",
    "mrf1_ecc_fatal",
    "trf0_ecc",
    "trf0_ecc_fatal",
    "trf1_ecc",
    "trf1_ecc_fatal",
    "urf_ecc",
    "urf_ecc_fatal",
    "crf_ecc",
    "crf_ecc_fatal",
    "vmem_bank0_ecc",
    "vmem_bank0_ecc_fatal",
    "vmem_bank1_ecc",
    "vmem_bank1_ecc_fatal",
    "vmem_bank2_ecc",
    "vmem_bank2_ecc_fatal",
    "vmem_bank3_ecc",
    "vmem_bank3_ecc_fatal",
    "vmem_bank4_ecc",
    "vmem_bank4_ecc_fatal",
    "vmem_bank5_ecc",
    "vmem_bank5_ecc_fatal",
    "vmem_bank6_ecc",
    "vmem_bank6_ecc_fatal",
    "vmem_bank7_ecc",
    "vmem_bank7_ecc_fatal",
    "bad_vld_addr",
    "vld_bank_conf",
    "bad_vst_addr",
    "vst_bank_conf",
    "dma_vmem_write_hazard",
    "dma_vmem_read_hazard",
};
constexpr size_t VECTOR_CORE_ERRORS_CNT =
    sizeof(VECTOR_CORE_ERRORS) / sizeof(std::string);
} // namespace VeloceCSR

#define NSEC_PER_SEC 1000000000

/** \fn struct timespec timespec_normalise(struct timespec ts)
 *  \brief Normalises a timespec structure.
 *
 * Returns a normalised version of a timespec structure, according to the
 * following rules:
 *
 * 1) If tv_nsec is >=1,000,000,00 or <=-1,000,000,000, flatten the surplus
 *    nanoseconds into the tv_sec field.
 *
 * 2) If tv_nsec is negative, decrement tv_sec and roll tv_nsec up to represent
 *    the same value attainable by ADDING nanoseconds to tv_sec.
 */
timespec
timespec_normalise(timespec ts)
{
    while (ts.tv_nsec >= NSEC_PER_SEC)
    {
        ++(ts.tv_sec);
        ts.tv_nsec -= NSEC_PER_SEC;
    }

    while (ts.tv_nsec <= -NSEC_PER_SEC)
    {
        --(ts.tv_sec);
        ts.tv_nsec += NSEC_PER_SEC;
    }

    if (ts.tv_nsec < 0)
    {
        /* Negative nanoseconds isn't valid according to POSIX.
         * Decrement tv_sec and roll tv_nsec over.
         */

        --(ts.tv_sec);
        ts.tv_nsec = (NSEC_PER_SEC + ts.tv_nsec);
    }

    return ts;
}

/** \fn struct timespec timespec_add(struct timespec ts1, struct timespec ts2)
 *  \brief Returns the result of adding two timespec structures.
 */
timespec
timespec_add(timespec ts1, timespec ts2)
{
    /* Normalise inputs to prevent tv_nsec rollover if whole-second values
     * are packed in it.
     */
    ts1 = timespec_normalise(ts1);
    ts2 = timespec_normalise(ts2);

    ts1.tv_sec += ts2.tv_sec;
    ts1.tv_nsec += ts2.tv_nsec;

    return timespec_normalise(ts1);
}

/** \fn struct timespec timespec_sub(struct timespec ts1, struct timespec ts2)
 *  \brief Returns the result of subtracting ts2 from ts1.
 */
timespec
timespec_sub(timespec ts1, timespec ts2)
{
    /* Normalise inputs to prevent tv_nsec rollover if whole-second values
     * are packed in it.
     */
    ts1 = timespec_normalise(ts1);
    ts2 = timespec_normalise(ts2);

    ts1.tv_sec -= ts2.tv_sec;
    ts1.tv_nsec -= ts2.tv_nsec;

    return timespec_normalise(ts1);
}

inline void
TimeAppend(timespec &dest, timespec start, timespec end)
{
    dest = timespec_add(dest, timespec_sub(end, start));
}

struct Runner;

struct FfiInfo
{
    Runner *ctx = nullptr;
    std::unordered_map<std::string,
                       std::function<void(Runner *, std::vector<uint32_t>)>>
        fns{};
};

inline FfiInfo &
GlobalFfiInfo()
{
    static FfiInfo info;
    return info;
}

inline void
FfiCall(std::string name, std::vector<uint32_t> args)
{
    FfiInfo &info = GlobalFfiInfo();
    if (info.ctx != nullptr)
    {
        info.fns[name](info.ctx, args);
    }
}

struct Runner
{
    enum class RunType
    {
        InstGen,
        Simulator,
#ifdef VELOCE_AVAILABLE
        Veloce,
#endif
        LateInit
    };

    struct PayLoad
    {
        RunType type;
        Device_Simulator *simulator = nullptr;

        explicit PayLoad(RunType type) : type(type)
        {
            switch (type)
            {
            case RunType::Simulator:
            {
                std::clog << COLOR::ORANGE << "Prepare Device::Simulator"
                          << COLOR::WHITE << std::endl;
                IKnowItIsRunningInSimulatorNotVeloceAndISureINeedFiveTransposeHack() =
                    true;
                GlobalDeviceConfig().VMemSize = 4096 * 1024;
                simulator = new Device_Simulator(Device::DEVICE_SIMULATOR);
                simulator->OpenDeviceWithHBM();
            }
            break;
#ifdef VELOCE_AVAILABLE
            case RunType::Veloce:
            {
                std::clog << COLOR::ORANGE << "Prepare Device::Veloce"
                          << COLOR::WHITE << std::endl;
            }
            break;
#endif
            case RunType::InstGen:
            {
                std::clog << COLOR::ORANGE << "Prepare Device::InstGen"
                          << COLOR::WHITE << std::endl;
            }
            break;
            default:
                break;
            }
        }

        ~PayLoad()
        {
            delete simulator;
        }

        PayLoad(const PayLoad &) = delete;
        PayLoad &operator=(const PayLoad &) = delete;

        PayLoad(PayLoad &&p) noexcept
        {
            this->type = p.type;
            this->simulator = p.simulator;
            p.simulator = nullptr;
        }

        PayLoad &
        operator=(PayLoad &&p) noexcept
        {
            if (this != &p)
            {
                this->type = p.type;
                this->simulator = p.simulator;
                p.simulator = nullptr;
            }
            return *this;
        }
    };

    PayLoad payload;

    // OPTIONS
    std::string logfile = "./EXEC.log";
    std::string dirbase = "";
    bool showDiff = false;
    bool checkErrorStatus = false;
    RunType runType = RunType::Simulator;
    bool showFuncCall = true;
    uint32_t readLccPerExec = 1000;
    bool doReadLcc = false;
    uint32_t bundleSize = 14000;
    bool skipCheckpoint = false;
    uint32_t skipExec = 0;
    bool useScheduler = false;
    bool showSchedulerVerboseInfo = false;

    // INTERNAL STATEMENTS
    bool requireClearErrorStatus = false;

    // INTERNAL CONST VAL
    const uint32_t lccInitSMemAddr = GlobalDeviceConfig().SMemSize / 2;

    // INTERNAL COUNT
    uint64_t instSize = 0;
    uint64_t execTotCnt = 0;
    uint64_t cycles = 0;
    std::vector<size_t> cycleLog = {};
    std::vector<size_t> instCountLog = {};

    uint64_t instSizeDivinandi = 0;
    uint64_t execTotCntDivinandi = 0;

    uint32_t lccSMemAddr = lccInitSMemAddr;

    uint32_t ToHttSize = 12 * 1024 * 128;
    uint32_t
    OutfeedSize()
    {
        return (ToHttSize / (kMultipleOfBytes / 4));
    }

    // TIME PERFORM

    enum STATE
    {
        S_GEN,
        S_BUNDLE,
        S_RESET,
        S_WRITE_IMEM,
        S_EXEC,
        S_CHECK,
        S_ASSERT,
        S_CLEAR,
        S_OTHER,
    };

    STATE curState = S_OTHER;
    timespec stateTot[9]{};
    timespec stateStart{};

    void
    StateStopWatchStart()
    {
#ifdef __linux__
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stateStart);
#endif
    }

    void
    SwitchState(STATE newState)
    {
#ifdef __linux__
        timespec end;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        STATE oldState = curState;
        TimeAppend(stateTot[oldState], stateStart, end);
        curState = newState;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stateStart);
#endif
    }

    explicit Runner(RunType type = RunType::LateInit) : payload(type) {}
    Runner(Runner &&) = default;
    virtual ~Runner() = default;
    virtual void Main(Inst2 &, std::vector<Instruction *> &) = 0;
    virtual void
    OnExecuted(Inst2 &)
    {
    }

    bool
    RegisterArgHelper(ArgHelper &arg)
    {
        using namespace ArgHelperUtils;
        bool res = true;
        res &= arg.reg("--log",
                       1,
                       Set(logfile),
                       "Set the log file save path (including file name)");
        res &= arg.reg("--dir",
                       1,
                       Set(dirbase),
                       "Set the root path for reading files, affecting the "
                       "ReadFile, WriteFile and LoadDir series of functions");
        res &= arg.reg("--show",
                       0,
                       Flag(showDiff),
                       "Print diff and pause when checkpoint is failed");
        res &= arg.reg("--error-check",
                       0,
                       Flag(checkErrorStatus),
                       "Check Veloce error status after each execuation");
        res &= arg.reg("--record-cycle",
                       0,
                       Flag(doReadLcc),
                       "Record cycle number of each execuation");
        res &= arg.reg("--no-fncall",
                       0,
                       DownFlag(showFuncCall),
                       "Hide funcation call info");
        res &= arg.reg("--bundle-size",
                       1,
                       Set(bundleSize),
                       "Set the bundle size when instruction split");
        res &= arg.reg("--skip-checkpoint",
                       0,
                       Flag(skipCheckpoint),
                       "Skip all checkpoints, *BUT* checkpoints still play a "
                       "role in instruction split");
        res &=
            arg.reg("--skip-exec", 1, Set(skipExec), "Skip first n execuation");
        res &= arg.reg(
            "--enable-scheduler",
            0,
            Flag(useScheduler),
            "Experimental Features, Schedule instructions before execuate");
        res &= arg.reg("--show-scheduler-verbose",
                       0,
                       Flag(showSchedulerVerboseInfo),
                       "Experimental Features, Show scheduler info");
        if (payload.type == RunType::LateInit)
        {
            res &= arg.reg("-s",
                           0,
                           Assign(runType, RunType::Simulator),
                           "Set run target to Simulator");
#ifdef VELOCE_AVAILABLE
            res &= arg.reg("-v",
                           0,
                           Assign(runType, RunType::Veloce),
                           "Set run target to Veloce");
#endif
            res &= arg.reg("-i",
                           0,
                           Assign(runType, RunType::InstGen),
                           "Set run target to InstGen");
        }
        return res;
    }

    virtual std::string
    CompareFileNameMap(const std::string &name)
    {
        return name;
    }

    virtual bool
    CheckPointFilter(const std::string &name,
                     const std::vector<float> &data,
                     const std::string &compare_file)
    {
        return true;
    }

    virtual std::vector<SpyInfo>
    SpyInfoFilter(const std::vector<SpyInfo> &spyInfos)
    {
        return spyInfos;
    }

    Inst2 *globalInst2 = nullptr;

    void
    Divino()
    {
        PayLoad oldPayLoad(RunType::InstGen);
        const bool oldShowFnCall = ShowFuncCallInfo();
        const uint32_t oldAddr = HBMAddr().GetOr(0);

        std::swap(oldPayLoad, payload);
        ShowFuncCallInfo() = false;

        Inst2 inst2;
        inst2.logLevel = DebugLevel::Silence;

        inst2.execFunc = [this](std::vector<Instruction *> &bundle,
                                const std::vector<SpyInfo> &spyInfos) -> bool
        { return Exec(bundle, spyInfos); };

        globalInst2 = &inst2;

        Main(inst2, inst2.inst.insts);

        globalInst2 = nullptr;

        inst2.Exec();

        instSizeDivinandi = instSize;
        execTotCntDivinandi = execTotCnt;
        instSize = 0;
        execTotCnt = 0;

        GlobalState().callCount.clear();
        HBMAddr() = oldAddr;
        ShowFuncCallInfo() = oldShowFnCall;
        std::swap(oldPayLoad, payload);

        std::clog << "DIVINATION: INST(S): " << instSizeDivinandi
                  << ", EXEC(S): " << execTotCntDivinandi << std::endl;
    }

    void
    StopWatchStart(std::vector<Instruction *> &bundle)
    {
        Inst2 inst2;
        inst2(SStoreDirect, 28, inst2.inst.ImmeU((lccInitSMemAddr - 1) * 4));
        inst2(SStoreDirect, 29, inst2.inst.ImmeU((lccInitSMemAddr - 2) * 4));
        inst2(ReadLcc, 28, 29);
        inst2(SStoreDirect, 28, inst2.inst.ImmeU((lccSMemAddr++) * 4));
        inst2(SStoreDirect, 29, inst2.inst.ImmeU((lccSMemAddr++) * 4));
        inst2(SLoadDirect, inst2.inst.ImmeU((lccInitSMemAddr - 1) * 4), 28);
        inst2(SLoadDirect, inst2.inst.ImmeU((lccInitSMemAddr - 2) * 4), 29);
        bundle.insert(bundle.begin(),
                      inst2.inst.insts.begin(),
                      inst2.inst.insts.end());
    }

    void
    StopWatchStop(std::vector<Instruction *> &bundle)
    {
        Inst2 inst2(bundle);
        inst2(SStoreDirect, 30, inst2.inst.ImmeU((lccInitSMemAddr - 1) * 4));
        inst2(SStoreDirect, 31, inst2.inst.ImmeU((lccInitSMemAddr - 2) * 4));
        inst2(ReadLcc, 30, 31);
        inst2(SStoreDirect, 30, inst2.inst.ImmeU((lccSMemAddr++) * 4));
        inst2(SStoreDirect, 31, inst2.inst.ImmeU((lccSMemAddr++) * 4));
        inst2(SLoadDirect, inst2.inst.ImmeU((lccInitSMemAddr - 1) * 4), 30);
        inst2(SLoadDirect, inst2.inst.ImmeU((lccInitSMemAddr - 2) * 4), 31);
    }

    void
    Go()
    {
        StateStopWatchStart();

        if (payload.type == RunType::LateInit)
        {
            payload = PayLoad(runType);
        }

        ShowFuncCallInfo() = showFuncCall;

        std::ofstream flog(logfile, std::ios::app);
        flog << DateTime() << " NEW EXEC, COMPILED AT " << __DATE__ " " __TIME__
             << std::endl;
        flog.close();

        Inst2 inst2;
        inst2.bundleInstSize = bundleSize;
        inst2.useScheduler = useScheduler;
        inst2.showSchedulerVerboseInfo = showSchedulerVerboseInfo;

        inst2.execFunc = [this](std::vector<Instruction *> &bundle,
                                const std::vector<SpyInfo> &spyInfos) -> bool
        { return Exec(bundle, spyInfos); };

        globalInst2 = &inst2;

        GlobalFfiInfo().ctx = this;

        SwitchState(S_GEN);

        Main(inst2, inst2.inst.insts);

        globalInst2 = nullptr;

        inst2.Exec();

        if (doReadLcc)
        {
            LogCycle();
        }

        SwitchState(S_OTHER);

        GlobalFfiInfo().ctx = nullptr;

        OnExecuted(inst2);

        fprintf(stderr,
                "%s %llds + %dns\n",
                "S_GEN",
                stateTot[S_GEN].tv_sec,
                stateTot[S_GEN].tv_nsec);
        fprintf(stderr,
                "%s %llds + %dns\n",
                "S_BUNDLE",
                stateTot[S_BUNDLE].tv_sec,
                stateTot[S_BUNDLE].tv_nsec);
        fprintf(stderr,
                "%s %llds + %dns\n",
                "S_RESET",
                stateTot[S_RESET].tv_sec,
                stateTot[S_RESET].tv_nsec);
        fprintf(stderr,
                "%s %ds + %dns\n",
                "S_WRITE_IMEM",
                stateTot[S_WRITE_IMEM].tv_sec,
                stateTot[S_WRITE_IMEM].tv_nsec);
        fprintf(stderr,
                "%s %llds + %dns\n",
                "S_EXEC",
                stateTot[S_EXEC].tv_sec,
                stateTot[S_EXEC].tv_nsec);
        fprintf(stderr,
                "%s %llds + %dns\n",
                "S_CHECK",
                stateTot[S_CHECK].tv_sec,
                stateTot[S_CHECK].tv_nsec);
        fprintf(stderr,
                "%s %ds + %dns\n",
                "S_ASSERT",
                stateTot[S_ASSERT].tv_sec,
                stateTot[S_ASSERT].tv_nsec);
        fprintf(stderr,
                "%s %llds + %dns\n",
                "S_CLEAR",
                stateTot[S_CLEAR].tv_sec,
                stateTot[S_CLEAR].tv_nsec);
        fprintf(stderr,
                "%s %llds + %dns\n",
                "S_OTHER",
                stateTot[S_OTHER].tv_sec,
                stateTot[S_OTHER].tv_nsec);
    }

    bool
    HBMInfeedInput(const float *data,
                   uint32_t size,
                   uint32_t offsetInKFloat = 0)
    {
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            Device_Simulator *sim = payload.simulator;
            sim->WriteToHBM((uint32_t *)(data), size, offsetInKFloat);
        }
        break;
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            std::clog << COLOR::BLUE << "HBM INFEED SIZE: " << size
                      << ", MAYBE COUNT: "
                      << (size / (kMaxDataLength / 4) / kMultipleOfBytes)
                      << COLOR::WHITE << std::endl;
            Veloce veloce(Device::VELOCE);
            veloce.OpenDevice();
            veloce.SetInfeedSync(DMA_INFEED_1, 0);
            veloce.DirectWrite((const char *)data,
                               Veloce::mem_id::HBM,
                               (size / (kMultipleOfBytes / 4)),
                               (offsetInKFloat / (kMultipleOfBytes / 4)),
                               0);

            std::vector<Instruction *> bundle;
            {
                Instruction *ins = new Instruction();
                ScalarOperationState scalar(S_HALT,
                                            0 /*perm*/,
                                            0 /*s_x*/,
                                            0 /*s_y*/,
                                            0 /*s_dest*/);
                ins->SetOperationState(Instruction::SCALARONE, &scalar);
                CompleteInstruction(ins);
                bundle.push_back(ins);

                AddNoop(10, bundle);
            }

            std::string imem_string = "";
            int instruction_num = ImemObj2StringTest(bundle, imem_string);
            int length = (imem_string.length() + kMultipleOfBytes - 1) /
                         kMultipleOfBytes;
            veloce.WriteToImem(imem_string.c_str(), length, instruction_num);

            veloce.ResetPC();
            veloce.Execute(50);

            veloce.CloseDevice();
        }
        break;
#endif
        default:
            break;
        }
        return true;
    }

    bool
    VMemInfeedInput(const float *data,
                    uint32_t size,
                    uint32_t offsetInKFloat = 0)
    {
        const auto block = kMultipleOfBytes / 4;
        if (size % block != 0)
        {
            const auto halfSize = DownRound(size, block);
            VMemInfeedInput(data, halfSize, offsetInKFloat);
            const auto newData = new float[block];
            memcpy(newData, data + halfSize, block * sizeof(float));
            std::cerr << COLOR::ORANGE << "VMem Infeed Size UpRound From "
                      << size << " To " << halfSize + 128
                      << ", Attention Data Conflict\n"
                      << COLOR::WHITE;
            VMemInfeedInput(newData, block, offsetInKFloat + halfSize);
            delete[] newData;
            return true;
        }
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            Device_Simulator *sim = payload.simulator;
            sim->WriteToVmemWithOffset((const char *)(data),
                                       (size / (kMultipleOfBytes / 4)),
                                       offsetInKFloat);
        }
        break;
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            std::clog << COLOR::BLUE << "VMEM INFEED SIZE: " << size
                      << ", MAYBE COUNT: "
                      << (size / (kMaxDataLength / 4) / kMultipleOfBytes)
                      << COLOR::WHITE << std::endl;
            Veloce veloce(Device::VELOCE);
            veloce.OpenDevice();
            veloce.SetInfeedSync(DMA_INFEED_1, 0);
            veloce.WriteToVmemWithOffset((const char *)data,
                                         (size / (kMultipleOfBytes / 4)),
                                         (offsetInKFloat * 4));

            std::vector<Instruction *> bundle;
            {
                Instruction *ins = new Instruction();
                ScalarOperationState scalar(S_HALT,
                                            0 /*perm*/,
                                            0 /*s_x*/,
                                            0 /*s_y*/,
                                            0 /*s_dest*/);
                ins->SetOperationState(Instruction::SCALARONE, &scalar);
                CompleteInstruction(ins);
                bundle.push_back(ins);

                AddNoop(10, bundle);
            }

            std::string imem_string = "";
            int instruction_num = ImemObj2StringTest(bundle, imem_string);
            int length = (imem_string.length() + kMultipleOfBytes - 1) /
                         kMultipleOfBytes;
            veloce.WriteToImem(imem_string.c_str(), length, instruction_num);

            veloce.ResetPC();
            veloce.Execute(50);

            veloce.CloseDevice();
        }
        break;
#endif
        default:
            break;
        }
        return true;
    }

    bool
    SMemInfeedInput(const float *data,
                    uint32_t size,
                    uint32_t offsetInKFloat = 0)
    {
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            Device_Simulator *sim = payload.simulator;
            Xys *xys = sim->*get(ThiefHandle<Device_Simulator, Xys *>());
            for (size_t i = 0; i < size; i++)
            {
                xys->GetSmem()->Store(offsetInKFloat + i,
                                      *(const uint32_t *)(data + i));
            }
        }
        break;
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            std::clog << COLOR::BLUE << "SMEM INFEED SIZE: " << size
                      << ", MAYBE COUNT: "
                      << (size / (kMaxDataLength / 4) / kMultipleOfBytes)
                      << COLOR::WHITE << std::endl;
            Veloce veloce(Device::VELOCE);
            veloce.OpenDevice();
            veloce.SetInfeedSync(DMA_INFEED_1, 0);
            veloce.WriteToSmemWithOffset(
                (const char *)data,
                (size / (kMultipleOfBytes / 4)),
                (offsetInKFloat / (kMultipleOfBytes / 4)));
            std::vector<Instruction *> bundle;
            {
                Instruction *ins = new Instruction();
                ScalarOperationState scalar(S_HALT,
                                            0 /*perm*/,
                                            0 /*s_x*/,
                                            0 /*s_y*/,
                                            0 /*s_dest*/);
                ins->SetOperationState(Instruction::SCALARONE, &scalar);
                CompleteInstruction(ins);
                bundle.push_back(ins);

                AddNoop(10, bundle);
            }

            std::string imem_string = "";
            int instruction_num = ImemObj2StringTest(bundle, imem_string);
            int length = (imem_string.length() + kMultipleOfBytes - 1) /
                         kMultipleOfBytes;
            veloce.WriteToImem(imem_string.c_str(), length, instruction_num);

            veloce.ResetPC();
            veloce.Execute(50);

            veloce.CloseDevice();
        }
        break;
#endif
        default:
            break;
        }
        return true;
    }

    bool
    ExecInSim(const std::vector<Instruction *> &bundle,
              const std::vector<SpyInfo> &spyInfos,
              Device_Simulator *sim)
    {
        SwitchState(S_WRITE_IMEM);
        sim->WriteToImem(bundle);
        SwitchState(S_EXEC);
        sim->Execute(3000000);

        SwitchState(S_ASSERT);
        bool res = true;

        for (auto &sinfo : spyInfos)
        {
            if (sinfo.addr >= 4096 * 1024)
            {
                res &= CheckPoint(
                    sinfo.name,
                    GetHBMInSim<float>(sinfo.addr - 4096 * 1024,
                                       sinfo.addr - 4096 * 1024 + sinfo.len,
                                       payload.simulator),
                    CompareFileNameMap(sinfo.compare_file));
            }
            else
            {
                res &= CheckPoint(sinfo.name,
                                  GetVMemInSim<float>(sinfo.addr,
                                                      sinfo.addr + sinfo.len,
                                                      payload.simulator),
                                  CompareFileNameMap(sinfo.compare_file));
            }
        }

        return res;
    }

#ifdef VELOCE_AVAILABLE
    static uint64_t
    VeloceReadCsr(uint64_t address)
    {
        FILE *fp_read;
        std::string result = "";
        result.reserve(128);
        std::string command = "cd /sys/bus/pci/devices/0000:01:00.0 && echo " +
                              std::to_string(address) +
                              " > reg_addr && cat reg_val";
        fp_read = popen(command.c_str(), "r");
        fscanf(fp_read, "%s", result.c_str());
        pclose(fp_read);
        return std::stoull(result, nullptr, 16);
    }

    static void
    VeloceWriteCsr(uint64_t address, uint32_t value)
    {
        FILE *fp_read;
        std::string command = "cd /sys/bus/pci/devices/0000:01:00.0 && echo " +
                              std::to_string(address) + " > reg_addr && echo " +
                              std::to_string(value) + " > reg_val";
        fp_read = popen(command.c_str(), "w");
        pclose(fp_read);
        return;
    }

    template <size_t ERROR_CNT>
    static bool
    VeloceAErrorCheck(uint64_t status,
                      const std::string (&errors)[ERROR_CNT],
                      const std::string prefix = "")
    {
        bool ok = true;

        std::bitset<ERROR_CNT> error(status);
        for (size_t i = 0; i < ERROR_CNT; i++)
        {
            if (error.test(i))
            {
                std::cerr << COLOR::ORANGE << prefix << errors[i]
                          << COLOR::WHITE << std::endl;
                ok &= false;
            }
        }

        return ok;
    }

    void
    VeloceClearErrorStatus()
    {
        if (!checkErrorStatus && !requireClearErrorStatus)
        {
            return;
        }
        using namespace VeloceCSR;

        VeloceWriteCsr(SCALAR_ERROR, 0);
        VeloceWriteCsr(VECTOR_PROGRAM_ERROR, 0);
        VeloceWriteCsr(VECTOR_FATAL_ERROR, 0);
        for (uint8_t i = 0; i < 128; i++)
        {
            VeloceWriteCsr(VECTOR_CORE_ERROR(i), 0);
        }

        requireClearErrorStatus = false;
    }

    bool
    VeloceErrorCheck(bool forceCheck = false)
    {
        if (!checkErrorStatus && !forceCheck)
        {
            return true;
        }
        bool ok = true;

        using namespace VeloceCSR;

        ok &= VeloceAErrorCheck<SCALAR_ERRORS_CNT>(VeloceReadCsr(SCALAR_ERROR),
                                                   SCALAR_ERRORS);
        ok &= VeloceAErrorCheck<VECTOR_PROGRAM_ERRORS_CNT>(
            VeloceReadCsr(VECTOR_PROGRAM_ERROR),
            VECTOR_PROGRAM_ERRORS);
        ok &= VeloceAErrorCheck<VECTOR_FATAL_ERRORS_CNT>(
            VeloceReadCsr(VECTOR_FATAL_ERROR),
            VECTOR_FATAL_ERRORS);

        for (uint8_t i = 0; i < 128; i++)
        {
            ok &= VeloceAErrorCheck<VECTOR_CORE_ERRORS_CNT>(
                VeloceReadCsr(VECTOR_CORE_ERROR(i)),
                VECTOR_CORE_ERRORS,
                "core" + std::to_string(i) + "_");
        }

        requireClearErrorStatus = !ok;

        return ok;
    }
#endif

    template <class T>
    T
    PeekVRegInSim(uint32_t ridx, uint32_t cidx, Device_Simulator *sim)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        Xys *xys = sim->*get(ThiefHandle<Device_Simulator, Xys *>());
        Processor &vectorProcessor = xys->*get(ThiefHandle<Xys, Processor>());
        RegisterFile *regs =
            vectorProcessor.*get(ThiefHandle<Processor, RegisterFile *>());
        auto reg = regs->FindMappedDataIndex(ridx);
        uint32_t val = reg->Read(cidx);
        return *(T *)(&val);
    }

#ifdef VELOCE_AVAILABLE
    template <class T>
    T
    PeekVRegInVeloce(uint32_t ridx, uint32_t cidx)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        uint32_t subcore = cidx / 128;
        uint32_t core = cidx % 128;
        VeloceWriteCsr(0x02010008, 0x0);
        uint32_t val =
            0x880000 + ((subcore & 0x7) << 15) + (ridx & ((1 << 16) - 1));
        VeloceWriteCsr(0x02010008, val);
        uint32_t addr = 0x02800000 + core * 0x10000 + 0x1e0;
        uint32_t data = VeloceReadCsr(addr);
        return *(T *)(&data);
    }
#endif

    template <class T = float>
    T
    PeekVReg(uint32_t ridx, uint32_t cidx)
    {
        if (globalInst2 != nullptr && globalInst2->inst.instCount() != 0)
        {
            globalInst2->Exec();
        }
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            return PeekVRegInSim<T>(ridx, cidx, payload.simulator);
        }
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            return PeekVRegInVeloce<T>(ridx, cidx);
        }
#endif
        default:
            return 0.0f;
        }
    }

    template <class T>
    T
    PeekSRegInSim(uint32_t ridx, Device_Simulator *sim)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        Xys *xys = sim->*get(ThiefHandle<Device_Simulator, Xys *>());
        Processor &scalarProcessor =
            xys->*get(ThiefHandle<Xys, Processor, CaseId::SCALAR_PROCESSOR>());
        RegisterFile *regs =
            scalarProcessor.*get(ThiefHandle<Processor, RegisterFile *>());
        auto reg = regs->FindMappedDataIndex(ridx);
        uint32_t val = reg->Read(0);
        return *(T *)(&val);
    }

    template <class T = float>
    T
    PeekSReg(uint32_t ridx)
    {
        if (globalInst2 != nullptr && globalInst2->inst.instCount() != 0)
        {
            globalInst2->Exec();
        }
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            return PeekSRegInSim<T>(ridx, payload.simulator);
        }
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            std::clog << COLOR::ORANGE << "UNSUPPORTED PEEK SREG IN VELOCE\n"
                      << COLOR::WHITE;
            return 0.0f;
        }
#endif
        default:
            return 0.0f;
        }
    }

#ifdef VELOCE_AVAILABLE
    bool
    PcAndErrorCheck(int expectedPc)
    {
        const int pc = VeloceReadCsr(VeloceCSR::PC) & 0xffff;
        if (pc != expectedPc)
        {
            std::cerr << COLOR::RED << "ACTUAL PC: " << pc
                      << ", EXPECTED: " << expectedPc << "\nENTER TO CONTINUE\n"
                      << COLOR::WHITE;
            int c = getchar();
        }

        if (!VeloceErrorCheck(pc != expectedPc))
        {
            std::cerr << COLOR::RED << "\nENTER TO CONTINUE\n" << COLOR::WHITE;
            int c = getchar();
        }

        return pc != expectedPc;
    }

    bool
    ExecInVeloce(const std::vector<Instruction *> &bundle,
                 const std::vector<SpyInfo> &spyInfos,
                 uint32_t expectedPc)
    {
        {
            std::string imem_string = "";
            int instruction_num = ImemObj2StringTest(bundle, imem_string);
            int length = (imem_string.length() + kMultipleOfBytes - 1) /
                         kMultipleOfBytes;
            SwitchState(S_RESET);
            VeloceClearErrorStatus();
            Veloce veloce(Device::VELOCE);
            veloce.OpenDevice();
            SwitchState(S_WRITE_IMEM);
            veloce.WriteToImem(imem_string.c_str(), length, instruction_num);
            veloce.ResetPC();
            SwitchState(S_EXEC);
            veloce.Execute(50);
            veloce.CloseDevice();
            SwitchState(S_CHECK);
            PcAndErrorCheck(expectedPc);
        }

        SwitchState(S_ASSERT);
        bool res = true;

        if (!spyInfos.empty())
        {
            uint32_t maxAddr = 0;
            const auto VMemSize = GlobalDeviceConfig().VMemSize;
            for (auto &sinfo : spyInfos)
            {
                if (sinfo.addr + sinfo.len < VMemSize)
                {
                    maxAddr = std::max(maxAddr, sinfo.addr + sinfo.len);
                }
            }
            auto output = GetVMemInVeloce<float>(0, maxAddr);
            for (auto &sinfo : spyInfos)
            {
                if (sinfo.addr + sinfo.len >= VMemSize)
                {
                    continue;
                }
                res &= CheckPoint(
                    sinfo.name,
                    std::vector<float>(output.begin() + sinfo.addr,
                                       output.begin() + sinfo.addr + sinfo.len),
                    CompareFileNameMap(sinfo.compare_file));
            }
        }

        return res;
    }
#endif

    template <class T>
    std::vector<T>
    GetVMemInSim(uint32_t addr_start, uint32_t addr_end, Device_Simulator *sim)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        Xys *xys = sim->*get(ThiefHandle<Device_Simulator, Xys *>());
        std::vector<T> res;
        const auto VMemSize = GlobalDeviceConfig().VMemSize;
        for (int i = addr_start; i < addr_end && i < VMemSize; i++)
        {
            auto val = xys->BackdoorStoreVectorMemory(i);
            res.push_back(*(T *)(&val));
        }
        return res;
    }

#ifdef VELOCE_AVAILABLE
    template <class T>
    std::vector<T>
    GetVMemInVeloce(uint32_t addr_start, uint32_t addr_end)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        ToHttSize = ((addr_end + 127) / 128) * 128;
        if (ToHttSize == 0)
        {
            std::clog << COLOR::ORANGE << "WARNING! ZERO-LENGTH OUTFEED\n"
                      << COLOR::WHITE;
            return std::vector<T>();
        }
        std::clog << COLOR::BLUE << "VMEM OUTFEED SIZE " << ToHttSize << ", "
                  << OutfeedSize() << COLOR::WHITE << std::endl;
        std::vector<Instruction *> insts;
        AddNoop(10, insts);
        VmemToHtt(insts, OutfeedSize());

        {
            Instruction *ins = new Instruction();
            ScalarOperationState scalar(S_HALT, 0, 0, 0, 0);
            ins->SetOperationState(Instruction::SCALARONE, &scalar);
            CompleteInstruction(ins);
            insts.push_back(ins);

            AddNoop(10, insts);
        }

        std::string imem_string = "";
        int instruction_num = ImemObj2StringTest(insts, imem_string);
        int length =
            (imem_string.length() + kMultipleOfBytes - 1) / kMultipleOfBytes;

        VeloceClearErrorStatus();
        Veloce veloce(Device::VELOCE);
        veloce.OpenDevice();
        veloce.WriteToImem(imem_string.c_str(), length, instruction_num);
        veloce.ResetPC();
        veloce.Execute(50);
        auto output = GetHtt<T>(veloce, OutfeedSize());
        veloce.CloseDevice();

        PcAndErrorCheck(14);

        if (addr_start == 0)
        {
            output.resize(addr_end);
            return output;
        }
        return std::vector<T>(output.begin() + addr_start,
                              output.begin() + addr_end);
    }
#endif

    template <class T = float>
    std::vector<T>
    GetVMem(uint32_t addr_start, uint32_t addr_end)
    {
        if (globalInst2 != nullptr && globalInst2->inst.instCount() != 0)
        {
            globalInst2->Exec();
        }
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            return GetVMemInSim<T>(addr_start, addr_end, payload.simulator);
        }
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            return GetVMemInVeloce<T>(addr_start, addr_end);
        }
#endif
        default:
            return std::vector<T>(addr_end - addr_start);
        }
    }

    template <class T>
    std::vector<T>
    GetHBMInSim(uint32_t addr_start, uint32_t addr_end, Device_Simulator *sim)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        std::shared_ptr<Hbm> hbm =
            payload.simulator
                ->*get(ThiefHandle<Device_Simulator, std::shared_ptr<Hbm>>());
        std::vector<T> res;
        const auto HBMSize = GlobalDeviceConfig().HBMSize;
        for (int i = addr_start; i < addr_end && i < HBMSize; i++)
        {
            auto val = *(hbm->Load(i)).second;
            res.push_back(*(T *)(&val));
        }
        return res;
    }

#ifdef VELOCE_AVAILABLE
    template <class T>
    std::vector<T>
    GetHBMInVeloce(uint32_t addr_start, uint32_t addr_end)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        ToHttSize = ((addr_end + 127) / 128) * 128;
        if (ToHttSize == 0)
        {
            std::clog << COLOR::ORANGE << "WARNING! ZERO-LENGTH OUTFEED\n"
                      << COLOR::WHITE;
            return std::vector<T>();
        }
        std::clog << COLOR::BLUE << "OUTFEED SIZE " << ToHttSize << ", "
                  << OutfeedSize() << COLOR::WHITE << std::endl;
        std::vector<Instruction *> insts;
        AddNoop(10, insts);
        HbmToHtt(insts, OutfeedSize());

        {
            Instruction *ins = new Instruction();
            ScalarOperationState scalar(S_HALT, 0, 0, 0, 0);
            ins->SetOperationState(Instruction::SCALARONE, &scalar);
            CompleteInstruction(ins);
            insts.push_back(ins);

            AddNoop(10, insts);
        }

        std::string imem_string = "";
        int instruction_num = ImemObj2StringTest(insts, imem_string);
        int length =
            (imem_string.length() + kMultipleOfBytes - 1) / kMultipleOfBytes;

        VeloceClearErrorStatus();
        Veloce veloce(Device::VELOCE);
        veloce.OpenDevice();
        veloce.WriteToImem(imem_string.c_str(), length, instruction_num);
        veloce.ResetPC();
        veloce.Execute(50);
        auto output = GetHtt<T>(veloce, OutfeedSize());
        veloce.CloseDevice();

        PcAndErrorCheck(14);

        if (addr_start == 0)
        {
            output.resize(addr_end);
            return output;
        }
        return std::vector<T>(output.begin() + addr_start,
                              output.begin() + addr_end);
    }
#endif

    template <class T = float>
    std::vector<T>
    GetHBM(uint32_t addr_start, uint32_t addr_end)
    {
        if (globalInst2 != nullptr && globalInst2->inst.instCount() != 0)
        {
            globalInst2->Exec();
        }
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            return GetHBMInSim<T>(addr_start, addr_end, payload.simulator);
        }
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            return GetHBMInVeloce<T>(addr_start, addr_end);
        }
#endif
        default:
            return std::vector<T>(addr_end - addr_start);
        }
    }

    template <class T>
    std::vector<T>
    GetSMemInSim(uint32_t addr_start, uint32_t addr_end, Device_Simulator *sim)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        Xys *xys = sim->*get(ThiefHandle<Device_Simulator, Xys *>());
        std::vector<T> res;
        const auto SMemSize = GlobalDeviceConfig().SMemSize;
        for (int i = addr_start; i < addr_end && i < SMemSize; i++)
        {
            auto val = xys->ReadScalarMemory(i);
            res.push_back(*(T *)(&val));
        }
        return res;
    }

#ifdef VELOCE_AVAILABLE
    template <class T>
    std::vector<T>
    GetSMemInVeloce(uint32_t addr_start, uint32_t addr_end)
    {
        static_assert(sizeof(T) == 4U, "require sizeof(T) == 4U");
        ToHttSize = ((addr_end + 127) / 128) * 128;
        if (ToHttSize == 0)
        {
            std::clog << COLOR::ORANGE << "WARNING! ZERO-LENGTH OUTFEED\n"
                      << COLOR::WHITE;
            return std::vector<T>();
        }
        std::clog << COLOR::BLUE << "SMEM OUTFEED SIZE " << ToHttSize << ", "
                  << OutfeedSize() << COLOR::WHITE << std::endl;
        std::vector<Instruction *> insts;
        AddNoop(10, insts);
        DmaNonBlock(insts,
                    DMA_DEST::OUTFEED<0>::MISC | DMA_SRC::XYS<0>::SMEM,
                    0,
                    0,
                    ToHttSize);
        const auto exceptPc = insts.size() + 1;
        {
            Instruction *ins = new Instruction();
            ScalarOperationState scalar(S_HALT, 0, 0, 0, 0);
            ins->SetOperationState(Instruction::SCALARONE, &scalar);
            CompleteInstruction(ins);
            insts.push_back(ins);

            AddNoop(10, insts);
        }

        std::string imem_string = "";
        int instruction_num = ImemObj2StringTest(insts, imem_string);
        int length =
            (imem_string.length() + kMultipleOfBytes - 1) / kMultipleOfBytes;

        VeloceClearErrorStatus();
        Veloce veloce(Device::VELOCE);
        veloce.OpenDevice();
        veloce.WriteToImem(imem_string.c_str(), length, instruction_num);
        veloce.ResetPC();
        veloce.Execute(50);
        auto output = GetHtt<T>(veloce, OutfeedSize());
        veloce.CloseDevice();

        PcAndErrorCheck(exceptPc);

        if (addr_start == 0)
        {
            output.resize(addr_end);
            return output;
        }
        return std::vector<T>(output.begin() + addr_start,
                              output.begin() + addr_end);
    }
#endif

    template <class T = float>
    std::vector<T>
    GetSMem(uint32_t addr_start, uint32_t addr_end, bool ignoredUnExec = false)
    {
        if (globalInst2 != nullptr && globalInst2->inst.instCount() != 0 &&
            !ignoredUnExec)
        {
            globalInst2->Exec();
        }
        switch (payload.type)
        {
        case RunType::Simulator:
        {
            return GetSMemInSim<T>(addr_start, addr_end, payload.simulator);
        }
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
        {
            return GetSMemInVeloce<T>(addr_start, addr_end);
        }
#endif
        default:
            return std::vector<T>(addr_end - addr_start);
        }
    }

    bool
    Exec(std::vector<Instruction *> &bundle,
         const std::vector<SpyInfo> &spyInfos)
    {
        if (execTotCnt < skipExec)
        {
            execTotCnt++;
            fprintf(stderr, "Skip[%d]: %d\n", execTotCnt, bundle.size());
            for (auto &i : bundle)
            {
                delete i;
            }
            return true;
        }
        SwitchState(S_BUNDLE);
        int dmaCnt = 0;
        for (auto &i : bundle)
        {
            if (i->GetOperation(Instruction::SCALARONE)->GetOpCode() ==
                S_LOCAL_DMA)
            {
                dmaCnt++;
            }
        }
        instSize += bundle.size();
        instCountLog.push_back(bundle.size());
        execTotCnt++;
        if (doReadLcc)
        {
            StopWatchStart(bundle);
            StopWatchStop(bundle);
        }
        int expectedPc = bundle.size() + 1;
        if (execTotCntDivinandi != 0)
        {
            fprintf(stderr, "%d/%lld|", execTotCnt, execTotCntDivinandi);
        }
        const auto spyInfos2 =
            skipCheckpoint ? (std::vector<SpyInfo>{}) : SpyInfoFilter(spyInfos);
        fprintf(stderr,
                "Exec[%d]: %d, has %d dmas, has %d checkpoints\n",
                execTotCnt,
                bundle.size(),
                dmaCnt,
                spyInfos2.size());
        {
            Instruction *ins = new Instruction();
            ScalarOperationState scalar(S_HALT,
                                        0 /*perm*/,
                                        0 /*s_x*/,
                                        0 /*s_y*/,
                                        0 /*s_dest*/);
            ins->SetOperationState(Instruction::SCALARONE, &scalar);
            CompleteInstruction(ins);
            bundle.push_back(ins);
            AddNoop(10, bundle);
        }
        for (auto &sinfo : spyInfos2)
        {
            std::clog << COLOR::KANON << "CHECKPOINT " << sinfo.name << " "
                      << sinfo.addr << "-" << sinfo.addr + sinfo.len << " ? "
                      << sinfo.compare_file << COLOR::WHITE << std::endl;
        }
        switch (payload.type)
        {
        case RunType::Simulator:
            ExecInSim(bundle, spyInfos2, payload.simulator);
            break;
#ifdef VELOCE_AVAILABLE
        case RunType::Veloce:
            ExecInVeloce(bundle, spyInfos2, expectedPc);
            break;
#endif
        default:
            break;
        }
        SwitchState(S_CLEAR);
        for (auto &i : bundle)
        {
            delete i;
        }
        if (doReadLcc && execTotCnt % readLccPerExec == 0)
        {
            LogCycle();
        }
        SwitchState(S_GEN);
        return true;
    }

    void
    LogCycle()
    {
        std::clog << lccInitSMemAddr << ", " << lccSMemAddr << std::endl;
        std::vector<uint32_t> data =
            GetSMem<uint32_t>(lccInitSMemAddr, lccSMemAddr, true);
        std::clog << "GotSMem" << std::endl;
        auto size = (lccSMemAddr - lccInitSMemAddr) / 2;
        uint64_t *rep = (uint64_t *)data.data();
        std::ofstream clockLog("clock.log", std::ios::app);
        for (auto i = 0u; i < size; i += 2)
        {
            const uint64_t start = rep[i];
            const uint64_t end = rep[i + 1];
            std::clog << "[" << execTotCnt - ((size - i) / 2) << "]"
                      << (end - start) << std::endl;
            clockLog << "[" << execTotCnt - ((size - i) / 2) << "]"
                     << (end - start) << std::endl;
            cycles += (end - start);
            cycleLog.push_back(end - start);
        }
        lccSMemAddr = lccInitSMemAddr;
    }

    static std::tuple<double, double, double>
    Compare(const std::vector<float> &data, const std::vector<float> &truth)
    {
        double avgDiff = 0.0;
        double maxDiff = 0.0;
        std::vector<double> diffs(data.size());
        for (int i = 0; i < data.size(); i++)
        {
            double diff = Diff(data[i], truth[i]);
            avgDiff += diff / data.size();
            maxDiff = std::max(diff, maxDiff);
            diffs[i] = diff;
        }
        double sig = 0.0;
        for (auto f : diffs)
        {
            sig += (f - avgDiff) * (f - avgDiff) / data.size();
        }
        sig = std::sqrt(sig);
        return std::make_tuple(avgDiff, maxDiff, sig);
    }

    bool
    CheckPoint(const std::string &name,
               const std::vector<float> &data,
               const std::string &compare_file)
    {
        std::clog << DateTime() << " CHECK " << name << " WITH "
                  << compare_file;
        if (!CheckPointFilter(name, data, compare_file))
        {
            return true;
        }
        std::vector<float> truth(data.size());
        int res;
        if (compare_file.back() == 'n')
        {
            res = ReadFileBin(truth.data(), compare_file, truth.size());
        }
        else
        {
            res = ReadFile(truth.data(), compare_file, truth.size());
        }

        if (res == -1)
        {
            return false;
        }

        auto cmpRes = Compare(data, truth);
        double avgDiff = std::get<0>(cmpRes);
        double maxDiff = std::get<1>(cmpRes);
        double sig = std::get<2>(cmpRes);

        bool pass = !(avgDiff > 0.05 || std::isnan(avgDiff));

        if (showDiff && (!pass))
        {
            ShowData(name, data, truth);
            std::clog << "CHECK " << name << " WITH " << compare_file
                      << std::endl;
        }

        std::clog << (pass ? COLOR::GREEN : COLOR::RED)
                  << "AVGDIFF: " << std::fixed << avgDiff * 100.0 << "%\t"
                  << "MAXDIFF: " << std::fixed << maxDiff * 100.0 << "%\t"
                  << "SIGMA:   " << std::fixed << sig * 100.0 << "%\n"
                  << COLOR::WHITE;
        std::ofstream flog(logfile, std::ios::app);
        flog << DateTime() << " CHECK " << name << " WITH " << compare_file
             << "\tAVGDIFF: " << std::fixed << avgDiff * 100.0 << "%"
             << "\tMAXDIFF: " << std::fixed << maxDiff * 100.0 << "%"
             << "\tSIGMA: " << std::fixed << sig * 100.0 << "%" << std::endl;
        flog.close();
        if (showDiff && !pass)
        {
            int c = getchar();
        }

        return pass;
    }

    static std::string
    DateTime()
    {
        std::time_t t = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::tm *tm = std::localtime(&t);
        char buf[512];
        strftime(buf, 512, "%Y/%m/%d %T", tm);
        return std::string(buf);
    }

    static void
    HbmToHtt(std::vector<Instruction *> &instruction_list, uint32_t outfeedSize)
    {
        Instruction *inst;
        int misc = 0b1000000100000000;
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, 0);
            ScalarOperationState store(S_U32_MOVE, 0, 0, 32, 1);
            inst->SetOperationState(Instruction::SCALARONE, &store);
            ScalarOperationState store2(S_U32_MOVE, 0, 0, 33, 2);
            inst->SetOperationState(Instruction::SCALARTWO, &store2);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);

            // inst->SetImmediateValue(Instruction::IMMEDIATE0, 2);  //dma size
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                    HelperGetAddress(outfeedSize).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                    HelperGetAddress(outfeedSize).first);
            ScalarOperationState store3(S_U32_MOVE, 0, 0, 44, 3);
            inst->SetOperationState(Instruction::SCALARONE, &store3);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);

            inst = new Instruction();
            ScalarOperationState dma_local(S_LOCAL_DMA, 0, 1, 3, 2, 46, misc);
            inst->SetOperationState(Instruction::SCALARONE, &dma_local);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
    }

    static void
    VmemToHtt(std::vector<Instruction *> &instruction_list,
              uint32_t outfeedSize)
    {
        int misc = 0b1000001000000000;
        Instruction *inst;
        if (1)
        {
            inst = new Instruction();
            inst->SetImmediateValue(Instruction::IMMEDIATE0, 0);
            inst->SetImmediateValue(Instruction::IMMEDIATE1, 0);
            ScalarOperationState store(S_U32_MOVE, 0, 0, 32, 1);
            inst->SetOperationState(Instruction::SCALARONE, &store);
            ScalarOperationState store2(S_U32_MOVE, 0, 0, 33, 2);
            inst->SetOperationState(Instruction::SCALARTWO, &store2);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);

            inst = new Instruction();
            // inst->SetImmediateValue(Instruction::IMMEDIATE0, 2);
            inst->SetImmediateValue(Instruction::IMMEDIATE0,
                                    HelperGetAddress(outfeedSize).second);
            inst->SetImmediateValue(Instruction::IMMEDIATE1,
                                    HelperGetAddress(outfeedSize).first);
            ScalarOperationState store3(S_U32_MOVE, 0, 0, 44, 3);
            inst->SetOperationState(Instruction::SCALARONE, &store3);
            CompleteInstruction(inst);

            instruction_list.push_back(inst);
            inst = new Instruction();
            ScalarOperationState dma_local(S_LOCAL_DMA, 0, 1, 3, 2, 46, misc);
            inst->SetOperationState(Instruction::SCALARONE, &dma_local);
            CompleteInstruction(inst);
            instruction_list.push_back(inst);
        }
    }

#ifdef VELOCE_AVAILABLE
    template <class T>
    static std::vector<T>
    GetHtt(Veloce &veloce, uint32_t outfeedSize)
    {
        uint32_t httSize = outfeedSize * (kMultipleOfBytes / sizeof(T));
        std::vector<T> output(httSize);
        veloce.LoadingDataFromChip((char *)(output.data()), outfeedSize, 0);
        return output;
    }
#endif

    int64_t
    ReadFile(float *data,
             const std::string &filename,
             const int64_t expectedSize)
    {
        std::clog << "  READ FILE " << dirbase << filename;
        FILE *fd = fopen((dirbase + filename).c_str(), "r");
        if (fd == nullptr)
        {
            std::cerr << COLOR::RED << " FAIL" << COLOR::WHITE << std::endl;
            return -1;
        }
        char buf[256];
        uint32_t size = 0;
        while (fgets(buf, 256, fd) != nullptr &&
               (expectedSize == -1 || size < expectedSize))
        {
            *data = std::strtof(buf, nullptr);
            data++;
            size++;
        }
        fclose(fd);
        std::clog << " SIZE: " << size << std::endl;
        return size;
    }

    int64_t
    WriteFile(const float *data,
              const std::string &filename,
              const size_t dataSize)
    {
        std::clog << "  WRITE FILE " << dirbase << filename;
        FILE *fd = fopen((dirbase + filename).c_str(), "w");
        if (fd == nullptr)
        {
            std::cerr << COLOR::RED << " FAIL" << COLOR::WHITE << std::endl;
            return -1;
        }
        for (size_t i = 0; i < dataSize; i++)
        {
            fprintf(fd, "%.18e\n", data[i]);
        }
        fclose(fd);
        std::clog << " SIZE: " << dataSize << std::endl;
        return dataSize;
    }

    int64_t
    ReadFileBin(float *data, const std::string &filename, const int64_t bufSize)
    {
        std::clog << "  READ FILE " << dirbase << filename;
        FILE *fd = fopen((dirbase + filename).c_str(), "r");
        if (fd == nullptr)
        {
            std::cerr << COLOR::RED << " FAIL" << COLOR::WHITE << std::endl;
            return -1;
        }
        uint32_t size = 0;
        fseek(fd, 0, SEEK_END);
        size = ftell(fd) / sizeof(float);
        fseek(fd, 0, SEEK_SET);
        uint32_t readSize =
            bufSize < 0 ? size : std::min(size, (uint32_t)bufSize);
        fread(data, sizeof(float), readSize, fd);
        fclose(fd);
        std::clog << " SIZE: " << readSize << "/" << size << std::endl;
        return size;
    }

    int64_t
    WriteFileBin(const float *data,
                 const std::string &filename,
                 const size_t dataSize)
    {
        std::clog << "  WRITE FILE " << dirbase << filename;
        FILE *fd = fopen((dirbase + filename).c_str(), "wb");
        if (fd == nullptr)
        {
            std::cerr << COLOR::RED << " FAIL" << COLOR::WHITE << std::endl;
            return -1;
        }
        auto writeSize = fwrite(data, sizeof(float), dataSize, fd);
        fclose(fd);
        std::clog << " SIZE: " << writeSize << "/" << dataSize << std::endl;
        return writeSize;
    }

    int64_t
    ReadFileT(float *data,
              const std::string &filename,
              const int64_t height,
              const int64_t width)
    {
        std::clog << "T READ FILE " << dirbase << filename;
        FILE *fd = fopen((dirbase + filename).c_str(), "r");
        if (fd == nullptr)
        {
            std::cerr << COLOR::RED << " FAIL" << COLOR::WHITE << std::endl;
            return -1;
        }
        uint32_t size = 0;
        for (auto i = 0; i < height; i++)
        {
            for (auto j = 0; j < width; j++)
            {
                char buf[256];
                if (fgets(buf, 256, fd) == nullptr)
                {
                    return size;
                }
                data[j * height + i] = std::strtof(buf, nullptr);
                size++;
            }
        }
        fclose(fd);
        std::clog << " SIZE: " << size << std::endl;
        return size;
    }

    int64_t
    WriteFileT(const float *data,
               const std::string &filename,
               const int64_t height,
               const int64_t width)
    {
        std::clog << "T WRITE FILE " << dirbase << filename;
        FILE *fd = fopen((dirbase + filename).c_str(), "w");
        if (fd == nullptr)
        {
            std::cerr << COLOR::RED << " FAIL" << COLOR::WHITE << std::endl;
            return -1;
        }
        for (auto i = 0; i < height; i++)
        {
            for (auto j = 0; j < width; j++)
            {
                fprintf(fd, "%.18e\n", data[j * height + i]);
            }
        }
        fclose(fd);
        std::clog << " SIZE: " << height * width << std::endl;
        return height * width;
    }

#ifdef __linux__
    static void
    DirWalking(const std::string &path,
               const std::function<void(const std::string &)> cb)
    {
        DIR *dir = opendir(path.c_str());
        if (dir == nullptr)
        {
            std::cerr << COLOR::RED << "OPENDIR " << path << " FAIL\n"
                      << COLOR::WHITE;
            return;
        }
        dirent *file = nullptr;
        while ((file = readdir(dir)) != nullptr)
        {
            cb(file->d_name);
        }
        closedir(dir);
    }
#else
    static void
    DirWalking(const std::string &path,
               const std::function<void(const std::string &)> cb)
    {
        for (auto &p : std::filesystem::directory_iterator("sandbox"))
        {
            if (p.is_regular_file())
            {
                cb(p.path().filename().string());
            }
        }
    }
#endif

    std::map<std::string, uint32_t>
    LoadBinDir(const std::string &path,
               float *data,
               std::function<bool(const std::string &)> filter = nullptr,
               std::function<void(const std::string &, const float *, size_t)>
                   onRead = nullptr)
    {
        uint32_t off = HBMAddr().GetOr(0);
        std::map<std::string, uint32_t> weightMap;
        DirWalking(dirbase + path,
                   [this, data, &path, &off, &filter, &onRead, &weightMap](
                       const std::string &filename)
                   {
                       if (filename == "." || filename == ".." ||
                           (filter != nullptr && !filter(filename)))
                       {
                           return;
                       }
                       std::clog << off << ": ";
                       auto len = ReadFileBin(data + off, path + filename, -1);
                       if (len == -1)
                       {
                           return;
                       }
                       if (onRead != nullptr)
                       {
                           onRead(filename, data + off, len);
                       }
                       weightMap[filename.substr(0, filename.size() - 4)] = off;
                       off += len;
                       off = ((off + 127) / 128) * 128;
                   });
        HBMAddr() = off;
        return weightMap;
    }

    std::map<std::string, uint32_t>
    LoadDir(
        const std::string &path,
        float *data,
        std::function<std::tuple<bool, uint32_t, uint32_t>(const std::string &)>
            transFilter = nullptr,
        std::function<bool(const std::string &)> filter = nullptr,
        std::function<void(const std::string &, const float *, size_t)> onRead =
            nullptr)
    {
        uint32_t off = HBMAddr().GetOr(0);
        std::map<std::string, uint32_t> weightMap;
        DirWalking(
            dirbase + path,
            [this,
             data,
             &path,
             &off,
             &transFilter,
             &filter,
             &onRead,
             &weightMap](const std::string &filename)
            {
                if (filename == "." || filename == ".." ||
                    (filter != nullptr && !filter(filename)))
                {
                    return;
                }
                int64_t len = -1;
                std::clog << off << ": ";
                if (filename.back() == 'n')
                {
                    len = ReadFileBin(data + off, path + filename, -1);
                }
                else
                {
                    if (transFilter != nullptr)
                    {
                        auto trans = transFilter(filename);
                        if (std::get<0>(trans))
                        {
                            len = ReadFileT(data + off,
                                            path + filename,
                                            std::get<1>(trans),
                                            std::get<2>(trans));
                        }
                        else
                        {
                            len = ReadFile(data + off, path + filename, -1);
                        }
                    }
                    else
                    {
                        len = ReadFile(data + off, path + filename, -1);
                    }
                }
                if (len == -1)
                {
                    return;
                }
                if (onRead != nullptr)
                {
                    onRead(filename, data + off, len);
                }
                weightMap[filename.substr(0, filename.size() - 4)] = off;
                off += len;
                off = ((off + 127) / 128) * 128;
            });
        HBMAddr() = off;
        return weightMap;
    }
};

#endif
