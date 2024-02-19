#pragma once

#ifndef _FUNC_HELPER_H_
#    define _FUNC_HELPER_H_

namespace FuncHelperFeatureTest
{
constexpr int FUNC_HELPER_VERSION = 221123;
constexpr bool VMEM_4M_SUPPORT = true;
constexpr bool VMEM_6M_SUPPORT = true;
constexpr bool CONV2D_SELF_TRANSPOSE = true;
constexpr bool GLOBAL_HBM_ADDR = true;
constexpr bool DEFAULT_NO_TRANSPOSE_HACK = true;
constexpr bool DMA_ALL_512B_GRANULE = true;
constexpr bool DMA_128_ALIGN_CHECK = true;
constexpr bool EMBEDDING_EXPAND_FIRST = true;
constexpr bool EMBEDDING_ANY_VF2S_FIX = true;
constexpr bool INST2_VER_EXPORT = true;
constexpr bool EMBBEDDING_INDEX_F2S_TRUNC = true;
constexpr bool EXPAND_INF_FILL = true;
} // namespace FuncHelperFeatureTest

#    include "InstHelper2.h"
#    include "instruction/Instruction.h"
#    include <vector>

bool &IKnowItIsRunningInSimulatorNotVeloceAndISureINeedFiveTransposeHack();

bool &ShowFuncCallInfo();

namespace InstVer
{
void PermuteInst(Inst2 &inst2,
                 uint32_t vMemSrcAddr,
                 const std::vector<uint32_t> &dimSize,
                 const std::vector<uint32_t> &newDims,
                 uint32_t vMemDestAddr);

void LinearExVMemIOInst(Inst2 &inst2,
                        uint32_t splitHeight,
                        uint32_t splitWidth,
                        uint32_t vmemInputAddr,
                        const std::array<uint32_t, 4> &inputSize,
                        uint32_t hbmWeightAddr,
                        const std::array<uint32_t, 2> &weightSize,
                        uint32_t vmemOutputAddr);

void EmbeddingExInst(Inst2 &inst2,
                     uint32_t hbmEmbeddingAddr,
                     uint32_t embeddingHeight,
                     uint32_t embeddingWidth,
                     uint32_t indicesAddr,
                     uint32_t indicesSize,
                     uint32_t outputAddr,
                     uint32_t weight_addr);

void Embeddings(Inst2 &inst,
                uint32_t hbmEmbeddingDataAddr,
                uint32_t embeddingHeight,
                uint32_t embeddingWidth,
                int input_addr,
                int inputSize,
                int output_addr,
                int weight_input_addr);

void MatMulInst(Inst2 &inst2,
                uint32_t leftMatAddr,
                const std::array<uint32_t, 4> &leftMatSize,
                uint32_t rightMatAddr,
                const std::array<uint32_t, 4> &rightMatSize,
                uint32_t outputAddr);
} // namespace InstVer

void DokodemoLoad(std::vector<Instruction *> &instList,
                  const std::vector<uint32_t> &reservedVReg,
                  const std::vector<uint32_t> &reservedSReg,
                  const std::vector<uint32_t> &reservedVMask,
                  const std::vector<uint32_t> &reservedPermit,
                  uint32_t vMemSrcAddr,
                  uint32_t dataLength,
                  uint32_t vRegDest,
                  uint32_t vRegOffset);

void DokodemoStore(std::vector<Instruction *> &instList,
                   const std::vector<uint32_t> &reservedVReg,
                   const std::vector<uint32_t> &reservedSReg,
                   const std::vector<uint32_t> &reservedVMask,
                   const std::vector<uint32_t> &reservedPermit,
                   uint32_t vRegSrc,
                   uint32_t vRegOffset,
                   uint32_t dataLength,
                   uint32_t vMemDestAddr);

void Transpose(std::vector<Instruction *> &instList,
               const std::vector<uint32_t> &reservedVReg,
               const std::vector<uint32_t> &reservedSReg,
               const std::vector<uint32_t> &reservedVMask,
               const std::vector<uint32_t> &reservedPermit,
               uint32_t vMemSrcAddr,
               uint32_t height,
               uint32_t width,
               uint32_t vMemDestAddr);

void Permute(std::vector<Instruction *> &instList,
             const std::vector<uint32_t> &reservedVReg,
             const std::vector<uint32_t> &reservedSReg,
             const std::vector<uint32_t> &reservedVMask,
             const std::vector<uint32_t> &reservedPermit,
             uint32_t vMemSrcAddr,
             const std::vector<uint32_t> &dimSize,
             const std::vector<uint32_t> &newDims,
             uint32_t vMemDestAddr);

// this function require addr align to 128
void Embedding(std::vector<Instruction *> &instList,
               const std::vector<uint32_t> &reservedVReg,
               const std::vector<uint32_t> &reservedSReg,
               const std::vector<uint32_t> &reservedVMask,
               const std::vector<uint32_t> &reservedPermit,
               uint32_t embeddingAddr,
               uint32_t embeddingHeight,
               uint32_t embeddingWidth,
               uint32_t indicesAddr,
               uint32_t indicesSize,
               uint32_t outputAddr);



// Height and Width of weight Should Align to 128
void Linear(std::vector<Instruction *> &instList,
            const std::vector<uint32_t> &reservedVReg,
            const std::vector<uint32_t> &reservedSReg,
            const std::vector<uint32_t> &reservedVMask,
            const std::vector<uint32_t> &reservedPermit,
            uint32_t inputAddr,
            const std::array<uint32_t, 4> &inputSize,
            uint32_t weightAddr,
            const std::array<uint32_t, 2> &weightSize,
            uint32_t outputAddr);

// For Width > 2048 and Height is 128
void LinearSplitWidth(std::vector<Instruction *> &instList,
                      const std::vector<uint32_t> &reservedVReg,
                      const std::vector<uint32_t> &reservedSReg,
                      const std::vector<uint32_t> &reservedVMask,
                      const std::vector<uint32_t> &reservedPermit,
                      uint32_t splitWidth,
                      uint32_t inputAddr,
                      const std::array<uint32_t, 4> &inputSize,
                      uint32_t weightAddr,
                      const std::array<uint32_t, 2> &weightSize,
                      uint32_t outputAddr);

// For Width > 2048 and Height > 512
void LinearSplit(std::vector<Instruction *> &instList,
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
                 uint32_t blockCount);

void LinearExVMemIO(std::vector<Instruction *> &instList,
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
                    uint32_t vmemOutputAddr);

void Padding(std::vector<Instruction *> &instList,
             uint32_t srcAddr,
             uint32_t width,
             uint32_t destAddr,
             uint32_t newWidth,
             uint32_t height);

void DePadding(std::vector<Instruction *> &instList,
               uint32_t srcAddr,
               uint32_t width,
               uint32_t destAddr,
               uint32_t newWidth,
               uint32_t height);

void LinearExHiVwVo(std::vector<Instruction *> &instList,
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
                    uint32_t vmemOutputAddr);

// support embeddingWidth: 1, 2, 4, 8, 16, 32, 64
void EmbeddingAny(std::vector<Instruction *> &instList,
                  const std::vector<uint32_t> &reservedVReg,
                  const std::vector<uint32_t> &reservedSReg,
                  const std::vector<uint32_t> &reservedVMask,
                  const std::vector<uint32_t> &reservedPermit,
                  uint32_t embeddingAddr,
                  uint32_t embeddingHeight,
                  uint32_t embeddingWidth,
                  uint32_t indicesAddr,
                  uint32_t indicesSize,
                  uint32_t outputAddr);

void EmbeddingEx(std::vector<Instruction *> &instList,
                 const std::vector<uint32_t> &reservedVReg,
                 const std::vector<uint32_t> &reservedSReg,
                 const std::vector<uint32_t> &reservedVMask,
                 const std::vector<uint32_t> &reservedPermit,
                 uint32_t hbmEmbeddingAddr,
                 uint32_t embeddingHeight,
                 uint32_t embeddingWidth,
                 uint32_t indicesAddr,
                 uint32_t indicesSize,
                 uint32_t outputAddr);

void MatMul(std::vector<Instruction *> &instList,
            const std::vector<uint32_t> &reservedVReg,
            const std::vector<uint32_t> &reservedSReg,
            const std::vector<uint32_t> &reservedVMask,
            const std::vector<uint32_t> &reservedPermit,
            uint32_t leftMatAddr,
            const std::array<uint32_t, 4> &leftMatSize,
            uint32_t rightMatAddr,
            const std::array<uint32_t, 4> &rightMatSize,
            uint32_t outputAddr);

void Softmax(std::vector<Instruction *> &instList,
             uint32_t input_addr,
             uint32_t output_addr,
             uint32_t num,
             uint32_t size);

void Dma(std::vector<Instruction *> &instList,
         uint16_t misc,
         uint32_t src_addr,
         uint32_t dest_addr,
         uint32_t length);

void DmaNonBlock(std::vector<Instruction *> &instList,
                 uint16_t misc,
                 uint32_t src_addr,
                 uint32_t dest_addr,
                 uint32_t length);

#endif
