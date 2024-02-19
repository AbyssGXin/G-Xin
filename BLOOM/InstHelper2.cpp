#include "InstHelper2.h"
#include <algorithm>
#include <cassert>
#include <deque>
#include <set>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <ortools/base/logging.h>
#include <ortools/sat/cp_model.h>
#include <ortools/sat/cp_model.pb.h>
#include <ortools/sat/cp_model_solver.h>

#define PLACEHINT __FILE__ + std::to_string(__LINE__)

const uint16_t DMA_DEST::HBM = 0b0000100000000000;
const uint16_t DMA_DEST::FXC = 0b0000100010000000;
const uint16_t DMA_SRC::HBM = 0b0000000100000000;
const uint16_t DMA_SRC::FXC = 0b0000000100100000;


SReg::SReg(int32_t id, bool isFloat, Inst2 *inst2) noexcept
    : id(id), isFloat(isFloat), inst2(inst2)
{
}

SReg &
SReg::operator=(const SReg &sReg) noexcept
{
    assert(inst2 == sReg.inst2);

    if (id != sReg.id)
    {
        isFloat = sReg.isFloat;
        (*inst2)(SMov, sReg.id, id);
    }

    return *this;
}

SReg::SReg(SReg &&sReg) noexcept
    : id(sReg.id), isFloat(sReg.isFloat), inst2(sReg.inst2)
{
    sReg.moved = true;
}

SReg &
SReg::operator=(SReg &&sReg) noexcept
{
    if (id != sReg.id)
    {
        inst2->FreeSReg(this);
    }
    id = sReg.id;
    isFloat = sReg.isFloat;
    inst2 = sReg.inst2;
    sReg.moved = true;

    return *this;
}

SReg::~SReg()
{
    if (!moved)
    {
        inst2->FreeSReg(this);
    }
}

void
SReg::operator=(const VRegPx &vRegPx)
{
    assert(vRegPx.inst2 == inst2);

    assert(false && "Unimplemented SReg = VRegPx");
}

void
SReg::operator=(int32_t value)
{
    isFloat = false;
    (*inst2)(SMov, inst2->inst.ImmeS(value), id);
}

void
SReg::operator=(uint32_t value)
{
    isFloat = false;
    (*inst2)(SMov, inst2->inst.ImmeU(value), id);
}

void
SReg::operator=(float value)
{
    isFloat = true;
    (*inst2)(SMov, inst2->inst.ImmeF(value), id);
}

void
SReg::operator+=(int32_t value)
{
    assert(isFloat == false);

    (*inst2)(SAddS, id, inst2->inst.ImmeS(value), id);
}

void
SReg::operator+=(uint32_t value)
{
    assert(isFloat == false);

    (*inst2)(SAddS, id, inst2->inst.ImmeU(value), id);
}

void
SReg::operator+=(float value)
{
    assert(isFloat == true);

    (*inst2)(SAddF, id, inst2->inst.ImmeF(value), id);
}

void
SReg::operator+=(const SReg &value)
{
    assert(isFloat == value.isFloat);

    if (isFloat)
    {
        (*inst2)(SAddF, id, value.id, id);
    }
    else
    {
        (*inst2)(SAddS, id, value.id, id);
    }
}

void
SReg::operator-=(int32_t value)
{
    assert(isFloat == false);

    (*inst2)(SSubS, id, inst2->inst.ImmeS(value), id);
}

void
SReg::operator-=(uint32_t value)
{
    assert(isFloat == false);

    (*inst2)(SSubS, id, inst2->inst.ImmeU(value), id);
}

void
SReg::operator-=(float value)
{
    assert(isFloat == true);

    (*inst2)(SSubF, id, inst2->inst.ImmeF(value), id);
}

void
SReg::operator-=(const SReg &value)
{
    assert(isFloat == value.isFloat);

    if (isFloat)
    {
        (*inst2)(SSubF, id, value.id, id);
    }
    else
    {
        (*inst2)(SSubS, id, value.id, id);
    }
}

SReg
SReg::operator+(int32_t value) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    (*inst2)(SAddS, id, inst2->inst.ImmeS(value), ans.id);

    return ans;
}

SReg
SReg::operator+(uint32_t value) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    (*inst2)(SAddS, id, inst2->inst.ImmeU(value), ans.id);

    return ans;
}

SReg
SReg::operator+(float value) const
{
    assert(isFloat == true);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    (*inst2)(SAddF, id, inst2->inst.ImmeF(value), ans.id);

    return ans;
}

SReg
SReg::operator+(const SReg &value) const
{
    assert(isFloat == value.isFloat);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    if (isFloat)
    {
        (*inst2)(SAddF, id, value.id, ans.id);
    }
    else
    {
        (*inst2)(SAddS, id, value.id, ans.id);
    }

    return ans;
}

SReg
SReg::operator-(int32_t value) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    (*inst2)(SAddS, id, inst2->inst.ImmeS(value), ans.id);

    return ans;
}

SReg
SReg::operator-(uint32_t value) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    (*inst2)(SAddS, id, inst2->inst.ImmeU(value), ans.id);

    return ans;
}

SReg
SReg::operator-(float value) const
{
    assert(isFloat == true);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    (*inst2)(SAddF, id, inst2->inst.ImmeF(value), ans.id);

    return ans;
}

SReg
SReg::operator-(const SReg &value) const
{
    assert(isFloat == value.isFloat);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    if (isFloat)
    {
        (*inst2)(SSubF, id, value.id, ans.id);
    }
    else
    {
        (*inst2)(SSubS, id, value.id, ans.id);
    }

    return ans;
}

SReg
SReg::operator*(int32_t value) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SMulU, id, inst2->inst.ImmeS(value), ans.id);

    return ans;
}

SReg
SReg::operator*(uint32_t value) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SMulU, id, inst2->inst.ImmeU(value), ans.id);

    return ans;
}

SReg
SReg::operator*(float value) const
{
    assert(isFloat == true);

    auto ans = inst2->AllocSReg();
    ans.isFloat = isFloat;

    (*inst2)(SMulF, id, inst2->inst.ImmeF(value), ans.id);

    return ans;
}

SReg
SReg::operator*(const SReg &value) const
{
    assert(isFloat == value.isFloat);

    auto ans = inst2->AllocSReg();

    (*inst2)(SMulU, id, value.id, ans.id);

    return ans;
}

void
SReg::operator*=(int32_t value) const
{
    assert(isFloat == false);

    (*inst2)(SMulU, id, inst2->inst.ImmeS(value), id);
}

void
SReg::operator*=(uint32_t value) const
{
    assert(isFloat == false);

    (*inst2)(SMulU, id, inst2->inst.ImmeU(value), id);
}

void
SReg::operator*=(float value) const
{
    assert(isFloat == true);

    (*inst2)(SMulF, id, inst2->inst.ImmeF(value), id);
}

void
SReg::operator*=(const SReg &value) const
{
    assert(isFloat == value.isFloat);

    (*inst2)(SMulU, id, value.id, id);
}

void
SReg::operator>>=(uint32_t shr)
{
    assert(isFloat == false);

    (*inst2)(SShrU, id, inst2->inst.ImmeU(shr), id);
}

SReg
SReg::operator>>(uint32_t shr) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SShrU, id, inst2->inst.ImmeU(shr), ans.id);

    return ans;
}

void
SReg::operator>>=(const SReg &shr)
{
    assert(isFloat == false);

    (*inst2)(SShrU, id, shr.id, id);
}

SReg
SReg::operator>>(const SReg &shr) const
{
    assert(isFloat == false);
    assert(shr.isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SShrU, id, shr.id, ans.id);

    return ans;
}

void
SReg::operator<<=(uint32_t shl)
{
    assert(isFloat == false);

    (*inst2)(SShlU, id, inst2->inst.ImmeU(shl), id);
}

SReg
SReg::operator<<(uint32_t shl) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SShlU, id, inst2->inst.ImmeU(shl), ans.id);

    return ans;
}

void
SReg::operator<<=(const SReg &shl)
{
    assert(isFloat == false);
    assert(shl.isFloat == false);

    (*inst2)(SShlU, id, shl.id, id);
}

SReg
SReg::operator<<(const SReg &shl) const
{
    assert(isFloat == false);
    assert(shl.isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SShlU, id, shl.id, ans.id);

    return ans;
}

SReg
SReg::operator&(const SReg &value) const
{
    assert(isFloat == false);
    assert(value.isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SAndU, id, value.id, ans.id);

    return ans;
}

SReg
SReg::operator&(uint32_t value) const
{
    assert(isFloat == false);

    auto ans = inst2->AllocSReg();

    (*inst2)(SAndU, id, inst2->inst.ImmeU(value), ans.id);

    return ans;
}

void
SReg::operator&=(const SReg &value) const
{
    assert(isFloat == false);

    (*inst2)(SAndU, id, value.id, id);
}

void
SReg::operator&=(uint32_t value) const
{
    assert(isFloat == false);

    (*inst2)(SAndU, id, inst2->inst.ImmeU(value), id);
}

void
SReg::operator|=(const SReg &value) const
{
    assert(isFloat == false);

    (*inst2)(SOrU, id, value.id, id);
}

void
SReg::operator|=(uint32_t value) const
{
    assert(isFloat == false);

    (*inst2)(SOrU, id, inst2->inst.ImmeU(value), id);
}

Range::Range(uint32_t start, uint32_t end) noexcept
    : startIdx(start), endIdx(end)
{
    assert(start <= end);
}

VReg::VReg(int32_t id, bool isFloat, Inst2 *inst2) noexcept
    : id(id), isFloat(isFloat), inst2(inst2)
{
}

VReg &
VReg::operator=(const VReg &vReg) noexcept
{
    assert(inst2 == vReg.inst2);

    if (id != vReg.id)
    {
        isFloat = vReg.isFloat;
        (*inst2)(VMov, vReg.id, id);
    }

    return *this;
};

VReg::VReg(VReg &&vReg) noexcept
    : id(vReg.id), isFloat(vReg.isFloat), inst2(vReg.inst2)
{
    vReg.moved = true;
}

VReg &
VReg::operator=(VReg &&vReg) noexcept
{
    if (id != vReg.id)
    {
        inst2->FreeVReg(this);
    }
    id = vReg.id;
    isFloat = vReg.isFloat;
    inst2 = vReg.inst2;

    vReg.moved = true;

    return *this;
}

VReg::~VReg()
{
    if (!moved)
    {
        inst2->FreeVReg(this);
    }
}

VRegPx
VReg::operator[](uint32_t idx) const
{
    return VRegPx(id, idx, isFloat, inst2);
}

VRegSlice
VReg::operator[](Range range) const
{
    assert(range.endIdx <= 1024);
    return VRegSlice(id, range.startIdx, range.endIdx, isFloat, inst2);
}

VRegSliceDyn
VReg::operator[](OffsetLenDyn range) const
{
    assert(inst2 == range.inst2);

    return VRegSliceDyn(id, range.offsetSRegId, range.len, isFloat, inst2);
}

void
VReg::operator=(const VMem &vMem)
{
    assert(vMem.inst2 == inst2);
    assert(vMem.len % 128 == 0 && vMem.len <= 1024);

    isFloat = vMem.isFloat;

    if (vMem.startAddr % 128 == 0)
    {
        (*inst2)(VLoadWithMask,
                 vMem.startAddr / 128,
                 id,
                 (1 << (vMem.len / 128)) - 1);
    }
    else
    {
        (*this)[Range(0, vMem.len)] = vMem;
    }
}

void
VReg::operator=(const VMemDyn &vMem)
{
    assert(inst2 == vMem.inst2);
    assert(vMem.len == 1024);

    isFloat = vMem.isFloat;

    auto addr = inst2->AllocSReg();
    (*inst2)(SAddS, vMem.addrSRegId, inst2->inst.ImmeU(vMem.baseAddr), addr.id);

    (*inst2)(VLoadBySReg, addr.id, id);
}

void
VReg::operator=(int32_t value)
{
    isFloat = false;
    (*inst2)(VMov, inst2->inst.ImmeS(value), id);
}

void
VReg::operator=(uint32_t value)
{
    isFloat = false;
    (*inst2)(VMov, inst2->inst.ImmeU(value), id);
}

void
VReg::operator=(float value)
{
    isFloat = true;
    (*inst2)(VMov, inst2->inst.ImmeF(value), id);
}

VMask
VReg::operator<(const VReg &vReg)
{
    assert(isFloat == vReg.isFloat);

    auto vMask = inst2->AllocVMask();

    if (!isFloat)
    {
        (*inst2)(VLsS, id, vReg.id, vMask.id);
    }
    else
    {
        (*inst2)(VLsF, id, vReg.id, vMask.id);
    }

    return vMask;
}

VMask
VReg::operator>=(const VReg &vReg)
{
    assert(isFloat == vReg.isFloat);

    auto vMask = inst2->AllocVMask();

    if (!isFloat)
    {
        (*inst2)(VGeS, id, vReg.id, vMask.id);
    }
    else
    {
        (*inst2)(VGeF, id, vReg.id, vMask.id);
    }

    return vMask;
}

void
VReg::operator+=(int32_t value)
{
    assert(isFloat == false);

    (*inst2)(VAddS, id, inst2->inst.ImmeS(value), id);
}

void
VReg::operator+=(uint32_t value)
{
    assert(isFloat == false);

    (*inst2)(VAddS, id, inst2->inst.ImmeU(value), id);
}

void
VReg::operator+=(float value)
{
    assert(isFloat == true);

    (*inst2)(VAddF, id, inst2->inst.ImmeF(value), id);
}

void
VReg::operator+=(const VReg &value)
{
    assert(isFloat == value.isFloat);

    if (isFloat)
    {
        (*inst2)(VAddF, id, value.id, id);
    }
    else
    {
        (*inst2)(VAddS, id, value.id, id);
    }
}

void
VReg::operator&=(uint32_t value)
{
    assert(isFloat == false);

    (*inst2)(VAndU, id, inst2->inst.ImmeU(value), id);
}

VRegSlice::VRegSlice(uint32_t id,
                     uint32_t startIdx,
                     uint32_t endIdx,
                     bool isFloat,
                     Inst2 *inst2) noexcept
    : id(id), startIdx(startIdx), endIdx(endIdx), isFloat(isFloat), inst2(inst2)
{
}

void
VRegSlice::operator=(const VRegSlice &vRegSlice)
{
    assert(vRegSlice.inst2 == inst2);
    assert(vRegSlice.endIdx - vRegSlice.startIdx == endIdx - startIdx);

    assert(false && "Unimplemented VRegSlice = VRegSlice");
}

void
VRegSlice::operator=(const VMem &vMem)
{
    assert(vMem.inst2 == inst2);
    assert(vMem.len == endIdx - startIdx);

    isFloat = vMem.isFloat;

    if (startIdx == 0 && vMem.startAddr % 128 == 0 && vMem.len == 1024)
    {
        // direct load
        (*inst2)(VLoad, vMem.startAddr / 128, id);
    }
    else if (startIdx == 0 && vMem.startAddr % 128 == 0)
    {
        // direct load just use v-mask and col-mask
        auto vRegSelPx = inst2->GetSelectPxMaskT();
        auto vRegEnd = inst2->AllocVReg(PLACEHINT);
        auto temp = inst2->AllocVReg(PLACEHINT);
        vRegEnd = endIdx;
        auto vMaskR = vRegSelPx < vRegEnd;
        (*inst2)(VLoadEx,
                 vMaskR.id,
                 inst2->inst.ImmeU(vMem.startAddr / 128),
                 temp.id,
                 (1 << ((vMem.len + 127) / 128)) - 1);
        (*inst2)(VSel, vMaskR.id, id, temp.id, id);
    }
    else if (vMem.startAddr % 128 == startIdx % 128)
    {
        // in this case, no need permute
        // vmem: [    ================]
        // vreg: [    ================]
        auto margin = startIdx / 128;
        bool requireRotate = vMem.startAddr / 128 < margin;
        //           v vmem start  v start
        // 01234567:1AAA4567 => 012AAA67
        //             ^ vmem end    ^ end
        int rotateCnt = startIdx / 128 - vMem.startAddr / 128;
        auto vRegSelPx = inst2->AllocVReg(PLACEHINT);
        vRegSelPx = inst2->GetSelectPxMaskT();
        auto vRegStart = inst2->AllocVReg(PLACEHINT);
        vRegStart = startIdx;
        auto vMaskL = vRegSelPx >= vRegStart;
        GC(std::move(vRegStart));
        auto vRegEnd = inst2->AllocVReg(PLACEHINT);
        vRegEnd = endIdx;
        auto vMaskR = vRegSelPx < vRegEnd;
        GC(std::move(vRegEnd));
        GC(std::move(vRegSelPx));
        vMaskR &= vMaskL;
        auto temp = inst2->AllocVReg(PLACEHINT);
        if (!requireRotate)
        {
            (*inst2)(VLoadEx,
                     vMaskR.id,
                     inst2->inst.ImmeU(vMem.startAddr / 128 - margin),
                     temp.id,
                     ((1 << ((endIdx + 127) / 128)) - 1) ^
                         ((1 << (startIdx / 128)) - 1));
        }
        else
        {

            (*inst2)(VLoadBySReg, inst2->inst.ImmeU(0), temp.id);
            for (int i = 0; i < rotateCnt; i++)
            {
                (*inst2)(VSubRotR, temp.id, temp.id);
            }
        }
        (*inst2)(VSel, vMaskR.id, id, temp.id, id);
    }
    else if (vMem.startAddr / 128 == (vMem.startAddr + vMem.len - 1) / 128)
    {
        // in this case, only need permute after load
        // vmem: [    ======      ]
        // vreg: [         ====== ]
        // when vmem: 01234567;0A234567 => vreg: 01234A67
        //                ^ read at here for align
        // if vmem: ^0A234567 => vreg: 01234A67
        //           ^ we can only read at here than rotateR 4 times
        // vmem: 01234A67 => vreg: 0A234567
        //           ^ read at here for align
        // if vmem: 01234A67:$ => vreg: 0A234567
        //          ^ we can only read at here than rotateL 4 times
        // !!Attention: last case unimplemented all
        auto margin = startIdx / 128;
        bool requireRotate = vMem.startAddr / 128 < margin;
        auto realRightMove =
            (int32_t)startIdx % 128 - (int32_t)vMem.startAddr % 128;
        auto circuleLeftMove = (128 - realRightMove) % 128;
        auto itoa = inst2->AllocVReg(PLACEHINT);
        itoa = inst2->GetSelectPxMaskT();
        auto begin = inst2->AllocVReg(PLACEHINT);
        begin = startIdx;
        auto vMaskL = itoa >= begin;
        GC(std::move(begin));
        auto end = inst2->AllocVReg(PLACEHINT);
        auto curEndIdx = std::min(endIdx, (startIdx / 128) * 128 + 128);
        end = curEndIdx;
        auto vMaskR = itoa < end;
        GC(std::move(end));
        vMaskR &= vMaskL;
        GC(std::move(vMaskL));
        itoa += circuleLeftMove;
        itoa &= 127;
        auto vData = inst2->AllocVReg(PLACEHINT);
        if (!requireRotate)
        {
            (*inst2)(VLoadBySRegWithMask,
                     inst2->inst.ImmeU(vMem.startAddr / 128 - margin),
                     vData.id,
                     1 << (startIdx / 128));
        }
        else
        {
            int rotateCnt = startIdx / 128 - vMem.startAddr / 128;
            (*inst2)(VLoadBySRegWithMask,
                     inst2->inst.ImmeU(0),
                     vData.id,
                     1 << (vMem.startAddr / 128));
            for (int i = 0; i < rotateCnt; i++)
            {
                (*inst2)(VSubRotR, vData.id, vData.id);
            }
        }
        (*inst2)(SetPermute, itoa.id);
        (*inst2)(VPermute, vData.id);
        (*inst2)(TrfOut, vData.id);
        (*inst2)(VSel, vMaskR.id, id, vData.id, id);
        GC(std::move(vMaskR));
        GC(std::move(vData));
        GC(std::move(itoa));
        if (endIdx != curEndIdx)
        {
            auto restLen = endIdx - curEndIdx;
            (*this)[Range(curEndIdx - startIdx, endIdx - startIdx)] =
                vMem[Range(vMem.len - restLen, vMem.len)];
        }
    }
    else
    {
        auto end = 128 - vMem.startAddr % 128;
        (*this)[Range(0, end)] = vMem[Range(0, end)];
        for (end += 128; end < vMem.len; end += 128)
        {
            (*this)[Range(end - 128, end)] = vMem[Range(end - 128, end)];
        }
        (*this)[Range(end - 128, vMem.len)] = vMem[Range(end - 128, vMem.len)];
    }
}

VRegPx
VRegSlice::operator[](uint32_t idx) const
{
    assert(idx < endIdx);
    return VRegPx(id, startIdx + idx, isFloat, inst2);
}

VRegSlice
VRegSlice::operator[](Range range) const
{
    assert(startIdx + range.endIdx <= endIdx);
    return VRegSlice(id,
                     startIdx + range.startIdx,
                     startIdx + range.endIdx,
                     isFloat,
                     inst2);
}

VRegPx::VRegPx(int32_t id, uint32_t idx, bool isFloat, Inst2 *inst2) noexcept
    : id(id), idx(idx), isFloat(isFloat), inst2(inst2)
{
    // Empty
}

VMem::VMem(uint32_t startAddr,
           uint32_t len,
           bool isFloat,
           Inst2 *inst2) noexcept
    : startAddr(startAddr), len(len), isFloat(isFloat), inst2(inst2)
{
    // Empty
}

VMem &
VMem::operator=(const VMem &vMem) noexcept
{
    assert(inst2 == vMem.inst2);
    assert(len == vMem.len);

    isFloat = vMem.isFloat;
    auto temp = inst2->AllocVReg("Move Temp", vMem.isFloat);

    // if src has common area with dest, copy start from end
    // Bug maybe here
    if (vMem.startAddr + vMem.len > startAddr && vMem.startAddr < startAddr)
    {
        if (startAddr % 128 == 0 && vMem.startAddr % 128 == 0 && len >= 1024)
        {
            auto times = (len / 1024) * 8;
            int idx = times - 8;
            while (idx >= 0)
            {
                temp = vMem[OffLen(idx * 128, 1024)];
                (*this)[OffLen(idx * 128, 1024)] = temp;
                idx -= 8;
            }
        }
        else
        {
            for (int i = len / 1024 - 1024; i >= 0; i--)
            {
                temp = vMem[Range(i * 1024, i * 1024 + 1024)];
                (*this)[Range(i * 1024, i * 1024 + 1024)] = temp;
            }
        }
        if (len % 1024 != 0)
        {
            temp[Range(0, len % 1024)] = vMem[Range((len / 1024) * 1024, len)];
            (*this)[Range((len / 1024) * 1024, len)] =
                temp[Range(0, len % 1024)];
        }
    }
    else
    {
        if (startAddr % 128 == 0 && vMem.startAddr % 128 == 0 && len >= 1024)
        {
            auto times = (len / 1024) * 8;
            auto idx = 0;
            while (idx < times)
            {
                temp = vMem[OffLen(idx * 128, 1024)];
                (*this)[OffLen(idx * 128, 1024)] = temp;
                idx += 8;
            }
        }
        else
        {
            for (int i = 0; i < len / 1024; i++)
            {
                temp = vMem[Range(i * 1024, i * 1024 + 1024)];
                (*this)[Range(i * 1024, i * 1024 + 1024)] = temp;
            }
        }
        if (len % 1024 != 0)
        {
            temp[Range(0, len % 1024)] = vMem[Range((len / 1024) * 1024, len)];
            (*this)[Range((len / 1024) * 1024, len)] =
                temp[Range(0, len % 1024)];
        }
    }

    return *this;
}

VMem::VMem(VMem &&vMem) noexcept
    : startAddr(vMem.startAddr), len(vMem.len), isFloat(vMem.isFloat),
      inst2(vMem.inst2), owned(vMem.owned)
{
    vMem.owned = false;
}

VMem &
VMem::operator=(VMem &&vMem) noexcept
{
    startAddr = vMem.startAddr;
    len = vMem.len;
    isFloat = vMem.isFloat;
    inst2 = vMem.inst2;
    owned = vMem.owned;
    vMem.owned = false;

    return *this;
}

VMem::~VMem()
{
    if (owned)
    {
        // inst2->FreeVMem(this);
    }
}

VMem
VMem::operator[](Range range) const
{
    auto newLen = range.endIdx - range.startIdx;
    assert(len >= newLen);

    auto vmem = VMem(startAddr + range.startIdx, newLen, isFloat, inst2);
    vmem.owned = false;
    return vmem;
}

VMemDyn
VMem::operator[](OffsetLenDyn range) const
{
    assert(startAddr % 128 == 0 && "Only support 128 aligned address");
    assert(range.len % 128 == 0 && "Only support 128 aligned size");
    assert(inst2 == range.inst2);

    return VMemDyn(startAddr / 128,
                   range.offsetSRegId,
                   range.len,
                   isFloat,
                   inst2);
}

void
VMem::operator=(const VReg &vReg)
{
    assert(vReg.inst2 == inst2);
    assert(len % 128 == 0 && len <= kNumberOfSubcores);

    if (startAddr % kNumberOfCores == 0)
    {
        (*inst2)(VStoreWithMask,
                 vReg.id,
                 startAddr / kNumberOfCores,
                 (1 << (len / 128)) - 1);
    }
    else
    {
        *this = vReg[Range(0, kNumberOfSubcores)];
    }
}

void
VMem::operator=(const VRegSlice &vRegSlice)
{
    assert(vRegSlice.inst2 == inst2);
    assert(len == vRegSlice.endIdx - vRegSlice.startIdx);

    if (len == 0)
    {
        return;
    }

    isFloat = vRegSlice.isFloat;

    if (vRegSlice.startIdx == 0 && startAddr % 128 == 0 && len % 128 == 0)
    {
        (*inst2)(VStoreWithMask,
                 vRegSlice.id,
                 startAddr / 128,
                 (1 << (len / 128)) - 1);
    }
    else if (vRegSlice.startIdx == 0 && startAddr % 128 == 0)
    {
        auto vRegSelPx = inst2->AllocVReg(PLACEHINT);
        vRegSelPx = inst2->GetSelectPxMaskT();
        auto vRegEnd = inst2->AllocVReg(PLACEHINT);
        vRegEnd = vRegSlice.endIdx;
        auto vMaskR = vRegSelPx < vRegEnd;
        (*inst2)(VStoreEx,
                 vMaskR.id,
                 vRegSlice.id,
                 inst2->inst.ImmeU(startAddr / 128),
                 (1 << ((len + 127) / 128)) - 1);
    }
    else if (startAddr % 128 == 0 && vRegSlice.startIdx % 128 == 0)
    {
        // in this case, no need permute
        auto margin = vRegSlice.startIdx / 128;
        bool requireRotate = startAddr / 128 < margin;
        // start is both align to 128, no need mask the start
        if (vRegSlice.endIdx % 128 == 0)
        {
            // end is also align to 128, we can only use st_mask
            //          v vreg start    v start
            // vreg: 012AA567 => vmem: 0AA34567
            //           ^ vreg end      ^ end
            // vmem: 01234567:0AA34567
            //       verg: 01 2AA567
            if (!requireRotate)
            {
                (*inst2)(VStoreBySRegWithMask,
                         vRegSlice.id,
                         inst2->inst.ImmeU(startAddr / 128 - margin),
                         ((1 << ((vRegSlice.endIdx + 127) / 128)) - 1) ^
                             ((1 << (vRegSlice.startIdx / 128)) - 1));
            }
            else
            {
                VReg temp = inst2->AllocVReg("VSubRot");
                int rotLCnt = vRegSlice.startIdx / 128 - startAddr / 128;
                (*inst2)(VMov, vRegSlice.id, temp.id);
                for (int i = 0; i < rotLCnt; i++)
                {
                    (*inst2)(VSubRotL, temp.id, temp.id);
                }
                (*inst2)(VStoreBySRegWithMask,
                         temp.id,
                         inst2->inst.ImmeU(0),
                         ((1 << ((startAddr + len + 127) / 128)) - 1) ^
                             ((1 << (startAddr / 128)) - 1));
            }
        }
        else
        {
            // in this case, end not align to 128, we need mask the end
            auto vRegSelPx = inst2->AllocVReg(PLACEHINT);
            vRegSelPx = inst2->GetSelectPxMaskT();
            auto vRegEnd = inst2->AllocVReg(PLACEHINT);
            if (!requireRotate)
            {
                vRegEnd = vRegSlice.endIdx;
                auto vMask = vRegSelPx < vRegEnd;
                (*inst2)(VStoreEx,
                         vMask.id,
                         vRegSlice.id,
                         inst2->inst.ImmeU(startAddr / 128 - margin),
                         ((1 << ((vRegSlice.endIdx + 127) / 128)) - 1) ^
                             ((1 << (vRegSlice.startIdx / 128)) - 1));
            }
            else
            {
                vRegEnd = startAddr + len;
                auto vMask = vRegSelPx < vRegEnd;
                VReg temp = inst2->AllocVReg("VSubRot");
                int rotLCnt = vRegSlice.startIdx / 128 - startAddr / 128;
                (*inst2)(VMov, vRegSlice.id, temp.id);
                for (int i = 0; i < rotLCnt; i++)
                {
                    (*inst2)(VSubRotL, temp.id, temp.id);
                }
                (*inst2)(VStoreEx,
                         vMask.id,
                         temp.id,
                         inst2->inst.ImmeU(0),
                         ((1 << ((startAddr + len + 127) / 128)) - 1) ^
                             ((1 << (startAddr / 128)) - 1));
            }
        }
    }
    else if (startAddr % 128 == vRegSlice.startIdx % 128)
    {
        // in this case, start is not align to 128, we need mask start
        // if requireRotate, the vmem must reach the start, so startAddr no need
        // calc mod in that case, after rotate, vreg addr is just vmem addr
        auto margin = vRegSlice.startIdx / 128;
        bool requireRotate = startAddr / 128 < margin;
        auto vRegSelPx = inst2->AllocVReg(PLACEHINT);
        vRegSelPx = inst2->GetSelectPxMaskT();
        auto vRegStart = inst2->AllocVReg(PLACEHINT);
        if (!requireRotate)
        {
            vRegStart = vRegSlice.startIdx;
        }
        else
        {
            vRegStart = startAddr;
        }
        auto vMaskL = vRegSelPx >= vRegStart;
        if (vRegSlice.endIdx % 128 == 0)
        {
            if (!requireRotate)
            {
                (*inst2)(VStoreEx,
                         vMaskL.id,
                         vRegSlice.id,
                         inst2->inst.ImmeU(startAddr / 128 - margin),
                         ((1 << ((vRegSlice.endIdx + 127) / 128)) - 1) ^
                             ((1 << (vRegSlice.startIdx / 128)) - 1));
            }
            else
            {
                int rotLCnt = vRegSlice.startIdx / 128 - startAddr / 128;
                VReg temp = inst2->AllocVReg("RotTemp");
                (*inst2)(VMov, vRegSlice.id, temp.id);
                for (int i = 0; i < rotLCnt; i++)
                {
                    (*inst2)(VSubRotL, temp.id, temp.id);
                }
                (*inst2)(VStoreEx,
                         vMaskL.id,
                         temp.id,
                         inst2->inst.ImmeU(0),
                         ((1 << ((startAddr + len + 127) / 128)) - 1) ^
                             ((1 << (startAddr / 128)) - 1));
            }
        }
        else
        {
            auto vRegEnd = inst2->AllocVReg(PLACEHINT);
            if (!requireRotate)
            {
                vRegEnd = vRegSlice.endIdx;
            }
            else
            {
                vRegEnd = startAddr + len;
            }
            auto vMaskR = vRegSelPx < vRegEnd;
            vMaskR &= vMaskL;
            if (!requireRotate)
            {
                (*inst2)(VStoreEx,
                         vMaskR.id,
                         vRegSlice.id,
                         inst2->inst.ImmeU(startAddr / 128 - margin),
                         ((1 << ((vRegSlice.endIdx + 127) / 128)) - 1) ^
                             ((1 << (vRegSlice.startIdx / 128)) - 1));
            }
            else
            {
                int rotLCnt = vRegSlice.startIdx / 128 - startAddr / 128;
                VReg temp = inst2->AllocVReg("RotTemp");
                (*inst2)(VMov, vRegSlice.id, temp.id);
                for (int i = 0; i < rotLCnt; i++)
                {
                    (*inst2)(VSubRotL, temp.id, temp.id);
                }
                (*inst2)(VStoreEx,
                         vMaskR.id,
                         temp.id,
                         inst2->inst.ImmeU(0),
                         ((1 << ((startAddr + len + 127) / 128)) - 1) ^
                             ((1 << (startAddr / 128)) - 1));
            }
        }
    }
    else if (vRegSlice.startIdx / 128 == (vRegSlice.endIdx - 1) / 128)
    {
        // in this case,
        // if vreg: [    ========][            ]
        //    vmem: [       =====][===         ]
        // we can only spilt it to two operation
        // first vreg: [    =====   ] then vreg: [         ===]
        //    to vmem: [       =====]   to vmem: [===         ]

        auto vRegSelPx = inst2->AllocVReg(PLACEHINT);
        vRegSelPx = inst2->GetSelectPxMaskT();
        auto offset = startAddr % 128;
        auto itoa = inst2->AllocVReg(PLACEHINT);
        auto realMove = (int32_t)vRegSlice.startIdx % 128 - (int32_t)offset;
        auto move = (realMove + 128) % 128;
        itoa = inst2->GetSelectPxMaskT();
        itoa += move;
        itoa &= 127;
        (*inst2)(SetPermute, itoa.id);
        (*inst2)(VPermute, vRegSlice.id);
        auto newData = inst2->AllocVReg(PLACEHINT);
        (*inst2)(TrfOut, newData.id);
        auto margin = (vRegSlice.startIdx / 128);
        bool requireRotate = startAddr / 128 < margin;
        auto end = inst2->AllocVReg(PLACEHINT);
        int rotLCnt = vRegSlice.startIdx / 128 - startAddr / 128;
        end = vRegSlice.endIdx - realMove - (requireRotate ? rotLCnt * 128 : 0);
        auto begin = inst2->AllocVReg(PLACEHINT);
        begin =
            vRegSlice.startIdx - realMove - (requireRotate ? rotLCnt * 128 : 0);
        auto vMaskR = vRegSelPx < end;
        auto vMaskL = vRegSelPx >= begin;
        vMaskL &= vMaskR;

        // as the code, it not check this margin
        // assert(startAddr / 128 >= margin);

        if (!requireRotate)
        {
            (*inst2)(VStoreEx,
                     vMaskL.id,
                     newData.id,
                     inst2->inst.ImmeU(startAddr / 128 - margin),
                     1 << margin);
        }
        else
        {
            VReg temp = inst2->AllocVReg("VRot");
            (*inst2)(VMov, newData.id, temp.id);
            for (int i = 0; i < rotLCnt; i++)
            {
                (*inst2)(VSubRotL, temp.id, temp.id);
            }
            (*inst2)(VStoreEx,
                     vMaskL.id,
                     temp.id,
                     inst2->inst.ImmeU(0),
                     1 << (startAddr / 128));
        }
        if ((((vRegSlice.endIdx - 1) % 128) - realMove) >= 128)
        {
            (*this)[Range(128 - startAddr % 128, len)] =
                newData[Range(margin * 128, vRegSlice.endIdx - move)];
        }
    }
    else if (len <= 128)
    {
        int off = (vRegSlice.startIdx + 127) / 128 * 128 - vRegSlice.startIdx;
        (*this)[Range(0, off)] = vRegSlice[Range(0, off)];
        (*this)[Range(off, len)] = vRegSlice[Range(off, len)];
    }
    else
    {
        auto end = 128;
        for (; end < len; end += 128)
        {
            (*this)[Range(end - 128, end)] = vRegSlice[Range(end - 128, end)];
        }
        (*this)[Range(end - 128, len)] = vRegSlice[Range(end - 128, len)];
    }
}

void
VMemFillVReg(VMem &self, const VReg &val)
{
    auto copyTimes = self.len / 1024;
    auto idx = 0;
    auto times = copyTimes;

    while (idx < times)
    {
        auto addr = idx * 8;
        addr += self.startAddr / 128;
        (*self.inst2)(VStore, val.id, addr);
        idx += 1;
    }

    if (self.len % 1024 != 0)
    {
        self[Range(copyTimes * 1024, self.len)] =
            val[Range(0, self.len % 1024)];
    }
}

void
VMem::operator=(int32_t value)
{
    auto val = inst2->AllocVReg("Val");
    val = value;

    isFloat = false;

    VMemFillVReg(*this, val);
}

void
VMem::operator=(uint32_t value)
{
    auto val = inst2->AllocVReg("Val");
    val = value;

    isFloat = false;

    VMemFillVReg(*this, val);
}

void
VMem::operator=(float value)
{
    auto val = inst2->AllocVReg("Val");
    val = value;

    isFloat = true;

    VMemFillVReg(*this, val);
}

void
VMem::operator+=(const VMem &vmem)
{
    assert(len == vmem.len);
    assert(isFloat == vmem.isFloat);

    VReg a = inst2->AllocVReg("Val Left");
    VReg b = inst2->AllocVReg("Val Right");

    a.isFloat = isFloat;
    b.isFloat = isFloat;

    uint32_t i = 0;
    for (; i < len / 1024; i++)
    {
        a = (*this)[Range(i * 1024, (i + 1) * 1024)];
        b = vmem[Range(i * 1024, (i + 1) * 1024)];
        a += b;
        (*this)[Range(i * 1024, (i + 1) * 1024)] = a;
    }
    if (len % 1024 != 0)
    {
        a[Range(0, len % 1024)] = (*this)[Range(i * 1024, len)];
        b[Range(0, len % 1024)] = vmem[Range(i * 1024, len)];
        a += b;
        (*this)[Range(i * 1024, len)] = a[Range(0, len % 1024)];
    }
}

void
VMem::operator*=(float value)
{
    VReg a = inst2->AllocVReg("Val Left");

    a.isFloat = isFloat;

    uint32_t i = 0;
    for (; i < len / 1024; i++)
    {
        a = (*this)[Range(i * 1024, (i + 1) * 1024)];
        (*inst2)(VMulF, a.id, inst2->inst.ImmeF(value), a.id);
        (*this)[Range(i * 1024, (i + 1) * 1024)] = a;
    }
    if (len % 1024 != 0)
    {
        a[Range(0, len % 1024)] = (*this)[Range(i * 1024, len)];
        (*inst2)(VMulF, a.id, inst2->inst.ImmeF(value), a.id);
        (*this)[Range(i * 1024, len)] = a[Range(0, len % 1024)];
    }
}

void
VMem::BinaryOpSelfAssign(
    const VMem &vmem,
    std::function<void(VReg &ina, VReg &inb, VReg &out)> op)
{
    uint32_t i = 0;
    for (; i < len / 1024; i++)
    {
        VReg a = inst2->AllocVReg("");
        VReg b = inst2->AllocVReg("");
        VReg out = inst2->AllocVReg("");
        a = (*this)[Range(i * 1024, (i + 1) * 1024)];
        b = vmem[Range(i * 1024, (i + 1) * 1024)];
        op(a, b, out);
        (*this)[Range(i * 1024, (i + 1) * 1024)] = out;
    }
    if (len % 1024 != 0)
    {
        VReg a = inst2->AllocVReg("");
        VReg b = inst2->AllocVReg("");
        VReg out = inst2->AllocVReg("");
        a[Range(0, len % 1024)] = (*this)[Range(i * 1024, len)];
        b[Range(0, len % 1024)] = vmem[Range(i * 1024, len)];
        op(a, b, out);
        (*this)[Range(i * 1024, len)] = out[Range(0, len % 1024)];
    }
}

void
VMem::BinaryOp(const VMem &vmem,
               const VMem &outVmem,
               std::function<void(VReg &ina, VReg &inb, VReg &out)> op) const
{
    uint32_t i = 0;
    for (; i < len / 1024; i++)
    {
        VReg a = inst2->AllocVReg("");
        VReg b = inst2->AllocVReg("");
        VReg out = inst2->AllocVReg("");
        a = (*this)[Range(i * 1024, (i + 1) * 1024)];
        b = vmem[Range(i * 1024, (i + 1) * 1024)];
        op(a, b, out);
        outVmem[Range(i * 1024, (i + 1) * 1024)] = out;
    }
    if (len % 1024 != 0)
    {
        VReg a = inst2->AllocVReg("");
        VReg b = inst2->AllocVReg("");
        VReg out = inst2->AllocVReg("");
        a[Range(0, len % 1024)] = (*this)[Range(i * 1024, len)];
        b[Range(0, len % 1024)] = vmem[Range(i * 1024, len)];
        op(a, b, out);
        outVmem[Range(i * 1024, len)] = out[Range(0, len % 1024)];
    }
}

VMask::VMask(int32_t id, Inst2 *inst2) noexcept : id(id), inst2(inst2) {}

VMask::VMask(VMask &&vMask) noexcept : id(vMask.id), inst2(vMask.inst2)
{
    vMask.moved = true;
}

VMask &
VMask::operator=(VMask &&vMask) noexcept
{
    if (id != vMask.id)
    {
        inst2->FreeVMask(this);
    }
    id = vMask.id;
    inst2 = vMask.inst2;
    vMask.moved = true;
    return *this;
}

VMask::~VMask()
{
    if (!moved)
    {
        inst2->FreeVMask(this);
    }
}

void
VMask::operator&=(const VMask &vMask)
{
    bool useMiscOp = false;
    if (!useMiscOp)
    {
        auto const0 = inst2->AllocVReg(PLACEHINT);
        (*inst2)(VMov, CONST_U32_0, const0.id);
        auto maskV1 = inst2->AllocVReg(PLACEHINT);
        auto maskV2 = inst2->AllocVReg(PLACEHINT);
        (*inst2)(VSel, id, const0.id, CONST_U32_1, maskV1.id);
        (*inst2)(VSel, vMask.id, const0.id, CONST_U32_1, maskV2.id);
        (*inst2)(VAndU, maskV1.id, maskV2.id, maskV2.id);
        (*inst2)(VEqS, maskV2.id, CONST_U32_1, id);
    }
    else
    {
        (*inst2)(MVMaskAnd, vMask.id, id);
    }
}

OffsetLenDyn::OffsetLenDyn(const SReg &sReg, uint32_t len) noexcept
    : offsetSRegId(sReg.id), inst2(sReg.inst2), len(len)
{
}

void
VMemDyn::operator=(const VReg &vReg)
{
    assert(len == 1024);
    assert(inst2 == vReg.inst2);

    auto addr = inst2->AllocSReg();
    (*inst2)(SAddS, addrSRegId, inst2->inst.ImmeU(baseAddr), addr.id);

    (*inst2)(VStoreBySReg, vReg.id, addr.id);
}

VMemDyn::VMemDyn(uint32_t baseAddr,
                 uint32_t addrSRegId,
                 uint32_t len,
                 bool isFloat,
                 Inst2 *inst2) noexcept
    : baseAddr(baseAddr), addrSRegId(addrSRegId), len(len), isFloat(isFloat),
      inst2(inst2)
{
}

VRegSliceDyn::VRegSliceDyn(uint32_t id,
                           uint32_t startIdxSRegId,
                           uint32_t len,
                           bool isFloat,
                           Inst2 *inst2) noexcept
    : id(id), startIdxSRegId(startIdxSRegId), len(len), isFloat(isFloat),
      inst2(inst2)
{
}

void
Inst2::PushResource()
{
    resourcesStash.push_back(resource);
    resource = Resource();
}

void
Inst2::ForkAndPushResource()
{
    resourcesStash.push_back(resource);
}

void
Inst2::PopResource()
{
    resource = resourcesStash.back();
    resourcesStash.pop_back();
}

VReg
Inst2::AllocVReg(const std::string &hint)
{
    curUsingVReg++;
    maxUsingVReg = std::max(curUsingVReg, maxUsingVReg);
    auto id = resource.AllocVReg();
    if (logLevel >= DebugLevel::Info)
    {
        std::clog << hint << " V#" << id << "\n";
    }
    return VReg(id, false, this);
}

VReg
Inst2::AllocVReg(const std::string &name, bool isFloat)
{
    auto vReg = AllocVReg(name);
    vReg.isFloat = isFloat;
    vRegSymbolTable[vReg.id] = std::make_pair(name, isFloat);
    return vReg;
}

SReg
Inst2::AllocSReg()
{
    return SReg(resource.AllocSReg(), false, this);
}

VMask
Inst2::AllocVMask()
{
    return VMask(resource.AllocVMask(), this);
}

void
Inst2::FreeVReg(VReg *vReg)
{
    curUsingVReg--;
    resource.FreeVReg(vReg->id);
}

void
Inst2::FreeSReg(SReg *sReg)
{
    resource.FreeSReg(sReg->id);
}

void
Inst2::FreeVMask(VMask *vMask)
{
    resource.FreeVMask(vMask->id);
}

VMem
Inst2::Alloc(uint32_t size, const std::string &usage)
{
    auto addr = resource.AllocVMem(size, 1, usage);
    if (logLevel >= DebugLevel::Info)
    {
        std::clog << "Alloc VMem size " << size << " to " << addr << "\n";
    }
    VMem vMem(addr, size, false, this);
    return vMem;
}

VMem
Inst2::AllocF(uint32_t size, const std::string &usage)
{
    auto addr = resource.AllocVMem(size, 1, usage);
    if (logLevel >= DebugLevel::Info)
    {
        std::clog << "Alloc VMem size " << size << " to " << addr << "\n";
    }
    VMem vMem(addr, size, true, this);
    return vMem;
}

VMem
Inst2::Alloc(uint32_t size, uint32_t align, const std::string &usage)
{
    auto addr = resource.AllocVMem(size, align, usage);
    if (logLevel >= DebugLevel::Info)
    {
        std::clog << "Alloc VMem size " << size << " align " << align << " to "
                  << addr << "\n";
    }
    VMem vMem(addr, size, false, this);
    return vMem;
}

VMem
Inst2::AllocF(uint32_t size, uint32_t align, const std::string &usage)
{
    auto addr = resource.AllocVMem(size, align, usage);
    if (logLevel >= DebugLevel::Info)
    {
        std::clog << "Alloc VMem size " << size << " align " << align << " to "
                  << addr << "\n";
    }
    VMem vMem(addr, size, true, this);
    return vMem;
}

void
Inst2::FreeVMem(VMem *vmem)
{
    if (logLevel >= DebugLevel::Info)
    {
        std::clog << "Free VMem " << vmem->startAddr << " size " << vmem->len
                  << "\n";
    }
    resource.FreeVMem(vmem->startAddr, vmem->len);
}

VReg
Inst2::GetSelectPxMaskT()
{
    auto reg = AllocVReg("SelectPxMask");
    (*this)(VCoreId, reg.id);
    return reg;
}

void
Inst2::operator()(std::function<void(Instruction *)> op)
{
    inst(std::function<void(Instruction *)>(op));
}

void
Inst2::operator()(std::function<void(Instruction *, int)> op, int p0)
{
    inst(std::function<void(Instruction *, int)>(op), p0);
}

void
Inst2::operator()(std::function<void(Instruction *, int, int)> op,
                  int p0,
                  int p1)
{
    inst(std::function<void(Instruction *, int, int)>(op), p0, p1);
}

void
Inst2::operator()(std::function<void(Instruction *, int, int, int)> op,
                  int p0,
                  int p1,
                  int p2)
{
    inst(std::function<void(Instruction *, int, int, int)>(op), p0, p1, p2);
}

void
Inst2::operator()(std::function<void(Instruction *, int, int, int, int)> op,
                  int p0,
                  int p1,
                  int p2,
                  int p3)
{
    inst(std::function<void(Instruction *, int, int, int, int)>(op),
         p0,
         p1,
         p2,
         p3);
}

void
Inst2::operator()(
    std::function<void(Instruction *, int, int, int, int, int)> op,
    int p0,
    int p1,
    int p2,
    int p3,
    int p4)
{
    inst(std::function<void(Instruction *, int, int, int, int, int)>(op),
         p0,
         p1,
         p2,
         p3,
         p4);
}

void
Inst2::operator()(
    std::function<void(Instruction *, int, int, int, int, int, int)> op,
    int p0,
    int p1,
    int p2,
    int p3,
    int p4,
    int p5)
{
    inst(std::function<void(Instruction *, int, int, int, int, int, int)>(op),
         p0,
         p1,
         p2,
         p3,
         p4,
         p5);
}

void
Inst2::operator()(
    std::function<void(Instruction *, int, int, int, int, int, int, int)> op,
    int p0,
    int p1,
    int p2,
    int p3,
    int p4,
    int p5,
    int p6)
{
    inst(std::function<void(Instruction *, int, int, int, int, int, int, int)>(
             op),
         p0,
         p1,
         p2,
         p3,
         p4,
         p5,
         p6);
}

void
Inst2::operator()(std::function<void(Inst &)> op)
{
    inst.Asm(std::function<void(Inst &)>(op));
}

void
Inst2::operator()(std::function<void(Inst &, int)> op, int p0)
{
    inst.Asm(std::function<void(Inst &, int)>(op), p0);
}

void
Inst2::operator()(std::function<void(Inst &, int, int)> op, int p0, int p1)
{
    inst.Asm(std::function<void(Inst &, int, int)>(op), p0, p1);
}

void
Inst2::operator()(std::function<void(Inst &, int, int, int)> op,
                  int p0,
                  int p1,
                  int p2)
{
    inst.Asm(std::function<void(Inst &, int, int, int)>(op), p0, p1, p2);
}

void
Inst2::operator()(std::function<void(Inst &, int, int, int, int)> op,
                  int p0,
                  int p1,
                  int p2,
                  int p3)
{
    inst.Asm(std::function<void(Inst &, int, int, int, int)>(op),
             p0,
             p1,
             p2,
             p3);
}

void
Inst2::operator()(std::function<void(Inst &, int, int, int, int, int)> op,
                  int p0,
                  int p1,
                  int p2,
                  int p3,
                  int p4)
{
    inst.Asm(std::function<void(Inst &, int, int, int, int, int)>(op),
             p0,
             p1,
             p2,
             p3,
             p4);
}

void
Inst2::operator()(std::function<void(Inst &, int, int, int, int, int, int)> op,
                  int p0,
                  int p1,
                  int p2,
                  int p3,
                  int p4,
                  int p5)
{
    inst.Asm(std::function<void(Inst &, int, int, int, int, int, int)>(op),
             p0,
             p1,
             p2,
             p3,
             p4,
             p5);
}

void
Inst2::operator()(
    std::function<void(Inst &, int, int, int, int, int, int, int)> op,
    int p0,
    int p1,
    int p2,
    int p3,
    int p4,
    int p5,
    int p6)
{
    inst.Asm(std::function<void(Inst &, int, int, int, int, int, int, int)>(op),
             p0,
             p1,
             p2,
             p3,
             p4,
             p5,
             p6);
}

void
Inst2::Spy(std::string name, const VMem &vmem)
{
}

void
Inst2::Break(std::string name)
{
}

std::vector<Instruction *>
GetInstructions(Inst2 &self)
{
    if (self.useScheduler)
    {
        auto res = schedule(self.inst.insts, self.showSchedulerVerboseInfo);
        self.schedulerEffectLog.emplace_back(self.inst.insts.size(),
                                             res.size());
        return res;
    }
    return std::forward<std::vector<Instruction *>>(self.inst.insts);
}

void
ExecBundle(Inst2 &self, bool keepLastBundle = false)
{
    if (self.execFunc == nullptr || self.inst.insts.empty())
    {
        return;
    }
    auto bundles = InstructionsSpilt(GetInstructions(self),
                                     self.bundleInstSize,
                                     self.spies);
    for (size_t i = 0; i < bundles.size() - (keepLastBundle ? 1 : 0); i++)
    {
        auto &b = bundles[i];
        std::vector<SpyInfo> spyInfos;
        auto range = self.spies.equal_range(b.back());
        for (auto &it = range.first; it != range.second; ++it)
        {
            spyInfos.push_back(it->second);
        }
        self.spies.erase(b.back());
        std::clog << i << "/" << (bundles.size() - 1) << "> ";
        self.execFunc(b, spyInfos);
    }
    if (self.useScheduler)
    {
        for (const auto &i : self.inst.insts)
        {
            delete i;
        }
    }
    self.inst.insts.clear();
    if (keepLastBundle)
    {
        self.inst.insts.assign(bundles.back().begin(), bundles.back().end());
    }
}

void
Inst2::Spy(std::string name,
           const VMem &vmem,
           std::string filepath,
           bool forcePrint)
{
    ExecBundle(*this, true);
    spies.insert(
        std::make_pair(inst.insts.back(),
                       SpyInfo{name, vmem.startAddr, vmem.len, filepath}));
}

void
Inst2::Exec()
{
    ExecBundle(*this);
    spies.clear();
}

void
Inst2::Mock(std::string name, const VMem &vmem, std::string filepath)
{
}

void
Memcopy(const VMem &vmsrc, VMem &&vmdest)
{
    vmdest = vmsrc;
}

void
Memcopy(const VMem &vmsrc, VMem &vmdest)
{
    vmdest = vmsrc;
}

std::vector<std::vector<Instruction *>>
InstructionsSpilt(const std::vector<Instruction *> &instruction_lists,
                  int threshold)
{
    std::vector<bool> canBeSplit(instruction_lists.size(), true);
    for (size_t i = 0; i < instruction_lists.size(); i++)
    {
        auto inst = instruction_lists[i];
        auto S0Slot = inst->GetOperation(Instruction::SCALARONE);
        auto opcode = S0Slot->GetOpCode();
        if (opcode == S_BRANCH)
        {
            auto op = static_cast<ScalarOperationState *>(S0Slot);
            assert((op->GetIndexDest() == 0 || op->GetIndexDest() == 1) &&
                   "Only Support brabs And brrel");
            size_t destPc = i;
            auto imm0 =
                (int16_t)inst->GetImmediateValue(Instruction::IMMEDIATE0);
            if (op->GetIndexDest() == 0)
            {
                destPc = imm0;
            }
            else if (op->GetIndexDest() == 1)
            {
                destPc = i + imm0;
            }
            size_t begin = std::min(i, destPc);
            size_t end = std::max(i, destPc);
            for (size_t j = begin; j <= end; j++)
            {
                canBeSplit[j] = false;
            }
        }
        else if (opcode == S_CALL_ABOSOLUTE || opcode == S_CALL_REGISTER ||
                 opcode == S_CALL_RELATIVE)
        {
            assert(false && "Not Support call");
        }
    }

    std::vector<std::vector<Instruction *>> bundles;
    size_t curPc = 0;
    while (true)
    {
        size_t nextPc = curPc + threshold;
        if (nextPc >= instruction_lists.size() - 1)
        {
            nextPc = instruction_lists.size() - 1;
            std::vector<Instruction *> bundle;
            bundle.assign(instruction_lists.begin() + curPc,
                          instruction_lists.end());
            bundles.emplace_back(bundle);
            break;
        }
        while (!canBeSplit[nextPc] && nextPc > curPc)
        {
            nextPc--;
        }
        assert(nextPc != curPc && "No Place To Split");
        std::vector<Instruction *> bundle;
        bundle.assign(instruction_lists.begin() + curPc,
                      instruction_lists.begin() + nextPc);
        bundles.emplace_back(bundle);
        curPc = nextPc;
    }
    return bundles;
}

std::vector<std::vector<Instruction *>>
InstructionsSpilt(const std::vector<Instruction *> &instruction_lists,
                  int threshold,
                  const std::multimap<Instruction *, SpyInfo> &spiltHint)
{
    {
        std::vector<bool> canBeSplit(instruction_lists.size(), true);
        for (size_t i = 0; i < instruction_lists.size(); i++)
        {
            auto inst = instruction_lists[i];
            auto S0Slot = inst->GetOperation(Instruction::SCALARONE);
            auto opcode = S0Slot->GetOpCode();
            if (opcode == S_BRANCH)
            {
                auto op = static_cast<ScalarOperationState *>(S0Slot);
                assert((op->GetIndexDest() == 0 || op->GetIndexDest() == 1) &&
                       "Only Support brabs And brrel");
                size_t destPc = i;
                auto imm0 =
                    (int16_t)inst->GetImmediateValue(Instruction::IMMEDIATE0);
                if (op->GetIndexDest() == 0)
                {
                    destPc = imm0;
                }
                else if (op->GetIndexDest() == 1)
                {
                    destPc = i + imm0;
                }
                size_t begin = std::min(i, destPc);
                size_t end = std::max(i, destPc);
                for (size_t j = begin; j <= end; j++)
                {
                    canBeSplit[j] = false;
                }
            }
            else if (opcode == S_CALL_ABOSOLUTE || opcode == S_CALL_REGISTER ||
                     opcode == S_CALL_RELATIVE)
            {
                assert(false && "Not Support call");
            }
        }

        std::vector<std::vector<Instruction *>> bundles;
        size_t curPc = 0;
        while (true)
        {
            size_t nextPc = curPc + threshold;
            for (auto i = curPc; i < std::min(nextPc, instruction_lists.size());
                 i++)
            {
                if (spiltHint.count(instruction_lists[i]) != 0)
                {
                    nextPc = i + 1;
                    break;
                }
            }
            if (nextPc >= instruction_lists.size() - 1)
            {
                nextPc = instruction_lists.size() - 1;
                std::vector<Instruction *> bundle;
                bundle.assign(instruction_lists.begin() + curPc,
                              instruction_lists.end());
                bundles.emplace_back(bundle);
                break;
            }
            while (!canBeSplit[nextPc] && nextPc > curPc)
            {
                nextPc--;
            }
            assert(nextPc != curPc && "No Place To Split");
            std::vector<Instruction *> bundle;
            bundle.assign(instruction_lists.begin() + curPc,
                          instruction_lists.begin() + nextPc);
            bundles.emplace_back(bundle);
            curPc = nextPc;
        }
        return bundles;
    }
}

namespace std
{
namespace
{

// https://stackoverflow.com/questions/7110301/generic-hash-for-tuples-in-unordered-map-unordered-set

// Code from boost
// Reciprocal of the golden ratio helps spread entropy
//     and handles duplicates.
// See Mike Seymour in magic-numbers-in-boosthash-combine:
//     http://stackoverflow.com/questions/4948780

template <class T>
inline void
hash_combine(std::size_t &seed, T const &v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Recursive template code derived from Matthieu M.
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl
{
    static void
    apply(size_t &seed, Tuple const &tuple)
    {
        HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0>
{
    static void
    apply(size_t &seed, Tuple const &tuple)
    {
        hash_combine(seed, std::get<0>(tuple));
    }
};
} // namespace

template <typename... TT>
struct hash<std::tuple<TT...>>
{
    size_t
    operator()(std::tuple<TT...> const &tt) const
    {
        size_t seed = 0;
        HashValueImpl<std::tuple<TT...>>::apply(seed, tt);
        return seed;
    }
};
} // namespace std

namespace InstScheduler
{

const uint32_t InvailRegId = 74;

struct regTableEntry
{
    enum Tag
    {
        S_X,
        S_Y,
        S_DEST,
        V_X,
        V_Y,
        V_DEST,
        VS0,
        VS1,
        VS2,
        MTI_X,
        INVAILD
    };

    Tag tag = INVAILD;
    bool flag = false;
    uint32_t phyRegId = InvailRegId;
    uint32_t virRegId = InvailRegId;
};

struct ResourceScheduler
{
    enum class Tag
    {
        ScalarReg,
        VectorReg,
        PermitReg,
        VMaskReg,
        PCR,
        SPR,
        GMR,
        IAReg,
        SMem,
        VMem,
        CMem,
        VSF,
        URF,
        CRF,
        MRF,
        TRF,
        GSNF,
        GSTF,
        NWS_LOCK,
        SYNC_FLAG,
        HBM
    };

    int serialId = 0;
    using Entry = regTableEntry::Tag;

    Entry entry = Entry::INVAILD;

    Tag tag;
    uint32_t regId = 0;
    struct
    {
        uint64_t addr = 0;
        uint64_t size = 0;
    } range;

    bool
    operator==(const ResourceScheduler &rhs) const
    {
        return tag == rhs.tag && regId == rhs.regId &&
               range.addr == rhs.range.addr && range.size == rhs.range.size;
    }

    [[nodiscard]] bool
    isFifo() const
    {
        switch (tag)
        {
        case Tag::ScalarReg:
        case Tag::VectorReg:
        case Tag::PermitReg:
        case Tag::VMaskReg:
        case Tag::PCR:
        case Tag::SPR:
        case Tag::IAReg:
        case Tag::SMem:
        case Tag::VMem:
        case Tag::CMem:
        case Tag::NWS_LOCK:
        case Tag::SYNC_FLAG:
        case Tag::HBM:
        case Tag::GMR:
            return false;
        case Tag::VSF:
        case Tag::URF:
        case Tag::CRF:
        case Tag::MRF:
        case Tag::TRF:
        case Tag::GSNF:
        case Tag::GSTF:
            return true;
        }
        return false;
    }

    [[nodiscard]] bool
    isReg() const
    {
        switch (tag)
        {
        case Tag::ScalarReg:
        case Tag::VectorReg:
            return true;
        case Tag::PermitReg:
        case Tag::VMaskReg:
            // return true;
        case Tag::PCR:
        case Tag::SPR:
        case Tag::IAReg:
        case Tag::SMem:
        case Tag::VMem:
        case Tag::CMem:
        case Tag::NWS_LOCK:
        case Tag::SYNC_FLAG:
        case Tag::HBM:
        case Tag::GMR:
        case Tag::VSF:
        case Tag::URF:
        case Tag::CRF:
        case Tag::MRF:
        case Tag::TRF:
        case Tag::GSNF:
        case Tag::GSTF:
            return false;
        }
        return false;
    }

    [[nodiscard]] bool
    isGMR() const 
    {
        return tag == Tag::GMR;
    }

    static ResourceScheduler ScalarReg(int id, Entry entry)
    {
        ResourceScheduler res;
        res.tag = Tag::ScalarReg;
        res.regId = id;
        res.entry = entry;
        return res;
    }

    static ResourceScheduler VectorReg(int id, Entry entry)
    {
        ResourceScheduler res;
        res.tag = Tag::VectorReg;
        res.regId = id;
        res.entry = entry;
        return res;
    }

#define MREG(name)                                                             \
    static ResourceScheduler name(int id)                                      \
    {                                                                          \
        ResourceScheduler res;                                                 \
        res.tag = Tag::name;                                                   \
        res.regId = id;                                                        \
        return res;                                                            \
    }

#define SREG(name)                                                             \
    static ResourceScheduler name()                                            \
    {                                                                          \
        ResourceScheduler res;                                                 \
        res.tag = Tag::name;                                                   \
        return res;                                                            \
    }

#define MEM(name, bound)                                                       \
    static ResourceScheduler name(uint64_t addr, uint64_t size)                \
    {                                                                          \
        ResourceScheduler res;                                                 \
        res.tag = Tag::name;                                                   \
        res.range.addr = addr;                                                 \
        assert(addr < (bound));                                                \
        res.range.size = size;                                                 \
        return res;                                                            \
    }

    // MREG(ScalarReg)
    // MREG(VectorReg)
    MREG(PermitReg)
    MREG(VMaskReg)
    MREG(IAReg)
    MREG(MRF)
    MREG(TRF)
    MREG(GSNF)
    MREG(GSTF)
    MREG(NWS_LOCK)
    SREG(SMem)
    MEM(VMem, 4096 * 1024 / 32 + 8)
    SREG(HBM)
    SREG(CMem)
    MREG(PCR)
    MREG(SPR)
    MREG(GMR)
    SREG(VSF)
    SREG(URF)
    SREG(CRF)
    MEM(SYNC_FLAG, 0x2000)

#undef MREG
#undef SREG
#undef MEM
};

struct InstHint
{
    std::vector<std::pair<bool, uint32_t>> sregValue{32,
                                                     std::make_pair(false, 0)};
};

enum
{
    IMME0 = 1,
    IMME1 = 2,
    IMME2 = 4,
    IMME3 = 8,
    IMME4 = 16,
    IMME5 = 32,
    VSIMME0 = 64,
    VSIMME1 = 128,
    VSIMME2 = 256,
};

#define SET_RENAME(slot, tag, id) regTable[slot] = {tag, true, id, InvailRegId}
#define READ(res) readResource.push_back(res)
#define WRITE(res) writeResource.push_back(res)
#define READ_SIMPLE_S_X()                                                      \
    do                                                                         \
    {                                                                          \
        READ(ResourceScheduler::ScalarReg(x, regTableEntry::S_X));             \ 
        SET_RENAME(s_x, regTableEntry::S_X, x);                                \
        sRegCount--;                                                           \
    } while (false)                                                             
#define CHECK_Y_IMME_USE()                                                     \
    do                                                                         \
    {                                                                          \
        switch (y)                                                             \
        {                                                                      \
        case 32:                                                               \
        case 36:                                                               \
        case 40:                                                               \
            usedImme |= IMME0;                                                 \
            break;                                                             \
        case 33:                                                               \
        case 37:                                                               \
        case 41:                                                               \
            usedImme |= IMME1;                                                 \
            break;                                                             \
        case 34:                                                               \
        case 38:                                                               \
        case 42:                                                               \
            usedImme |= IMME2;                                                 \
            break;                                                             \
        case 35:                                                               \
        case 39:                                                               \
        case 43:                                                               \
            usedImme |= IMME3;                                                 \
            break;                                                             \
        case 44:                                                               \
            usedImme |= IMME0 | IMME1;                                         \
            break;                                                             \
        case 45:                                                               \
            usedImme |= IMME2 | IMME3;                                         \
            break;                                                             \
        case 64:                                                               \
        case 66:                                                               \
        case 68:                                                               \
            usedImme |= IMME4;                                                 \
            break;                                                             \
        case 65:                                                               \
        case 67:                                                               \
        case 69:                                                               \
            usedImme |= IMME5;                                                 \
            break;                                                             \
        case 70:                                                               \
            usedImme |= IMME4 | IMME5;                                         \
            break;                                                             \
        case 71:                                                               \
            usedImme |= VSIMME0;                                               \
            break;                                                             \
        case 72:                                                               \
            usedImme |= VSIMME1;                                               \
            break;                                                             \
        case 73:                                                               \
            usedImme |= VSIMME2;                                               \
            break;                                                             \
        }                                                                      \
    } while (false)
#define READ_SIMPLE_S_Y()                                                      \
    do                                                                         \
    {                                                                          \
        CHECK_Y_IMME_USE();                                                    \
        if (y < 32)                                                            \
        {                                                                      \
            READ(ResourceScheduler::ScalarReg(y, regTableEntry::S_Y));         \
            SET_RENAME(s_y, regTableEntry::S_Y, y);                            \
            sRegCount--;                                                       \
        }                                                                      \
    } while (false)
#define CHECK_SYNC_IMME_USE()                                                     \
    do                                                                         \
    {                                                                          \
        switch (syncflag)                                                       \
        {                                                                      \
        case 32:                                                               \
        case 36:                                                               \
        case 40:                                                               \
            usedImme |= IMME0;                                                 \
            break;                                                             \
        case 33:                                                               \
        case 37:                                                               \
        case 41:                                                               \
            usedImme |= IMME1;                                                 \
            break;                                                             \
        case 34:                                                               \
        case 38:                                                               \
        case 42:                                                               \
            usedImme |= IMME2;                                                 \
            break;                                                             \
        case 35:                                                               \
        case 39:                                                               \
        case 43:                                                               \
            usedImme |= IMME3;                                                 \
            break;                                                             \
        case 44:                                                               \
            usedImme |= IMME0 | IMME1;                                         \
            break;                                                             \
        case 45:                                                               \
            usedImme |= IMME2 | IMME3;                                         \
            break;                                                             \
        case 64:                                                               \
        case 66:                                                               \
        case 68:                                                               \
            usedImme |= IMME4;                                                 \
            break;                                                             \
        case 65:                                                               \
        case 67:                                                               \
        case 69:                                                               \
            usedImme |= IMME5;                                                 \
            break;                                                             \
        case 70:                                                               \
            usedImme |= IMME4 | IMME5;                                         \
            break;                                                             \
        case 71:                                                               \
            usedImme |= VSIMME0;                                               \
            break;                                                             \
        case 72:                                                               \
            usedImme |= VSIMME1;                                               \
            break;                                                             \
        case 73:                                                               \
            usedImme |= VSIMME2;                                               \
            break;                                                             \
        }                                                                      \
    } while (false)                    
#define READ_SIMPLE_V_X()                                                      \
    do                                                                         \
    {                                                                          \
        READ(ResourceScheduler::VectorReg(x, regTableEntry::V_X));             \
        SET_RENAME(v_x, regTableEntry::V_X, x);                                \
        vRegCount--;                                                           \
    } while (false)
#define VSIMME_MAP(val, vsi0, vsi1, vsi2)                                      \
    do                                                                         \
    {                                                                          \
        if ((val) == (vsi0))                                                   \
        {                                                                      \
            usedImme |= VSIMME0;                                               \
            auto sreg =                                                        \
                pInst->GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0); \
            READ(ResourceScheduler::ScalarReg(sreg, regTableEntry::VS0));      \
            SET_RENAME(vs_imm_0, regTableEntry::VS0, sreg);                    \
            sRegCount--;                                                       \
        }                                                                      \
        else if ((val) == (vsi1))                                              \
        {                                                                      \
            usedImme |= VSIMME1;                                               \
            auto sreg =                                                        \
                pInst->GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1); \
            READ(ResourceScheduler::ScalarReg(sreg, regTableEntry::VS1));      \
            SET_RENAME(vs_imm_1, regTableEntry::VS1, sreg);                    \
            sRegCount--;                                                       \
        }                                                                      \
        else if ((val) == (vsi2))                                              \
        {                                                                      \
            usedImme |= VSIMME2;                                               \
            auto sreg =                                                        \
                pInst->GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE2); \
            READ(ResourceScheduler::ScalarReg(sreg, regTableEntry::VS2));      \
            SET_RENAME(vs_imm_2, regTableEntry::VS2, sreg);                    \
            sRegCount--;                                                       \
        }                                                                      \
    } while (false)
#define READ_SIMPLE_V_Y()                                                      \
    do                                                                         \
    {                                                                          \
        CHECK_Y_IMME_USE();                                                    \
        if (y < 32)                                                            \
        {                                                                      \
            READ(ResourceScheduler::VectorReg(y, regTableEntry::V_Y));         \
            SET_RENAME(v_y, regTableEntry::V_Y, y);                            \
            vRegCount--;                                                       \
        }                                                                      \
        VSIMME_MAP(y, 71, 72, 73);                                             \
    } while (false)
#define READ_SIMPLE_S_DEST()                                                   \ 
    do                                                                         \
    {                                                                          \
        READ(ResourceScheduler::ScalarReg(dest, regTableEntry::S_DEST));       \
        SET_RENAME(s_dest, regTableEntry::S_DEST, dest);                       \
        sRegCount--;                                                           \
    } while (false)
#define WRITE_SIMPLE_S_DEST()                                                  \ 
    do                                                                         \
    {                                                                          \
        WRITE(ResourceScheduler::ScalarReg(dest, regTableEntry::S_DEST));      \
        SET_RENAME(s_dest, regTableEntry::S_DEST, dest);                       \
        sRegCount++;                                                           \
    } while (false)
#define WRITE_SIMPLE_V_DEST()                                                  \ 
    do                                                                         \
    {                                                                          \
        WRITE(ResourceScheduler::VectorReg(dest, regTableEntry::V_DEST));      \
        SET_RENAME(v_dest, regTableEntry::V_DEST, dest);                       \
        vRegCount++;                                                           \
    } while (false)
#define WRITE_SIMPLE_P_DEST() WRITE(ResourceScheduler::PermitReg(dest))
#define WRITE_SIMPLE_VMASK_DEST() WRITE(ResourceScheduler::VMaskReg(dest))
#define WRITE_SIMPLE_MRF() WRITE(ResourceScheduler::MRF(select))
#define READ_SIMPLE_MTI_X()                                                    \
    do                                                                         \
    {                                                                          \
        READ(ResourceScheduler::VectorReg(x, regTableEntry::MTI_X));           \
        SET_RENAME(mti_x, regTableEntry::MTI_X, x);                            \
        vRegCount--;                                                           \
    } while(false)
#define READ_SIMPLE_MTI_MASK() READ(ResourceScheduler::VMaskReg(mask))
#define READ_SIMPLE_V_LOADSTORE_BASE() VSIMME_MAP(base, 1, 2, 3)
#define READ_SIMPLE_V_LOADSTORE_STRIDE()                                       \
    do                                                                         \
    {                                                                          \
        VSIMME_MAP(stride, 1, 2, 3);                                           \
        if (4 <= stride && stride <= 7)                                        \
        {                                                                      \
            usedImme |= (IMME2 << (stride - 4));                               \
        }                                                                      \
    } while (false)
#define READ_SHUFFLE_V_LOAD_STRIDE()                                           \
    do                                                                         \
    {                                                                          \
        VSIMME_MAP(stride, 1, 2, 3);                                           \
        if (stride == 4)                                                       \
        {                                                                      \
            usedImme |= IMME0 | IMME1;                                         \
        }                                                                      \
        else if (stride == 5)                                                  \
        {                                                                      \
            usedImme |= IMME2 | IMME3;                                         \
        }                                                                      \
        else if (stride == 6)                                                  \
        {                                                                      \
            usedImme |= IMME4 | IMME5;                                         \
        }                                                                      \
    } while (false)
#define READ_SIMPLE_V_LOADSTORE_MASK()                                         \
    do                                                                         \
    {                                                                          \
        VSIMME_MAP(mask, 1, 2, 3);                                             \
        if (4 <= mask && mask <= 7)                                            \
        {                                                                      \
            usedImme |= (IMME2 << (mask - 4));                                 \
        }                                                                      \
    } while (false)

template <size_t L>
struct uint_t
{
    using type = typename uint_t<L + 1>::type;
};

template <>
struct uint_t<1>
{
    using type = bool;
};

template <>
struct uint_t<8>
{
    using type = uint8_t;
};

template <>
struct uint_t<16>
{
    using type = uint16_t;
};

template <>
struct uint_t<32>
{
    using type = uint32_t;
};

template <size_t E, size_t B>
typename uint_t<E - B + 1>::type
bitsel(uint32_t val)
{
    return static_cast<typename uint_t<E - B + 1>::type>(
        (val >> B) & ((1u << (E - B + 1)) - 1));
}

template <size_t E, size_t B>
typename uint_t<E>::type
bitmix(uint32_t val)
{
    return static_cast<typename uint_t<E>::type>(
        (val & ((1 << (E - B + 1)) - 1) << B));
}

struct DmaDesc
{
    bool trace_en = false;
    uint8_t dst_opcode = 0;
    uint8_t dst_core_id = 0;
    uint8_t dst_mem_id = 0;
    uint8_t src_opcode = 0;
    uint8_t src_core_id = 0;
    uint8_t src_mem_id = 0;
    uint8_t dma_type = 0;
    uint16_t dst_id = 0;
    uint8_t src_sync_core_id = 0;
    uint16_t src_sync_flag_id = 0;
    uint8_t dst1_sync_core_id = 0;
    uint16_t dst1_sync_flag_id = 0;
    uint8_t dst0_sync_core_id = 0;
    uint16_t dst0_sync_flag_id = 0;
    bool length_guanule = false;
    uint32_t length = 0;
    bool src_addr_guanule = false;
    uint32_t src_addr = 0;
    bool dst_addr_guanule = false;
    uint32_t dst_addr = 0;
    uint32_t src_stride = 0;
    uint32_t dst_stride = 0;

    static DmaDesc build_dma(uint32_t header,
                             uint32_t src_sync_flag,
                             uint32_t dst_sync_flag,
                             uint32_t length,
                             uint32_t src_addr,
                             uint32_t dst_addr,
                             uint32_t src_stride,
                             uint32_t dst_stride);
    static DmaDesc build_local_dma(uint32_t src_s0_x,
                                   uint32_t dest_s1_x,
                                   uint32_t length_s0_y,
                                   uint32_t sflag_s1_y,
                                   uint32_t misc);
    static DmaDesc build_stride_dma(uint32_t src_s0_x,
                                    uint32_t dest_s1_x,
                                    uint32_t length_s0_y,
                                    uint32_t sflag_s1_y,
                                    uint32_t misc,
                                    uint32_t src_stride_vs0,
                                    uint32_t dst_stride_vs1);

    uint16_t
    src_misc() const
    {
        return bitmix<15, 15>(dma_type) | bitmix<14, 14>(trace_en) |
               bitmix<10, 8>(src_core_id) | bitmix<5, 4>(src_mem_id) |
               bitmix<1, 0>(src_opcode);
    }

    uint16_t
    dst_misc() const
    {
        return bitmix<15, 15>(dma_type) | bitmix<14, 14>(trace_en) |
               bitmix<13, 11>(dst_core_id) | bitmix<7, 6>(dst_mem_id) |
               bitmix<3, 2>(dst_opcode);
    }
};

DmaDesc
DmaDesc::build_dma(uint32_t header,
                   uint32_t src_sync_flag,
                   uint32_t dst_sync_flag,
                   uint32_t length,
                   uint32_t src_addr,
                   uint32_t dst_addr,
                   uint32_t src_stride,
                   uint32_t dst_stride)
{
    return DmaDesc{bitsel<31, 31>(header),
                   bitsel<30, 29>(header),
                   bitsel<28, 26>(header),
                   bitsel<25, 24>(header),
                   bitsel<22, 21>(header),
                   bitsel<20, 18>(header),
                   bitsel<17, 16>(header),
                   bitsel<15, 14>(header),
                   bitsel<9, 0>(header),
                   bitsel<15, 13>(src_sync_flag),
                   bitsel<12, 0>(src_sync_flag),
                   bitsel<31, 29>(dst_sync_flag),
                   bitsel<28, 16>(dst_sync_flag),
                   bitsel<15, 13>(dst_sync_flag),
                   bitsel<12, 0>(dst_sync_flag),
                   bitsel<31, 31>(length),
                   bitsel<30, 0>(length),
                   bitsel<31, 31>(src_addr),
                   bitsel<30, 0>(src_addr),
                   bitsel<31, 31>(dst_addr),
                   bitsel<30, 0>(dst_addr),
                   src_stride,
                   dst_stride};
}

DmaDesc
DmaDesc::build_local_dma(uint32_t src_s0_x,
                         uint32_t dest_s1_x,
                         uint32_t length_s0_y,
                         uint32_t sflag_s1_y,
                         uint32_t misc)
{
    return DmaDesc{bitsel<14, 14>(misc),
                   bitsel<3, 2>(misc),
                   bitsel<13, 11>(misc),
                   bitsel<7, 6>(misc),
                   bitsel<1, 0>(misc),
                   bitsel<10, 8>(misc),
                   bitsel<5, 4>(misc),
                   bitsel<15, 15>(misc),
                   0,
                   0,
                   0,
                   bitsel<31, 29>(sflag_s1_y),
                   bitsel<28, 16>(sflag_s1_y),
                   bitsel<15, 13>(sflag_s1_y),
                   bitsel<12, 0>(sflag_s1_y),
                   bitsel<31, 31>(length_s0_y),
                   bitsel<30, 0>(length_s0_y),
                   bitsel<31, 31>(src_s0_x),
                   bitsel<30, 0>(src_s0_x),
                   bitsel<31, 31>(dest_s1_x),
                   bitsel<30, 0>(dest_s1_x),
                   0,
                   0};
}

DmaDesc
DmaDesc::build_stride_dma(uint32_t src_s0_x,
                          uint32_t dest_s1_x,
                          uint32_t length_s0_y,
                          uint32_t sflag_s1_y,
                          uint32_t misc,
                          uint32_t src_stride_vs0,
                          uint32_t dst_stride_vs1)
{
    return DmaDesc{bitsel<14, 14>(misc),
                   bitsel<3, 2>(misc),
                   bitsel<13, 11>(misc),
                   bitsel<7, 6>(misc),
                   bitsel<1, 0>(misc),
                   bitsel<10, 8>(misc),
                   bitsel<5, 4>(misc),
                   bitsel<15, 15>(misc),
                   0,
                   0,
                   0,
                   bitsel<31, 29>(sflag_s1_y),
                   bitsel<28, 16>(sflag_s1_y),
                   bitsel<15, 13>(sflag_s1_y),
                   bitsel<12, 0>(sflag_s1_y),
                   bitsel<31, 31>(length_s0_y),
                   bitsel<30, 0>(length_s0_y),
                   bitsel<31, 31>(src_s0_x),
                   bitsel<30, 0>(src_s0_x),
                   bitsel<31, 31>(dest_s1_x),
                   bitsel<30, 0>(dest_s1_x),
                   src_stride_vs0,
                   dst_stride_vs1};
}

uint32_t
GetImmeVal(Instruction &inst, uint32_t immeIdx)
{
    switch (immeIdx)
    {
    case 32:
    case 33:
    case 34:
    case 35:
    {
        return inst.GetImmediateValue(
            static_cast<Instruction::ImmediateValueType>(
                Instruction::IMMEDIATE0 + immeIdx - 32));
    }
    case 36:
    case 37:
    case 38:
    case 39:
    {
        return inst.GetImmediateValue(
            static_cast<Instruction::ImmediateValueType>(
                Instruction::IMMEDIATE0 + immeIdx - 36));
    }
    case 40:
    case 41:
    case 42:
    case 43:
    {
        auto imme =
            inst.GetImmediateValue(static_cast<Instruction::ImmediateValueType>(
                Instruction::IMMEDIATE0 + immeIdx - 40));
        return 0xffff0000 + imme;
    }
    case 44:
    case 45:
    {
        uint32_t immeL =
            inst.GetImmediateValue(static_cast<Instruction::ImmediateValueType>(
                Instruction::IMMEDIATE0 + immeIdx - 44));
        uint32_t immeH =
            inst.GetImmediateValue(static_cast<Instruction::ImmediateValueType>(
                Instruction::IMMEDIATE0 + immeIdx - 44 + 1));
        return (immeH << 16u) + immeL;
    }
    case 46:
    case 47:
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    {
        const uint32_t constImme[] = {0x00000000,
                                      0x80000000,
                                      0x00000001,
                                      0x3f800000,
                                      0x3f000000,
                                      0x40000000,
                                      0x40490fdb,
                                      0x402df854,
                                      0x0000ffff};
        return constImme[immeIdx - 46];
    }
    case 56:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    {
        const uint32_t constImme[] = {0xffffffff,
                                      0xbf800000,
                                      0xbf000000,
                                      0xc0000000,
                                      0xc0490fdb,
                                      0xc02df854,
                                      0xffff0000};
        return constImme[immeIdx - 56];
    }
    }
}

struct ScalarInst
{
    std::vector<ResourceScheduler> writeResource{};
    std::vector<ResourceScheduler> readResource{};
    Instruction *pInst = nullptr;
    int usedImme = 0;
    InstHint hint;
    uint16_t permit = 0;
    uint16_t op = 0;
    uint16_t x = 0;
    uint16_t y = 0;
    uint16_t dest = 0;

    // FOR DMA ONLY

    // reg only, x0
    uint32_t src_addr;
    // reg only, x1
    uint32_t dest_addr;
    // reg & imme, y0
    uint32_t length;
    // constval
    uint16_t misc;
    // reg & imme, y1
    uint32_t syncflag;

    int vRegCount = 0;
    int sRegCount = 0;

    enum scalarReg 
    {  
        s_x,
        s_y,
        s_dest,
        vs_imm_0,
        vs_imm_1,
        vs_imm_2
    };

    std::vector<regTableEntry> regTable{6, regTableEntry()}; 

    void setVirReg(const ResourceScheduler& res, int virtualId) 
    {
        for (auto& entry : regTable) 
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                entry.virRegId = virtualId;
            }
        }
    }

    int getVirReg(const ResourceScheduler& res) const
    {
        for (auto& entry : regTable)
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                return entry.virRegId;
            }
        }
    }

    static ScalarInst
    read(Instruction &inst, const InstHint &hints, bool slot0 = true)
    {
        auto s0 = reinterpret_cast<ScalarOperationState *>(
            inst.GetOperation(Instruction::SCALARONE));
        auto s1 = reinterpret_cast<ScalarOperationState *>(
            inst.GetOperation(Instruction::SCALARTWO));

        if (!slot0 && (s0->GetOpCode() == S_LOCAL_DMA ||
                       s0->GetOpCode() == S_STRIDED_DMA))
        {
            ScalarInst res{};
            res.pInst = &inst;
            res.op = S_NOOP;
            return res;
        }

        if (slot0 && (s0->GetOpCode() == S_LOCAL_DMA ||
                      s0->GetOpCode() == S_STRIDED_DMA))
        {
            ScalarInst res{};
            res.pInst = &inst;
            res.hint = hints;
            res.permit = s0->GetPermissionValue();
            res.op = s0->GetOpCode();
            res.src_addr = s0->GetIndexX();
            res.dest_addr = s0->GetIndexX1();
            res.length = s0->GetIndexY();
            res.syncflag = s0->GetIndexY1();
            res.misc = s0->GetIndexMisc();

            res.x = s0->GetIndexX();
            res.y = s0->GetIndexY();
            res.dest = s0->GetIndexX1();
            return res;
        }

        auto op = slot0 ? s0 : s1;

        ScalarInst res{};
        res.pInst = &inst;
        res.hint = hints;
        res.permit = op->GetPermissionValue();
        res.op = op->GetOpCode();
        res.x = op->GetIndexX();
        res.y = op->GetIndexY();
        res.dest = op->GetIndexDest();
        return res;
    }

    bool
    isNoop()
    {
        return op == S_NOOP;
    }

    void
    checkResource()
    {
        switch (op)
        {
        case S_NOOP:
        case S_HALT:
        {
            break;
        }
        case S_POP:
        {
            READ(ResourceScheduler::VSF());
            WRITE_SIMPLE_S_DEST();
            break;
        }
        case S_DELAY:
        {
            break;
        }
        case S_SMEM_LOAD:
        {
            READ_SIMPLE_S_Y();
            READ(ResourceScheduler::SMem());
            WRITE_SIMPLE_S_DEST();
            break;
        }
        case S_SMEM_LOAD_OFFSET:
        {
            READ_SIMPLE_S_X();
            READ_SIMPLE_S_Y();
            READ(ResourceScheduler::SMem());
            WRITE_SIMPLE_S_DEST();
            break;
        }
        case S_SMEM_STORE:
        {
            READ_SIMPLE_S_X();
            READ_SIMPLE_S_Y();
            WRITE(ResourceScheduler::SMem());
            break;
        }
        case S_SET:
        case S_BRANCH:
        case S_CALL_ABOSOLUTE:
        case S_CALL_RELATIVE:
        case S_CALL_REGISTER:
        case S_FENCE:
        case S_DMA:
        {
            break;
        }
        case S_LOCAL_DMA:
        {
            DmaDesc desc = DmaDesc::build_local_dma(0, 0, 0, 0, misc);
            if (desc.dst0_sync_flag_id != 0)
            {
                // TODO, actually, we should check core_id first
                WRITE(ResourceScheduler::SYNC_FLAG(desc.dst0_sync_flag_id, 1));
            }
            if (desc.dst1_sync_flag_id != 0)
            {
                WRITE(ResourceScheduler::SYNC_FLAG(desc.dst1_sync_flag_id, 1));
            }
            if (length > 31 || hint.sregValue[length].first)
            {
                uint32_t len_val = 0;
                if (length >= 32)
                {
                    len_val = GetImmeVal(*pInst, length);
                }
                else
                {
                    len_val = hint.sregValue[length].second;
                }
                if (bitsel<31, 31>(len_val))
                {
                    assert(false && "unsupport 4B");
                }
                if (hint.sregValue[src_addr].first)
                {
                    uint32_t src_addr_val = hint.sregValue[src_addr].second;
                    if (bitsel<31, 31>(src_addr_val))
                    {
                        assert(false && "unsupport 4B");
                    }
                    if (desc.src_misc() == DMA_SRC::HBM)
                    {
                        READ_SIMPLE_S_X();
                        READ_SIMPLE_S_Y();
                        READ_SIMPLE_S_DEST();
                        CHECK_SYNC_IMME_USE();
                        READ(ResourceScheduler::HBM());
                    }
                    else if (desc.src_misc() == DMA_SRC::XYS<0>::VMEM)
                    {
                        READ_SIMPLE_S_X();
                        READ_SIMPLE_S_Y();
                        READ_SIMPLE_S_DEST();
                        CHECK_SYNC_IMME_USE();
                        READ(ResourceScheduler::VMem(src_addr_val, len_val));
                    }
                    else
                    {
                        assert(false && "unimplemented");
                    }
                }
                if (hint.sregValue[dest_addr].first)
                {
                    uint32_t dst_addr_val = hint.sregValue[dest_addr].second;
                    if (bitsel<31, 31>(dst_addr_val))
                    {
                        assert(false && "unsupport 4B");
                    }
                    if (desc.dst_misc() == DMA_DEST::HBM)
                    {
                        WRITE(ResourceScheduler::HBM());
                    }
                    else if (desc.dst_misc() == DMA_DEST::XYS<0>::VMEM)
                    {
                        WRITE(ResourceScheduler::VMem(dst_addr_val, len_val));
                    }
                    else
                    {
                        assert(false && "unimplemented");
                    }
                }
            }
            else
            {
                if (desc.src_misc() == DMA_SRC::HBM)
                {
                    READ_SIMPLE_S_X();
                    READ_SIMPLE_S_Y();
                    READ_SIMPLE_S_DEST();
                    CHECK_SYNC_IMME_USE();
                    READ(ResourceScheduler::HBM());
                }
                else if (desc.src_misc() == DMA_SRC::XYS<0>::VMEM)
                {
                    READ_SIMPLE_S_X();
                    READ_SIMPLE_S_Y();
                    READ_SIMPLE_S_DEST();
                    CHECK_SYNC_IMME_USE();
                    READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
                else
                {
                    assert(false && "unimplemented");
                }
                if (desc.dst_misc() == DMA_DEST::HBM)
                {
                    WRITE(ResourceScheduler::HBM());
                }
                else if (desc.dst_misc() == DMA_DEST::XYS<0>::VMEM)
                {
                    WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
                else
                {
                    assert(false && "unimplemented");
                }
            }
            break;
        }
        case S_STRIDED_DMA:
        {
            usedImme |= VSIMME0 | VSIMME1;
            auto srcvs0 =
                pInst->GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0);
            auto dstvs1 =
                pInst->GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1);
            DmaDesc desc = DmaDesc::build_local_dma(0, 0, 0, 0, misc);
            if (length > 31 || hint.sregValue[length].first)
            {
                uint32_t len_val = 0;
                if (length >= 32)
                {
                    len_val = GetImmeVal(*pInst, length);
                }
                else
                {
                    len_val = hint.sregValue[length].second;
                }
                if (bitsel<31, 31>(len_val))
                {
                    assert(false && "unsupport 4B");
                }
                if (hint.sregValue[src_addr].first)
                {
                    uint32_t src_addr_val = hint.sregValue[src_addr].second;
                    if (bitsel<31, 31>(src_addr_val))
                    {
                        assert(false && "unsupport 4B");
                    }
                    if (desc.src_misc() == DMA_SRC::HBM)
                    {
                        READ_SIMPLE_S_X();
                        READ_SIMPLE_S_Y();
                        VSIMME_MAP(1, 1, 2, 3);
                        VSIMME_MAP(2, 1, 2, 3);
                        READ_SIMPLE_S_DEST();
                        CHECK_SYNC_IMME_USE();
                        READ(ResourceScheduler::HBM());
                    }
                    else if (desc.src_misc() == DMA_SRC::XYS<0>::VMEM)
                    {
                        READ_SIMPLE_S_X();
                        READ_SIMPLE_S_Y();
                        VSIMME_MAP(1, 1, 2, 3);
                        VSIMME_MAP(2, 1, 2, 3);
                        READ_SIMPLE_S_DEST();
                        CHECK_SYNC_IMME_USE();
                        for (int i = 0; i < len_val; i++)
                        {
                            READ(ResourceScheduler::VMem(src_addr_val +
                                                             i * srcvs0,
                                                         1));
                        }
                    }
                    else
                    {
                        assert(false && "unimplemented");
                    }
                }
                if (hint.sregValue[dest_addr].first)
                {
                    uint32_t dst_addr_val = hint.sregValue[dest_addr].second;
                    if (bitsel<31, 31>(dst_addr_val))
                    {
                        assert(false && "unsupport 4B");
                    }
                    if (desc.dst_misc() == DMA_DEST::HBM)
                    {
                        WRITE(ResourceScheduler::HBM());
                    }
                    else if (desc.dst_misc() == DMA_DEST::XYS<0>::VMEM)
                    {
                        for (int i = 0; i < len_val; i++)
                        {
                            WRITE(ResourceScheduler::VMem(dst_addr_val +
                                                              i * dstvs1,
                                                          len_val));
                        }
                    }
                    else
                    {
                        assert(false && "unimplemented");
                    }
                }
            }
            else
            {
                if (desc.src_misc() == DMA_SRC::HBM)
                {
                    READ_SIMPLE_S_X();
                    READ_SIMPLE_S_Y();
                    READ_SIMPLE_S_DEST();
                    CHECK_SYNC_IMME_USE();
                    READ(ResourceScheduler::HBM());
                }
                else if (desc.src_misc() == DMA_SRC::XYS<0>::VMEM)
                {
                    READ_SIMPLE_S_X();
                    READ_SIMPLE_S_Y();
                    READ_SIMPLE_S_DEST();
                    CHECK_SYNC_IMME_USE();
                    READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
                else
                {
                    assert(false && "unimplemented");
                }
                if (desc.dst_misc() == DMA_DEST::HBM)
                {
                    WRITE(ResourceScheduler::HBM());
                }
                else if (desc.dst_misc() == DMA_DEST::XYS<0>::VMEM)
                {
                    WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
                else
                {
                    assert(false && "unimplemented");
                }
            }
            break;
        }
        case S_READ:
        {
            READ_SIMPLE_S_DEST();
            break;
        }
        case S_CONVERT_S32_TO_F32:
        {
            READ_SIMPLE_S_Y();
            WRITE_SIMPLE_S_DEST();
            break;
        }
        case S_CONVERT_F32_TO_S32:
        case S_S32_ADDITION:
        case S_S32_SUBTRACTION:
        case S_U32_AND:
        case S_U32_OR:
        case S_U32_XOR:
        case S_U32_SHIFTLEFT:
        case S_U32_SHIFTRIGHT:
        {
            READ_SIMPLE_S_X();
            READ_SIMPLE_S_Y();
            WRITE_SIMPLE_S_DEST();
            break;
        }
        case S_U32_MOVE:
        case S_U32_COUNTLEADINGZEROES:
        {
            READ_SIMPLE_S_Y();
            WRITE_SIMPLE_S_DEST();
            break;
        }
        case S_U32_MULTIPLICATION:
        case S_F32_ADDITION:
        case S_F32_SUBTRACTION:
        case S_F32_MULTIPLICATION:
        case S_F32_MAX:
        case S_F32_MIN:
        {
            READ_SIMPLE_S_X();
            READ_SIMPLE_S_Y();
            WRITE_SIMPLE_S_DEST();
            break;
        }
        case S_S32_EQUAL:
        case S_S32_NOTEQUAL:
        case S_S32_GREATER:
        case S_S32_GREATEREQUAL:
        case S_S32_LESSER:
        case S_S32_LESSER_EQUAL:
        case S_U32_CARRY:
        case S_F32_EQUAL:
        case S_F32_NOTEQUAL:
        case S_F32_GREATER:
        case S_F32_GREATEREQUAL:
        case S_F32_LESSER:
        case S_F32_LESSEREQUAL:
        {
            READ_SIMPLE_S_X();
            READ_SIMPLE_S_Y();
            WRITE_SIMPLE_P_DEST();
            break;
        }
        case S_F32_IS_INF_OR_NAN:
        {
            READ_SIMPLE_S_X();
            WRITE_SIMPLE_P_DEST();
            break;
        }
        case S_PERMISSION_OR:
        case S_ARITHMETIC_SHIFT_RIGHT:
        {
            READ_SIMPLE_S_X();
            READ_SIMPLE_S_Y();
            WRITE_SIMPLE_P_DEST();
            break;
        }
        }
    }

    std::vector<int>
    forceSplitPlace()
    {
        switch (op)
        {
        case S_HALT:
        case S_DELAY:
        case S_FENCE:
        case S_DMA:
        {
            return {1, 0};
        }

        case S_BRANCH:
        {
            assert((dest == 0 || dest == 1) && "Only Support brabs And brrel");
            auto imm0 =
                (int16_t)pInst->GetImmediateValue(Instruction::IMMEDIATE0);
            return {1, 0, -imm0};
        }
        case S_CALL_RELATIVE:
        {
            auto imm0 =
                (int16_t)pInst->GetImmediateValue(Instruction::IMMEDIATE0);
            return {1, 0, -imm0};
        }
        }
        return {};
    }
};

struct VectorInst
{
    std::vector<ResourceScheduler> writeResource{};
    std::vector<ResourceScheduler> readResource{};
    Instruction *pInst = nullptr;
    int usedImme = 0;
    uint16_t permit = 0;
    uint16_t op = 0;
    uint16_t x = 0;
    uint16_t y = 0;
    uint16_t dest = 0;

    int vRegCount = 0;
    int sRegCount = 0;

    enum vectorReg 
    {  
        v_x,
        v_y,
        v_dest,
        vs_imm_0,
        vs_imm_1,
        vs_imm_2
    };

    std::vector<regTableEntry> regTable{6, regTableEntry()}; 

    void setVirReg(const ResourceScheduler& res, int virtualId) 
    {
        for(auto& entry : regTable) 
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                entry.virRegId = virtualId;
            }
        }
    }

    int getVirReg(const ResourceScheduler& res) const
    {
        for (auto& entry : regTable)
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                return entry.virRegId;
            }
        }
    }

    static VectorInst
    read(Instruction &inst, const InstHint &hints, bool slot0 = true)
    {
        auto op = reinterpret_cast<VectorOperationState *>(inst.GetOperation(
            slot0 ? Instruction::VECTORONE : Instruction::VECTORTWO));
        VectorInst res{};
        res.pInst = &inst;
        res.permit = op->GetPermissionValue();
        res.op = op->GetOpCode();
        res.x = op->GetIndexX();
        res.y = op->GetIndexY();
        res.dest = op->GetIndexDest();
        return res;
    }

    bool
    isNoop()
    {
        return op == V_NOOP;
    }

    void
    checkResource()
    {
        switch (op)
        {
        case V_NOOP:
        {
            break;
        }
        case V_HALF_FLOAT_PACK:
        case V_TWO_LOWER_INT8_PACK:
        case V_FOUR_INT8_PACK:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SUBCORE_ROTATE:
        {
            READ_SIMPLE_V_X();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_RNG_GENERATE_RANDOM_NUMBER:
        case V_RNG_READ_SEED:
        {
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_RNG_RESEED:
        {
            READ_SIMPLE_V_X();
            break;
        }
        case V_F32_SOFTSIGN:
        case V_F32_LOG2:
        {
            READ_SIMPLE_V_X();
            WRITE(ResourceScheduler::URF());
            break;
        }
        case V_F32_SIGMOID:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE(ResourceScheduler::URF());
            break;
        }
        case V_F32_RECIPROCAL:
        case V_F32_SQUAREROOT_RECIPROCAL:
        case V_F32_POWER:
        case V_F32_SOFTPLUS:
        case V_F32_EXPONENT:
        {
            READ_SIMPLE_V_X();
            WRITE(ResourceScheduler::URF());
            break;
        }
        case V_CONVERT_S32_TO_F32:
        {
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_CONVERT_F32_TO_S32:
        case V_S32_ADDITION:
        case V_S32_SUBTRACTION:
        case V_U32_AND:
        case V_U32_OR:
        case V_U32_XOR:
        case V_U32_SHIFTLEFT:
        case V_U32_SHIFTRIGHT:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_U32_MOVE:
        case V_U32_COUNTLEADINGZEROES:
        {
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_U32_MULTIPLICATION:
        {
            break;
        }
        case V_F32_ADDITION:
        case V_F32_SUBTRACTION:
        case V_F32_MULTIPLICATION:
        case V_F32_MAX:
        case V_F32_MIN:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_S32_EQUAL:
        case V_S32_NOTEQUAL:
        case V_S32_GREATER:
        case V_S32_GREATEREQUAL:
        case V_S32_LESSER:
        case V_S32_LESSER_EQUAL:
        case V_U32_CARRY:
        case V_F32_EQUAL:
        case V_F32_NOTEQUAL:
        case V_F32_GREATER:
        case V_F32_GREATEREQUAL:
        case V_F32_LESSER:
        case V_F32_LESSEREQUAL:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_VMASK_DEST();
            break;
        }
        case V_F32_IS_INF_OR_NAN:
        {
            READ_SIMPLE_V_X();
            WRITE_SIMPLE_VMASK_DEST();
            break;
        }
        case V_PERMISSION_OR:
        {
            READ(ResourceScheduler::VMaskReg(x));
            READ(ResourceScheduler::VMaskReg(y));
            WRITE_SIMPLE_VMASK_DEST();
            break;
        }
        case V_ARITHMETIC_SHIFT_RIGHT:
        case V_ROUND_ARITHMETIC_SHIFT_RIGHT:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_VMASK_DEST();
            break;
        }
        case V_SELECT_VMASK0:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(0));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SELECT_VMASK1:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(1));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SELECT_VMASK2:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(2));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SELECT_VMASK3:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(3));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SELECT_VMASK4:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(4));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SELECT_VMASK5:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(5));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SELECT_VMASK6:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(6));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SELECT_VMASK7:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            READ(ResourceScheduler::VMaskReg(7));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_GET_V_CORE_ID:
        {
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_SET_VMASK:
        {
            READ_SIMPLE_V_Y();
            WRITE(ResourceScheduler::VMaskReg(dest));
            break;
        }
        case V_EXTRACT:
        {
            READ_SIMPLE_V_X();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_COMPOSE_FLOAT:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_COUNT_NUMBER_OF_ONE:
        {
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_RELUX:
        case V_CLAMP:
        {
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_Y();
            WRITE_SIMPLE_V_DEST();
            break;
        }
        }
    }

    std::vector<int>
    forceSplitPlace()
    {
        return {};
    }
};

std::pair<bool, uint32_t>
GetStrideLike(uint32_t stride, uint32_t defa, InstHint &hint, Instruction &inst)
{
    if (stride == 0)
    {
        return std::make_pair(true, defa);
    }
    if (1 <= stride && stride <= 3)
    {
        auto sreg =
            inst.GetImmediateValue(static_cast<Instruction::ImmediateValueType>(
                Instruction::VECTORSCALARIMMEDIATE0 + stride - 1));
        return hint.sregValue[sreg];
    }
    else
    {
        auto imme =
            inst.GetImmediateValue(static_cast<Instruction::ImmediateValueType>(
                Instruction::IMMEDIATE2 + stride - 4));
        return std::make_pair(true, imme);
    }
}

struct VectorLoadInst
{
    std::vector<ResourceScheduler> writeResource{};
    std::vector<ResourceScheduler> readResource{};
    Instruction *pInst = nullptr;
    int usedImme = 0;
    uint16_t permit = 0;
    uint16_t op = 0;
    uint16_t dest = 0;
    uint16_t base = 0;
    uint16_t offset = 0;
    uint16_t stride = 0;
    uint16_t ia = 0;
    uint16_t mask = 0;
    InstHint hint;

    int vRegCount = 0;
    int sRegCount = 0;

    enum vectorLoadReg 
    {  
        v_dest,
        vs_imm_0,
        vs_imm_1,
        vs_imm_2
    };

    std::vector<regTableEntry> regTable{4, regTableEntry()}; 

    void setVirReg(const ResourceScheduler& res, int virtualId) 
    {
        for(auto& entry : regTable) 
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                entry.virRegId = virtualId;
            }
        }
    }

    int getVirReg(const ResourceScheduler& res) const
    {
        for (auto& entry : regTable)
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                return entry.virRegId;
            }
        }
    }

    static VectorLoadInst
    read(Instruction &inst, const InstHint &hint)
    {
        auto op = reinterpret_cast<VectorLoadOperationState *>(
            inst.GetOperation(Instruction::VECTORLOAD));
        VectorLoadInst res{};
        res.pInst = &inst;
        res.permit = op->GetPermissionValue();
        res.op = op->GetOpCode();
        res.dest = op->GetIndexDest();
        res.base = op->GetBase();
        res.offset = op->GetOffset();
        res.stride = op->GetStride();
        res.ia = op->GetIA();
        res.mask = op->GetMask();
        res.hint = hint;
        return res;
    }

    bool
    isNoop()
    {
        return op == V_LOAD_NOOP;
    }

    void
    checkResource()
    {
        switch (op)
        {
        case V_LOAD_NOOP:
        {
            break;
        }
        case V_LOAD:
        {
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            if (base == 1 || base == 2 || base == 3)
            {
                auto sreg = pInst->GetImmediateValue(
                    static_cast<Instruction::ImmediateValueType>(
                        Instruction::VECTORSCALARIMMEDIATE0 + base - 1));
                if (hint.sregValue[sreg].first)
                {
                    auto sv = GetStrideLike(stride, 1, hint, *pInst);
                    if (sv.first)
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            READ(ResourceScheduler::VMem(
                                (hint.sregValue[sreg].second / 4) +
                                    i * sv.second,
                                128 / 128));
                        }
                    }
                    else
                    {
                        READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                    }
                }
                else if (base == 0)
                {
                    READ(ResourceScheduler::VMem(0, 1024 / 32));
                }
                WRITE_SIMPLE_V_DEST();
                break;
            }
        }
        case V_LOAD_WITH_OFFSET:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            auto offsetV = pInst->GetImmediateValue(
                static_cast<Instruction::ImmediateValueType>(
                    Instruction::IMMEDIATE2 + offset));
            if (base == 1 || base == 2 || base == 3)
            {
                auto sreg = pInst->GetImmediateValue(
                    static_cast<Instruction::ImmediateValueType>(
                        Instruction::VECTORSCALARIMMEDIATE0 + base - 1));
                if (hint.sregValue[sreg].first)
                {
                    auto sv = GetStrideLike(stride, 1, hint, *pInst);
                    auto addr = hint.sregValue[sreg].second + offsetV;
                    if (sv.first)
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            READ(ResourceScheduler::VMem(addr / 4 +
                                                             i * sv.second,
                                                         128 / 128));
                        }
                    }
                    else
                    {
                        READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                    }
                }
                else
                {
                    READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
            }
            else if (base == 0)
            {
                auto sv = GetStrideLike(stride, 1, hint, *pInst);
                if (sv.first)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        READ(
                            ResourceScheduler::VMem(offsetV / 4 + i * sv.second,
                                                    128 / 128));
                    }
                }
                else
                {
                    READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
            }
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_LOAD_WITH_VMASK0:
        case V_LOAD_WITH_VMASK1:
        case V_LOAD_WITH_VMASK2:
        case V_LOAD_WITH_VMASK3:
        case V_LOAD_WITH_VMASK4:
        case V_LOAD_WITH_VMASK5:
        case V_LOAD_WITH_VMASK6:
        case V_LOAD_WITH_VMASK7:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ(ResourceScheduler::VMaskReg(op - V_LOAD_WITH_VMASK0));
            auto offsetV = pInst->GetImmediateValue(
                static_cast<Instruction::ImmediateValueType>(
                    Instruction::IMMEDIATE2 + offset));
            if (base == 1 || base == 2 || base == 3)
            {
                auto sreg = pInst->GetImmediateValue(
                    static_cast<Instruction::ImmediateValueType>(
                        Instruction::VECTORSCALARIMMEDIATE0 + base - 1));
                if (hint.sregValue[sreg].first)
                {
                    auto sv = GetStrideLike(stride, 1, hint, *pInst);
                    auto addr = hint.sregValue[sreg].second + offsetV;
                    if (sv.first)
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            READ(ResourceScheduler::VMem(addr / 4 +
                                                             i * sv.second,
                                                         128 / 128));
                        }
                    }
                    else
                    {
                        READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                    }
                }
                else
                {
                    READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
            }
            else if (base == 0)
            {
                auto sv = GetStrideLike(stride, 1, hint, *pInst);
                if (sv.first)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        READ(
                            ResourceScheduler::VMem(offsetV / 4 + i * sv.second,
                                                    128 / 128));
                    }
                }
                else
                {
                    READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
            }
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_LOAD_INDEXED:
        {
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ(ResourceScheduler::IAReg(ia));
            READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_LOAD_INDEXED_WITH_OFFSET:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ(ResourceScheduler::IAReg(ia));
            READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_LOAD_INDEXED_WITH_VMASK0:
        case V_LOAD_INDEXED_WITH_VMASK1:
        case V_LOAD_INDEXED_WITH_VMASK2:
        case V_LOAD_INDEXED_WITH_VMASK3:
        case V_LOAD_INDEXED_WITH_VMASK4:
        case V_LOAD_INDEXED_WITH_VMASK5:
        case V_LOAD_INDEXED_WITH_VMASK6:
        case V_LOAD_INDEXED_WITH_VMASK7:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ(ResourceScheduler::IAReg(ia));
            READ(ResourceScheduler::VMaskReg(op - V_LOAD_INDEXED_WITH_VMASK0));
            READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_LOAD_WITH_SHUFFLE:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SHUFFLE_V_LOAD_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_LOAD_INDEXED_WITH_SHUFFLE:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SHUFFLE_V_LOAD_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ(ResourceScheduler::IAReg(ia));
            READ(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case V_LOAD_FXC:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ(ResourceScheduler::VMaskReg(mask));
            READ(ResourceScheduler::CMem());
            WRITE(ResourceScheduler::CRF());
            break;
        }
        }
    }

    std::vector<int>
    forceSplitPlace()
    {
        return {};
    }
};

struct VectorStoreInst
{
    std::vector<ResourceScheduler> writeResource{};
    std::vector<ResourceScheduler> readResource{};
    Instruction *pInst = nullptr;
    int usedImme = 0;
    uint16_t permit = 0;
    uint16_t op = 0;
    uint16_t x = 0;
    uint16_t base = 0;
    uint16_t offset = 0;
    uint16_t stride = 0;
    uint16_t ia = 0;
    uint16_t mask = 0;
    InstHint hint;

    int vRegCount = 0;
    int sRegCount = 0;

    enum vectorStoreReg 
    {  
        v_x,
        vs_imm_0,
        vs_imm_1,
        vs_imm_2
    };

    std::vector<regTableEntry> regTable{4, regTableEntry()}; 

    void setVirReg(const ResourceScheduler& res, int virtualId) 
    {
        for(auto& entry : regTable) 
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                entry.virRegId = virtualId;
            }
        }
    }

    int getVirReg(const ResourceScheduler& res) const
    {
        for (auto& entry : regTable)
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                return entry.virRegId;
            }
        }
    }

    static VectorStoreInst
    read(Instruction &inst, const InstHint &hints)
    {
        auto op = reinterpret_cast<VectorStoreOperationState *>(
            inst.GetOperation(Instruction::VECTORSTORE));
        VectorStoreInst res{};
        res.pInst = &inst;
        res.permit = op->GetPermissionValue();
        res.op = op->GetOpCode();
        res.x = op->GetIndexX();
        res.base = op->GetBase();
        res.offset = op->GetOffset();
        res.stride = op->GetStride();
        res.ia = op->GetIA();
        res.mask = op->GetMask();
        res.hint = hints;
        return res;
    }

    bool
    isNoop()
    {
        return op == V_STORE_NOOP;
    }

    void
    checkResource()
    {
        switch (op)
        {
        case V_STORE_NOOP:
        {
            break;
        }
        case V_STORE:
        {
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ_SIMPLE_V_X();
            if (base == 1 || base == 2 || base == 3)
            {
                auto sreg = pInst->GetImmediateValue(
                    static_cast<Instruction::ImmediateValueType>(
                        Instruction::VECTORSCALARIMMEDIATE0 + base - 1));
                if (hint.sregValue[sreg].first)
                {
                    auto addr = hint.sregValue[sreg].second;
                    WRITE(ResourceScheduler::VMem(addr / 4, 1024 / 32));
                }
                else
                {
                    WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
            }
            else if (base == 0)
            {
                WRITE(ResourceScheduler::VMem(0, 1024 / 32));
            }
            break;
        }
        case V_STORE_WITH_OFFSET:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ_SIMPLE_V_X();
            auto offsetV = pInst->GetImmediateValue(
                static_cast<Instruction::ImmediateValueType>(
                    Instruction::IMMEDIATE2 + offset));
            if (base == 1 || base == 2 || base == 3)
            {
                auto sreg = pInst->GetImmediateValue(
                    static_cast<Instruction::ImmediateValueType>(
                        Instruction::VECTORSCALARIMMEDIATE0 + base - 1));
                if (hint.sregValue[sreg].first)
                {

                    auto addr = hint.sregValue[sreg].second + offsetV;
                    WRITE(ResourceScheduler::VMem(addr / 4, 1024 / 32));
                }
                else
                {
                    WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
            }
            else if (base == 0)
            {
                WRITE(ResourceScheduler::VMem(offsetV / 4, 1024 / 32));
            }
            break;
        }
        case V_STORE_WITH_VMASK0:
        case V_STORE_WITH_VMASK1:
        case V_STORE_WITH_VMASK2:
        case V_STORE_WITH_VMASK3:
        case V_STORE_WITH_VMASK4:
        case V_STORE_WITH_VMASK5:
        case V_STORE_WITH_VMASK6:
        case V_STORE_WITH_VMASK7:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ_SIMPLE_V_X();
            READ(ResourceScheduler::VMaskReg(op - V_STORE_WITH_VMASK0));
            auto offsetV = pInst->GetImmediateValue(
                static_cast<Instruction::ImmediateValueType>(
                    Instruction::IMMEDIATE2 + offset));
            if (base == 1 || base == 2 || base == 3)
            {
                auto sreg = pInst->GetImmediateValue(
                    static_cast<Instruction::ImmediateValueType>(
                        Instruction::VECTORSCALARIMMEDIATE0 + base - 1));
                if (hint.sregValue[sreg].first)
                {

                    auto addr = hint.sregValue[sreg].second + offsetV;
                    WRITE(ResourceScheduler::VMem(addr / 4, 1024 / 32));
                }
                else
                {
                    WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
                }
            }
            else if (base == 0)
            {
                WRITE(ResourceScheduler::VMem(offsetV / 4, 1024 / 32));
            }
            break;
        }
        case V_STORE_INDEXED:
        {
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ_SIMPLE_V_X();
            READ(ResourceScheduler::IAReg(ia));
            WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            break;
        }
        case V_STORE_INDEXED_WITH_OFFSET:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ_SIMPLE_V_X();
            READ(ResourceScheduler::IAReg(ia));
            READ(ResourceScheduler::VMaskReg(op - V_STORE_INDEXED_WITH_OFFSET));
            WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            break;
        }
        case V_STORE_INDEXED_WITH_VMASK0:
        case V_STORE_INDEXED_WITH_VMASK1:
        case V_STORE_INDEXED_WITH_VMASK2:
        case V_STORE_INDEXED_WITH_VMASK3:
        case V_STORE_INDEXED_WITH_VMASK4:
        case V_STORE_INDEXED_WITH_VMASK5:
        case V_STORE_INDEXED_WITH_VMASK6:
        case V_STORE_INDEXED_WITH_VMASK7:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ_SIMPLE_V_LOADSTORE_MASK();
            READ_SIMPLE_V_X();
            READ(ResourceScheduler::IAReg(ia));
            WRITE(ResourceScheduler::VMem(0, 4096 * 1024 / 32));
            break;
        }
        case V_STORE_SET_IA_OF_CORE:
        case V_STORE_SET_IA_OF_SUBCORE:
        {
            READ_SIMPLE_V_X();
            WRITE(ResourceScheduler::IAReg(ia));
            break;
        }
        case V_STORE_PUSH_TO_SCALAR_CORE:
        {
            READ_SIMPLE_V_X();
            WRITE(ResourceScheduler::VSF());
            break;
        }
        case V_STORE_FXC:
        {
            usedImme |= IMME2 << offset;
            READ_SIMPLE_V_X();
            READ_SIMPLE_V_LOADSTORE_BASE();
            READ_SIMPLE_V_LOADSTORE_STRIDE();
            READ(ResourceScheduler::VMaskReg(mask));
            WRITE(ResourceScheduler::CMem());
            break;
        }
        }
    }

    std::vector<int>
    forceSplitPlace()
    {
        return {};
    }
};

struct MTIInst
{
    std::vector<ResourceScheduler> writeResource{};
    std::vector<ResourceScheduler> readResource{};
    Instruction *pInst = nullptr;
    int usedImme = 0;
    uint16_t permit = 0;
    uint16_t op = 0;
    uint16_t x = 0;
    uint16_t mask = 0;
    uint16_t select = 0;

    int vRegCount = 0;
    int sRegCount = 0;

    enum mtiReg 
    {  
        mti_x,
        vs_imm_0,
        vs_imm_1,
        vs_imm_2
    };

    std::vector<regTableEntry> regTable{4, regTableEntry()}; 

    void setVirReg(const ResourceScheduler& res, int virtualId) 
    {
        for(auto& entry : regTable) 
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                entry.virRegId = virtualId;
            }
        }
    }

    int getVirReg(const ResourceScheduler& res) const
    {
        for (auto& entry : regTable)
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                return entry.virRegId;
            }
        }
    }

    static MTIInst
    read(Instruction &inst, const InstHint &hints)
    {
        auto op = reinterpret_cast<MTIOperationState *>(
            inst.GetOperation(Instruction::MTI));
        MTIInst res{};
        res.pInst = &inst;
        res.permit = op->GetPermissionValue();
        res.op = op->GetOpCode();
        res.x = op->GetIndexX();
        res.mask = op->GetMask();
        res.select = op->GetSelect();
        return res;
    }

    bool
    isNoop()
    {
        return op == MTI_NOOP;
    }

    void
    checkResource()
    {
        switch (op)
        {
        case MTI_NOOP:
        {
            break;
        }
        case MTI_MUL_FLOAT_ROUNDED:
        case MTI_MUL_HIGHER_F16:
        case MTI_MUL_LOWER_F16:
        case MTI_MUL_F16_PACKED:
        case MTI_MUL_INT8_PACKED:
        case MTI_MUL_INT8_LOWER16_PACKED:
        {
            READ_SIMPLE_MTI_X();
            READ(ResourceScheduler::GMR(select));
            WRITE_SIMPLE_MRF();
            break;
        }
        case MTI_MUL_GSNF_ROUNDED:
        case MTI_MUL_GSNF_HIGHER16:
        case MTI_MUL_GSNF_LOWER16:
        case MTI_MUL_GSNF_PACKED_F16:
        case MTI_MUL_GSNF_PACKED_INT8:
        case MTI_MUL_GSNF_PACKED_INT8_LOWER16:
        {
            READ_SIMPLE_MTI_X();
            // READ(ResourceScheduler::GMR(select));
            READ(ResourceScheduler::GSNF(select));
            WRITE(ResourceScheduler::GMR(select));
            WRITE_SIMPLE_MRF();
            break;
        }
        case MTI_MUL_GSTF_ROUNDED:
        case MTI_MUL_GSTF_HIGHER16:
        case MTI_MUL_GSTF_LOWER16:
        case MTI_MUL_GSTF_PACKED_F16:
        case MTI_MUL_GSTF_PACKED_INT8:
        case MTI_MUL_GSTF_PACKED_INT8_LOWER16:
        {
            READ_SIMPLE_MTI_X();
            // READ(ResourceScheduler::GMR(select));
            READ(ResourceScheduler::GSTF(select));
            WRITE(ResourceScheduler::GMR(select));
            WRITE_SIMPLE_MRF();
            break;
        }
        case MTI_MUL_MASK_ROUNDED:
        case MTI_MUL_MASK_HIGER16:
        case MTI_MUL_MASK_LOWER16:
        case MTI_MUL_MASK_PACKED_F16:
        case MTI_MUL_MASK_PACKED_INT8:
        case MTI_MUL_MASK_PACKED_INT8_LOWER16:
        {
            READ_SIMPLE_MTI_X();
            READ(ResourceScheduler::VMaskReg(mask));
            WRITE_SIMPLE_MRF();
            break;
        }
        case MTI_MUL_MASK_GSNF_ROUNDED:
        case MTI_MUL_MASK_GSNF_HIGER16:
        case MTI_MUL_MASK_GSNF_LOWER16:
        case MTI_MUL_MASK_GSNF_PACKED_F16:
        case MTI_MUL_MASK_GSNF_PACKED_INT8:
        case MTI_MUL_MASK_GSNF_PACKED_INT8_LOWER16:
        {
            READ_SIMPLE_MTI_X();
            READ(ResourceScheduler::GSNF(select));
            READ(ResourceScheduler::VMaskReg(mask));
            WRITE_SIMPLE_MRF();
            break;
        }
        case MTI_MUL_MASK_GSTF_ROUNDED:
        case MTI_MUL_MASK_GSTF_HIGER16:
        case MTI_MUL_MASK_GSTF_LOWER16:
        case MTI_MUL_MASK_GSTF_PACKED_F16:
        case MTI_MUL_MASK_GSTF_PACKED_INT8:
        case MTI_MUL_MASK_GSTF_PACKED_INT8_LOWER16:
        {
            READ_SIMPLE_MTI_X();
            READ(ResourceScheduler::GSTF(select));
            READ(ResourceScheduler::VMaskReg(mask));
            WRITE_SIMPLE_MRF();
            break;
        }
        case MTI_LOAD_GSNF:
        {
            READ(ResourceScheduler::GSNF(select));
            WRITE(ResourceScheduler::GMR(select));
            break;
        }
        case MTI_LOAD_GSTF:
        {
            READ(ResourceScheduler::GSTF(select));
            WRITE(ResourceScheduler::GMR(select));
            break;
        }
        case MTI_PUSHGAIN_FLOAT_ROUNDED:
        case MTI_PUSHGAIN_HIGHER_F16:
        case MTI_PUSHGAIN_LOWER_F16:
        // case MTI_PUSHGAIN_PACKED_INT8:
        {
            READ_SIMPLE_MTI_X();
            WRITE(ResourceScheduler::GSNF(select));
            break;
        }
        case MTI_PUSHGAIN_TRANSPOSE_ROUND:
        case MTI_PUSHGAIN_TRANSPOSE_HIGHER_F16:
        case MTI_PUSHGAIN_TRANSPOSE_LOWER_F16:
        // case MTI_PUSHGAIN_TRANSPOSE_PACKED_INT8:
        {
            READ_SIMPLE_MTI_X();
            WRITE(ResourceScheduler::GSTF(select));
            break;
        }
        case MTI_PUSHGAIN_MASK_ROUNDED:
        case MTI_PUSHGAIN_MASK_HIGHER16:
        case MTI_PUSHGAIN_MASK_LOWER16:
        // case MTI_PUSHGAIN_MASK_PACKED_INT8:
        {
            READ(ResourceScheduler::VMaskReg(mask));
            WRITE(ResourceScheduler::GSNF(select));
            break;
        }
        case MTI_PUSHGAIN_MASK_TRANSPOSE_ROUND:
        case MTI_PUSHGAIN_MASK_TRANSPOSE_HIGHER16:
        case MTI_PUSHGAIN_MASK_TRANSPOSE_LOWER16:
        case MTI_PUSHGAIN_MASK_PACKED_TRANSPOSE_INT16:
        {
            READ(ResourceScheduler::VMaskReg(mask));
            WRITE(ResourceScheduler::GSTF(select));
            break;
        }
        case MTI_TRANSPOSE_START:
        case MTI_TRANSPOSE_SEGMENT_START:
        case MTI_TRANSPOSE_PACKED_START:
        case MTI_TRANSPOSE_PACKED_SEGMENT_START:
        {
            READ_SIMPLE_MTI_X();
            VSIMME_MAP(mask, 1, 2, 3);
            if (4 <= mask && mask <= 7)
            {
                usedImme |= (IMME2 << (mask - 4));
            }
            WRITE(ResourceScheduler::NWS_LOCK(select));
            WRITE(ResourceScheduler::TRF(select));
            break;
        }
        case MTI_TRANSPOSE:
        case MTI_TRANSPOSE_END:
        case MTI_TRANSPOSE_SEGMENT:
        case MTI_TRANSPOSE_SEGMENT_END:
        case MTI_TRANSPOSE_PACKED:
        case MTI_TRANSPOSE_PACKED_END:
        case MTI_TRANSPOSE_PACKED_SEGMENT:
        case MTI_TRANSPOSE_PACKED_SEGMENT_END:
        {
            READ_SIMPLE_MTI_X();
            VSIMME_MAP(mask, 1, 2, 3);
            if (4 <= mask && mask <= 7)
            {
                usedImme |= (IMME2 << (mask - 4));
            }
            READ(ResourceScheduler::NWS_LOCK(select));
            WRITE(ResourceScheduler::TRF(select));
            break;
        }
        case MTI_TRANSPOSE_START_END:
        case MTI_TRANSPOSE_SEGMENT_START_END:
        case MTI_TRANSPOSE_PACKED_START_END:
        case MTI_TRANSPOSE_PACKED_SEGMENT_START_END:
        {
            READ_SIMPLE_MTI_X();
            VSIMME_MAP(mask, 1, 2, 3);
            if (4 <= mask && mask <= 7)
            {
                usedImme |= (IMME2 << (mask - 4));
            }
            WRITE(ResourceScheduler::TRF(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        case MTI_PERMUTE:
        case MTI_PERMUTE_PACKED:
        {
            READ_SIMPLE_MTI_X();
            READ(ResourceScheduler::PCR(select));
            WRITE(ResourceScheduler::TRF(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        case MTI_SET_PERMUTE:
        case MTI_SET_PERMUTE_SUBLANES:
        case MTI_SET_PERMUTE_BYTE:
        {
            READ_SIMPLE_MTI_X();
            WRITE(ResourceScheduler::PCR(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        case MTI_SET_SPR:
        {
            READ_SIMPLE_MTI_X();
            WRITE(ResourceScheduler::SPR(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        case MTI_REDUCTION_V_SUM:
        case MTI_REDUCTION_V_MAX:
        case MTI_REDUCTION_V_MIN:
        case MTI_REDUCTION_V_MAX_INDEX:
        case MTI_REDUCTION_V_MIN_INDEX:
        {
            READ_SIMPLE_MTI_X();
            WRITE(ResourceScheduler::TRF(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        case MTI_REDUCTION_SEGMENTED_V_SUM:
        case MTI_REDUCTION_SEGMENTED_V_MAX:
        case MTI_REDUCTION_SEGMENTED_V_MIN:
        case MTI_REDUCTION_SEGMENTED_V_MAX_INDEX:
        case MTI_REDUCTION_SEGMENTED_V_MIN_INDEX:
        {
            READ_SIMPLE_MTI_X();
            READ(ResourceScheduler::SPR(select));
            WRITE(ResourceScheduler::TRF(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        case MTI_REDUCTION_PACKED_V_SUM:
        case MTI_REDUCTION_PACKED_V_MAX:
        case MTI_REDUCTION_PACKED_V_MIN:
        case MTI_REDUCTION_PACKED_V_MAX_INDEX:
        case MTI_REDUCTION_PACKED_V_MIN_INDEX:
        case MTI_REDUCTION_PACKED_SEGMENTED_V_SUM:
        case MTI_REDUCTION_PACKED_SEGMENTED_V_MAX:
        case MTI_REDUCTION_PACKED_SEGMENTED_V_MIN:
        case MTI_REDUCTION_PACKED_SEGMENTED_V_MAX_INDEX:
        case MTI_REDUCTION_PACKED_SEGMENTED_V_MIN_INDEX:
        {
            READ_SIMPLE_MTI_X();
            WRITE(ResourceScheduler::TRF(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        case MTI_ROTATE:
        case MTI_PACKED_ROTATE:
        {
            READ_SIMPLE_MTI_X();
            VSIMME_MAP(mask, 1, 2, 3);
            if (4 <= mask && mask <= 7)
            {
                usedImme |= (IMME2 << (mask - 4));
            }
            WRITE(ResourceScheduler::TRF(select));
            WRITE(ResourceScheduler::NWS_LOCK(select));
            break;
        }
        }
    }

    std::vector<int>
    forceSplitPlace()
    {
        return {};
    }
};

struct MTRInst
{
    std::vector<ResourceScheduler> writeResource{};
    std::vector<ResourceScheduler> readResource{};
    Instruction *pInst = nullptr;
    int usedImme = 0;
    uint16_t permit = 0;
    uint16_t op = 0;
    uint16_t dest = 0;
    uint16_t select = 0;

    int vRegCount = 0;
    int sRegCount = 0;

    enum mtrReg 
    {  
        v_dest
    };

    std::vector<regTableEntry> regTable{1, regTableEntry()}; 

    void setVirReg(const ResourceScheduler& res, int virtualId) 
    {
        for(auto& entry : regTable) 
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                entry.virRegId = virtualId;
            }
        }
    }

    int getVirReg(const ResourceScheduler& res) const
    {
        for (auto& entry : regTable)
        {
            if (entry.flag && entry.tag == res.entry && entry.phyRegId == res.regId) 
            {
                return entry.virRegId;
            }
        }
    }

    static MTRInst
    read(Instruction &inst, const InstHint &hints)
    {
        auto op = reinterpret_cast<MTROperationState *>(
            inst.GetOperation(Instruction::MTR));
        MTRInst res{};
        res.pInst = &inst;
        res.permit = op->GetPermissionValue();
        res.op = op->GetOpCode();
        res.dest = op->GetIndexDest();
        res.select = op->GetSelect();
        return res;
    }

    bool
    isNoop()
    {
        return op == MTR_NOOP;
    }

    void
    checkResource()
    {
        switch (op)
        {
        case MTR_NOOP:
        {
            break;
        }
        case MTR_READ_MATRIX_RESULT:
        {
            READ(ResourceScheduler::MRF(select));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case MTR_READ_TRANSPOSE_RESULT:
        {
            READ(ResourceScheduler::TRF(select));
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case MTR_READ_UNARY_EXECUTION_RESULT:
        {
            READ(ResourceScheduler::URF());
            WRITE_SIMPLE_V_DEST();
            break;
        }
        case MTR_READ_FUXI_CORD_RESULT:
        {
            READ(ResourceScheduler::CRF());
            WRITE_SIMPLE_V_DEST();
            break;
        }
        }
    }

    std::vector<int>
    forceSplitPlace()
    {
        return {};
    }
};

struct MISCInst
{
    std::vector<ResourceScheduler> writeResource{};
    std::vector<ResourceScheduler> readResource{};
    Instruction *pInst = nullptr;
    int usedImme = 0;
    uint16_t permit = 0;
    uint16_t op = 0;
    uint16_t operand = 0;
    uint16_t cond = 0;
    uint16_t target = 0;
    InstHint hint;

    int vRegCount = 0;
    int sRegCount = 0;

    void setVirReg(const ResourceScheduler& res, int virtualId) 
    {
        assert(false && "unreachable");
    }

    int getVirReg(const ResourceScheduler& res) const
    {
        assert(false && "unreachable");
    }

    static MISCInst
    read(Instruction &inst, const InstHint &hints)
    {
        auto op = reinterpret_cast<MiscOperationState *>(
            inst.GetOperation(Instruction::MISC));
        MISCInst res{};
        res.pInst = &inst;
        res.permit = op->GetPermissionValue();
        res.op = op->GetOpCode();
        res.operand = op->GetMiscOperand();
        res.cond = op->GetMiscCondition();
        res.target = op->GetMiscTarget();
        res.hint = hints;
        return res;
    }

    bool
    isNoop()
    {
        return op == MISC_NOOP;
    }

    void
    checkResource()
    {
        switch (op)
        {
        case MISC_NOOP:
        {
            break;
        }
        case MISC_SET_SYNC_FLAG:
        {
            if (1 <= operand && operand <= 3)
            {
                usedImme |= VSIMME0 << (operand - 1);
            }
            else if (4 <= operand && operand <= 7)
            {
                usedImme |= IMME2 << (operand - 4);
            }
            if (1 <= target && target <= 3)
            {
                usedImme |= VSIMME0 << (target - 1);
                auto si = pInst->GetImmediateValue(
                    static_cast<Instruction::ImmediateValueType>(Instruction::VECTORSCALARIMMEDIATE0 + target - 1));
                if (hint.sregValue[si].first)
                {
                    WRITE(ResourceScheduler::SYNC_FLAG(hint.sregValue[si].second, 1));
                } else
                {
                    WRITE(ResourceScheduler::SYNC_FLAG(1, 0x1fff));
                }
            }
            else if (4 <= target && target <= 7)
            {
                usedImme |= IMME2 << (target - 4);
                auto t = pInst->GetImmediateValue(static_cast<Instruction::ImmediateValueType>(Instruction::IMMEDIATE2 +
                    target - 4));
                WRITE(ResourceScheduler::SYNC_FLAG(t, 1));
            }
            break;
        }
        case MISC_SYNC_FLAG_INCREMENT:
        case MISC_SYNC:
        {
            break;
        }
        case MISC_VMASK_OPERATION:
        {
            READ(ResourceScheduler::VMaskReg(target));
            READ(ResourceScheduler::VMaskReg(operand));
            WRITE(ResourceScheduler::VMaskReg(target));
            break;
        }
        case MISC_READ_SYNC_FLAG:
        case MISC_INTERRUPT:
        {
            break;
        }
        case MISC_CLEAR_RESULT_FIFO:
        {
            if (operand == 0)
            {
                WRITE(ResourceScheduler::URF());
            }
            else if (operand == 1)
            {
                WRITE(ResourceScheduler::MRF(0));
            }
            else if (operand == 2)
            {
                WRITE(ResourceScheduler::MRF(1));
            }
            else if (operand == 3)
            {
                WRITE(ResourceScheduler::TRF(0));
            }
            else if (operand == 4)
            {
                WRITE(ResourceScheduler::TRF(1));
            }
            break;
        }
        case MISC_VECTOR_DELAY_SHORT:
        case MISC_VECTOR_DELAY_LONG:
        case MISC_REMOTE_SET_SYNC_FLAG:
        case MISC_REMOTE_SYNC_FLAG_INCREMENT:
        case MISC_TRACE:
        case MISC_SET_TRACEMARK:
        case MISC_CFENCE:
        {
            break;
        }
        }
    }

    std::vector<int>
    forceSplitPlace()
    {
        switch (op)
        {
        case MISC_SET_SYNC_FLAG:
        case MISC_SYNC_FLAG_INCREMENT:
        case MISC_SYNC:
        case MISC_READ_SYNC_FLAG:
        case MISC_INTERRUPT:
        case MISC_VECTOR_DELAY_SHORT:
        case MISC_VECTOR_DELAY_LONG:
        case MISC_REMOTE_SET_SYNC_FLAG:
        case MISC_REMOTE_SYNC_FLAG_INCREMENT:
        case MISC_TRACE:
        case MISC_SET_TRACEMARK:
        case MISC_CFENCE:
            return {1, 0};
        }
        return {};
    }
};

#undef READ
#undef WRITE
#undef READ_SIMPLE_S_X
#undef CHECK_Y_IMME_USE
#undef READ_SIMPLE_S_Y
#undef READ_SIMPLE_V_X
#undef VSIMME_MAP
#undef READ_SIMPLE_V_Y
#undef WRITE_SIMPLE_S_DEST
#undef WRITE_SIMPLE_V_DEST
#undef WRITE_SIMPLE_P_DEST
#undef WRITE_SIMPLE_VMASK_DEST
#undef WRITE_SIMPLE_MRF
#undef READ_SIMPLE_MTI_X
#undef READ_SIMPLE_MTI_MASK
#undef READ_SIMPLE_V_LOADSTORE_BASE
#undef READ_SIMPLE_V_LOADSTORE_STRIDE
#undef READ_SHUFFLE_V_LOAD_STRIDE
#undef READ_SIMPLE_V_LOADSTORE_MASK

struct Inst233
{
    enum class Tag
    {
        Scalar,
        Vector,
        VectorLoad,
        VectorStore,
        MTI,
        MTR,
        MISC
    };

    Tag tag;
    int rawInstId;
    int InstId;
    Instruction* rawInst;

    ScalarInst scalarInst;
    VectorInst vectorInst;
    VectorLoadInst vectorLoadInst;
    VectorStoreInst vectorStoreInst;
    MTIInst mtiInst;
    MTRInst mtrInst;
    MISCInst miscInst;

    static Inst233
    readScalar0(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Tag::Scalar;
        res.scalarInst = ScalarInst::read(inst, hint, true);
        return res;
    }

    static Inst233
    readScalar1(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::Scalar;
        res.scalarInst = ScalarInst::read(inst, hint, false);
        return res;
    }

    static Inst233
    readVector0(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::Vector;
        res.vectorInst = VectorInst::read(inst, hint, true);
        return res;
    }

    static Inst233
    readVector1(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::Vector;
        res.vectorInst = VectorInst::read(inst, hint, false);
        return res;
    }

    static Inst233
    readVectorLoad(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::VectorLoad;
        res.vectorLoadInst = VectorLoadInst::read(inst, hint);
        return res;
    }

    static Inst233
    readVectorStore(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::VectorStore;
        res.vectorStoreInst = VectorStoreInst::read(inst, hint);
        return res;
    }

    static Inst233
    readMTI(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::MTI;
        res.mtiInst = MTIInst::read(inst, hint);
        return res;
    }

    static Inst233
    readMTR(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::MTR;
        res.mtrInst = MTRInst::read(inst, hint);
        return res;
    }

    static Inst233
    readMISC(Instruction &inst, const InstHint &hint)
    {
        Inst233 res;
        res.tag = Inst233::Tag::MISC;
        res.miscInst = MISCInst::read(inst, hint);
        return res;
    }

#define DELIVER(ret, fn, call)                                                 \
    ret fn                                                                     \
    {                                                                          \
        switch (tag)                                                           \
        {                                                                      \
        case Tag::Vector:                                                      \
            return vectorInst.call;                                            \
        case Tag::VectorStore:                                                 \
            return vectorStoreInst.call;                                       \
        case Tag::VectorLoad:                                                  \
            return vectorLoadInst.call;                                        \
        case Tag::MTR:                                                         \
            return mtrInst.call;                                               \
        case Tag::MISC:                                                        \
            return miscInst.call;                                              \
        case Tag::MTI:                                                         \
            return mtiInst.call;                                               \
        case Tag::Scalar:                                                      \
            return scalarInst.call;                                            \
        }                                                                      \
    }

    DELIVER(bool, isNoop(), isNoop())

    DELIVER(void, checkResource(), checkResource())

    //    +place 1  +place 0
    //    v         v
    // ---+---------+
    //    | curInst |
    // ---+---------+
    DELIVER(std::vector<int>, forceSplitPlace(), forceSplitPlace())

    DELIVER(std::vector<ResourceScheduler>&, readResource(), readResource)

    DELIVER(std::vector<ResourceScheduler>&, writeResource(), writeResource)

    DELIVER(void, setVirReg(const ResourceScheduler& res, int virtualId), setVirReg(res, virtualId))

    DELIVER(int, getVirReg(const ResourceScheduler& res), getVirReg(res))

    DELIVER(int, sRegCount(), sRegCount)

    DELIVER(int, vRegCount(), vRegCount)

    DELIVER(int, usedImme(), usedImme)

#undef DELIVER
};

std::vector<Inst233>
read(Instruction &inst, const InstHint &hint)
{
    return {Inst233::readScalar0(inst, hint),
            Inst233::readScalar1(inst, hint),
            Inst233::readVector0(inst, hint),
            Inst233::readVector1(inst, hint),
            Inst233::readVectorLoad(inst, hint),
            Inst233::readVectorStore(inst, hint),
            Inst233::readMTI(inst, hint),
            Inst233::readMTR(inst, hint),
            Inst233::readMISC(inst, hint)};
}

std::unordered_map<Inst233 *, int> DAGHeight;

bool 
instCmp(Inst233* i1, Inst233* i2)
{
    if (!DAGHeight.empty())
    {
        if (i1->vRegCount() != i2->vRegCount())
        {
            return i1->vRegCount() < i2->vRegCount();
        }
        if (i1->sRegCount() != i2->sRegCount())
        {
            return i1->sRegCount() < i2->sRegCount();
        }
        if (DAGHeight[i1] != DAGHeight[i2])
        {
            return DAGHeight[i1] > DAGHeight[i2];
        }
        else
        {
            return i1 > i2;
        }
    }
    else
    {
        return i1 > i2;
    }
}

struct RegAllocBase
{    
    RegAllocBase() = default;
    RegAllocBase(std::unordered_map<int, int>&& refCount)
                : vRegRefCount(refCount), regRefCount() {}

    std::unordered_map<int, int> mapTable;
    std::unordered_map<int, int> vRegRefCount;
    std::array<std::pair<int, int>, 32> regRefCount;
    uint16_t allocIndex = 0;

    inline size_t size() const
    {
        return mapTable.size();
    }

    uint16_t getNewReg()
    {
        uint16_t i = allocIndex;
        do
        {
            if (refCount(i) == 0)
            {
                allocIndex = (i + 1) % 32;
                return i;
            }
            i = (i + 1) % 32;
        } while (i != allocIndex);
        assert(false && "reg exhaustion");
    }

    uint16_t alloc(const int virRegId)
    {
        uint16_t ret = 74;
        auto it = mapTable.find(virRegId);
        if (it != mapTable.end())
        {
            ret = it->second;
        }
        else
        {
            ret = getNewReg();
            regRefCount[ret] = {virRegId, vRegRefCount.at(virRegId)};
            mapTable.insert({virRegId, ret});
        }
        return ret;
    }

    inline int getVRegId(const uint16_t phyRegId) const
    {
        return regRefCount[phyRegId].first;
    }

    inline int& refCount(const uint16_t phyRegId)
    {
        return regRefCount[phyRegId].second;
    }

    inline auto unMap(const uint16_t phyRegId)
    {
        return mapTable.erase(getVRegId(phyRegId));
    }
};

struct InstRegAlloc 
{
    InstRegAlloc(RegAllocBase* Allocator) : Allocator(Allocator) {}

    std::vector<uint16_t> instRegList;
    RegAllocBase* Allocator;

    uint16_t
    alloc(int virtualId)
    {
        uint16_t phyRegId = Allocator->alloc(virtualId);
        instRegList.push_back(phyRegId);
        return phyRegId;
    }

    ~InstRegAlloc()
    {
        for(auto& regId : instRegList)
        { 
            if(--Allocator->refCount(regId) == 0)
            {
                Allocator->unMap(regId);
            }
        }
    }
};

RegAllocBase sregAllocBase;
RegAllocBase vregAllocBase;

struct DAG
{
    DAG() : availableNodes(instCmp) {}
    
    std::unordered_map<Inst233 *, std::unordered_set<Inst233 *>> edge;
    std::unordered_map<std::tuple<Inst233 *, Inst233 *>,
                       std::vector<ResourceScheduler>> edgeRes;
    
    std::unordered_map<Inst233 *, int> inDegree;
    std::unordered_set<Inst233 *> nodes;

    std::set<Inst233 * , std::function<bool(Inst233*, Inst233*)>> availableNodes;
    size_t edgeSize = 0;

    void 
    reSortAvailableNodes()
    {
        availableNodes.clear();
        for (auto& n : nodes)
        {
            if(inDegree[n] == 0)
            {
                availableNodes.insert(n);
            }
        }
    }

    void
    makeEdge(Inst233 *from, Inst233 *to, ResourceScheduler r)
    {
        auto res = edge[from].insert(to);
        edgeRes[std::make_tuple(from, to)].push_back(r);
        nodes.insert(from);
        nodes.insert(to);
        if (inDegree[from] == 0)
        {
            availableNodes.insert(from);
        }
        availableNodes.erase(to);
        if (res.second)
        {
            inDegree[to]++;
            edgeSize++;
        }
    }

    bool
    empty()
    {
        return nodes.empty();
    }

    std::unordered_set<Inst233 *> peeked{};

    bool
    peek(const std::function<bool(Inst233 *)> &lookAndTake)
    {
        for (auto &n : availableNodes)
        {
            if ((vregAllocBase.size() >= 28 || sregAllocBase.size() >= 28) &&
                (n->vRegCount() > 0 || n->sRegCount() > 0)) 
            {
                return false;
            }
            if (lookAndTake(n))
            {
                peeked.insert(n);
                // std::cout << "DAGHeight: " << DAGHeight[n] << std::endl;
                availableNodes.erase(n);
                return true;
            }
        }
        return false;
    }

    void
    resolvePeekedStage1()
    {
        for (auto &p : peeked)
        {
            take(p);
        }
        peeked.clear();
    }

    std::vector<Inst233 *> prepareEraseFromEdge;

    void
    resolvePeekedStage2()
    {
        for (auto &p : prepareEraseFromEdge)
        {
            edge.erase(p);
        }
        prepareEraseFromEdge.clear();
    }

    void
    take(Inst233 *p)
    {
        for (auto &e : edge[p])
        {
            if (--inDegree[e] == 0)
            {
                availableNodes.insert(e);
            }
        }
        nodes.erase(p);
        prepareEraseFromEdge.push_back(p);
    }

    void
    getHeight()
    {
        for (auto & node : nodes)
        {
            getNodeHeight(node);
        }
    }

    int
    getNodeHeight(Inst233* node)
    {
        if (DAGHeight.find(node) != DAGHeight.end())
        {
            return DAGHeight[node];
        }
        if (edge.find(node) == edge.end())
        {
            DAGHeight[node] = 1;
        }
        else
        {
            int maxChildHeight = 0;
            for (auto &child : edge[node])
            {
                maxChildHeight = std::max(maxChildHeight, getNodeHeight(child));
            }
            DAGHeight[node] = maxChildHeight + 1;
        }
        return DAGHeight[node];
    }
};

struct ResourceLinkLock
{
    ResourceLinkLock() = default;
    ResourceLinkLock(ResourceScheduler res) : res(res) {}

    ResourceScheduler res;
};

struct MakeDependencyLock
{
    MakeDependencyLock() = default;
    MakeDependencyLock(std::function<void(Inst233 *, Inst233 *, ResourceScheduler)> cb)
                        : onMakeDependency(cb) {}

    std::function<void(Inst233 *, Inst233 *, ResourceScheduler)>
        onMakeDependency;
};

template <class T>
struct RegisterLock : MakeDependencyLock, ResourceLinkLock
{
    T owner;
    std::vector<T> reader;
    bool hasOwner = false;

    void
    write(T t)
    {
        hasOwner = true;
        owner = t;
        for (auto &r : reader)
        {
            onMakeDependency(r, t, res);
        }
        reader.clear();
    }

    void
    read(T t)
    {
        if (hasOwner)
        {
            reader.push_back(t);
            onMakeDependency(owner, t, res);
        }
    }
};

template <class T>
struct VirtualRegisterLock : MakeDependencyLock, ResourceLinkLock
{
    T owner;
    std::vector<T> reader;
    bool hasOwner = false;

    VirtualRegisterLock(ResourceScheduler res, 
                        std::function<void(Inst233 *, Inst233 *, ResourceScheduler)> cb)
                        : MakeDependencyLock(cb), ResourceLinkLock(res) {}

    void
    write(T t)
    {
        hasOwner = true;
        owner = t;
        for (auto &r : reader)
        {
            onMakeDependency(r, t, res);
        }
        reader.clear();
    }

    void
    read(T t)
    {
        if (hasOwner)
        {
            reader.push_back(t);
            onMakeDependency(owner, t, res);
        }
    }
};

template <class T>
struct OrderWriteRegisterLock : MakeDependencyLock, ResourceLinkLock
{
    T owner;
    std::vector<T> reader;
    bool hasOwner = false;

    void
    write(T t)
    {
        if (hasOwner && reader.empty())
        {
            onMakeDependency(owner, t, res);
        }
        hasOwner = true;
        owner = t;
        // for (auto &r : reader)
        // {
        //     onMakeDependency(r, t, res);
        // }
        reader.clear();
    }

    void
    read(T t)
    {
        if (hasOwner)
        {
            reader.push_back(t);
            onMakeDependency(owner, t, res);
        }
    }
};

template <class T, size_t S>
struct RangeLock : MakeDependencyLock, ResourceLinkLock
{
    std::array<T, S> owner;
    std::array<std::vector<T>, S> reader;
    std::array<bool, S> inited;

    RangeLock()
    {
        inited.fill(false);
    }

    void
    write(T t, size_t addr, size_t size)
    {
        for (size_t i = 0; i < size && addr + i < S; i++)
        {
            inited[addr + i] = true;
            owner[addr + i] = t;
            for (auto &r : reader[addr + i])
            {
                onMakeDependency(r, t, res);
            }
            reader[addr + i].clear();
        }
    }
    void
    read(T t, size_t addr, size_t size)
    {
        for (size_t i = 0; i < size && addr + i < S; i++)
        {
            if (inited[addr + i])
            {
                reader[addr + i].push_back(t);
                onMakeDependency(owner[addr + i], t, res);
            }
        }
    }
};

template <class T>
struct SymmetryFifoLock : MakeDependencyLock, ResourceLinkLock
{
    std::deque<Inst233 *> owners;

    void
    write(T t)
    {
        owners.push_back(t);
    }

    void
    read(T t)
    {
        T ret = owners.front();
        owners.pop_front();
        onMakeDependency(ret, t, res);
    }
};

template <class T>
struct AsymmetryFifoLock : MakeDependencyLock, ResourceLinkLock
{
    std::deque<T> fifo;
    T lastR = {};
    bool hasLastR = false;
    int FiFOCount = 0;

    void
    write(T t)
    {
        if (fifo.empty())
        {
            fifo.push_back(t);
        }
        else
        {
            auto lastW = fifo.front();
            fifo.pop_front();
            onMakeDependency(lastW, t, res);
            fifo.push_back(t);
        }
        hasLastR = false;
    }

    void
    read(T t)
    {
        if (hasLastR)
        {
            onMakeDependency(lastR, t, res);
        }
        else if (!fifo.empty())
        {
            auto lastW = fifo.front();
            fifo.pop_front();
            onMakeDependency(lastW, t, res);
        } 
        else {
            // std::cout << COLOR::RED << "WARNING:FIFO read before write!" << std::endl;
        }
        hasLastR = true;
        lastR = t;
    }
};

template <class T>
struct ReadOnceFifoLock : MakeDependencyLock, ResourceLinkLock
{
    std::deque<T> fifo;
    int FiFOCount = 0;
    void
    write(T t)
    {
        if (fifo.empty())
        {
            fifo.push_back(t);
        }
        else
        {
            auto lastW = fifo.front();
            fifo.pop_front();
            onMakeDependency(lastW, t, res);
            fifo.push_back(t);
        }
    }

    void
    read(T t)
    {
        if (!fifo.empty())
        {
            auto lastW = fifo.front();
            fifo.pop_front();
            onMakeDependency(lastW, t, res);
        } 
        else 
        {
            std::cout << COLOR::RED << "WARNING:FIFO read before write!" << std::endl;
        }
    }
};

template <class T>
struct WriteOnceFifoLock : MakeDependencyLock, ResourceLinkLock
{
    std::deque<T> fifo;

    void
    write(T t)
    {
        fifo.clear();
        fifo.push_back(t);
    }

    void
    read(T t)
    {
        if (!fifo.empty())
        {
            auto last = fifo.back();
            onMakeDependency(last, t, res);
            fifo.push_back(t);
        } 
        else 
        {
            std::cout << COLOR::RED << "WARNING:FIFO read before write!" << std::endl;
        }
    }
};

template <class T>
struct GMRLock : MakeDependencyLock, ResourceLinkLock
{
    std::deque<T> fifo;

    void
    write(T t)
    {
        if (fifo.empty())
        {
            fifo.push_back(t);
        }
        else
        {
            fifo.front() = t;
        }
    }

    void
    read(T t)
    {
        if (!fifo.empty())
        {
            auto lastW = fifo.front();
            onMakeDependency(lastW, t, res);
        }
        else 
        {
            std::cout << COLOR::RED << "WARNING:GMR read before write!" << std::endl;
        }
    }
};

std::unordered_map<int, int>& vVRegRefCount = vregAllocBase.vRegRefCount;
std::unordered_map<int, int>& vSRegRefCount = sregAllocBase.vRegRefCount;

struct ResourcePool
{
    std::vector<VirtualRegisterLock<Inst233 *>> sreg;
    std::vector<VirtualRegisterLock<Inst233 *>> vreg;
    std::unordered_map<int, int> sregMapTable;
    std::unordered_map<int, int> vregMapTable;

    RegisterLock<Inst233 *> permit[8];
    RegisterLock<Inst233 *> vmask[8];
    RegisterLock<Inst233 *> pcr[2];
    RegisterLock<Inst233 *> spr[2];
    RegisterLock<Inst233 *> ia[2];

    RegisterLock<Inst233 *> smem;
    RangeLock<Inst233 *, 4096 * 1024 / 32> vmem;
    RegisterLock<Inst233 *> cmem;
    RegisterLock<Inst233 *> hbm;
    RangeLock<Inst233 *, 0x2000> sync;

    OrderWriteRegisterLock<Inst233 *> nws[2];

    // RangeLock<Inst233 *, 256 * 1024, 1> smem;
    // RangeLock<Inst233 *, 4096 * 1024 / 128, 8> vmem;
    // RangeLock<Inst233 *, 4096 * 1024 / 128, 8> cmem;
    GMRLock<Inst233 *> gmr[2];

    SymmetryFifoLock<Inst233 *> vsf;
    SymmetryFifoLock<Inst233 *> crf;
    SymmetryFifoLock<Inst233 *> urf;
    SymmetryFifoLock<Inst233 *> mrf[2];
    
    AsymmetryFifoLock<Inst233 *> trf[2];
    ReadOnceFifoLock<Inst233 *> gsnf[2];
    ReadOnceFifoLock<Inst233 *> gstf[2];


    std::function<void(Inst233 *, Inst233 *, ResourceScheduler)> onMakeDependency;

    void
    registerOnMakeDependency(
        std::function<void(Inst233 *, Inst233 *, ResourceScheduler)> cb)
    {
        onMakeDependency = cb;
#define REG_MULTI(var, r)                                                      \
    do                                                                         \
    {                                                                          \
        for (int i = 0; i < (sizeof(var) / sizeof(var[0])); i++)               \
        {                                                                      \
            (var)[i].onMakeDependency = cb;                                    \
            (var)[i].res = r(i);                                               \
        }                                                                      \
    } while (false)
#define REG_SINGLE(var, r)                                                     \
    do                                                                         \
    {                                                                          \
        (var).onMakeDependency = cb;                                           \
        (var).res = r();                                                       \
    } while (false)
        REG_MULTI(permit, ResourceScheduler::PermitReg);
        REG_MULTI(vmask, ResourceScheduler::VMaskReg);
        REG_MULTI(ia, ResourceScheduler::IAReg);
        REG_MULTI(pcr, ResourceScheduler::PCR);
        REG_MULTI(spr, ResourceScheduler::SPR);
        REG_MULTI(gmr, ResourceScheduler::GMR);
        REG_SINGLE(smem, ResourceScheduler::SMem);
        do
        {
            (vmem).onMakeDependency = cb;
            (vmem).res = ResourceScheduler::VMem(0, 4096 * 1024 / 32);
        } while (false);
        do
        {
            (sync).onMakeDependency = cb;
            (sync).res = ResourceScheduler::SYNC_FLAG(0, 0x2000);
        } while (false);
        REG_SINGLE(cmem, ResourceScheduler::CMem);
        REG_SINGLE(hbm, ResourceScheduler::HBM);
        REG_SINGLE(vsf, ResourceScheduler::VSF);
        REG_SINGLE(crf, ResourceScheduler::CRF);
        REG_SINGLE(urf, ResourceScheduler::URF);
        REG_MULTI(mrf, ResourceScheduler::MRF);
        REG_MULTI(trf, ResourceScheduler::TRF);
        REG_MULTI(gsnf, ResourceScheduler::GSNF);
        REG_MULTI(gstf, ResourceScheduler::GSTF);
        REG_MULTI(nws, ResourceScheduler::NWS_LOCK);
#undef REG_MULTI
#undef REG_SINGLE
    }

    void
    read(ResourceScheduler &res, Inst233 *pInst)
    {
#define CASE_MREG(tag, lock)                                                   \
    case ResourceScheduler::Tag::tag:                                          \
    {                                                                          \
        (lock)[res.regId].read(pInst);                                         \
        break;                                                                 \
    }

#define CASE_SREG(tag, lock)                                                   \
    case ResourceScheduler::Tag::tag:                                          \
    {                                                                          \
        (lock).read(pInst);                                                    \
        break;                                                                 \
    }

#define CASE_MEM(tag, lock)                                                    \
    case ResourceScheduler::Tag::tag:                                          \
    {                                                                          \
        (lock).read(pInst, res.range.addr, res.range.size);                    \
        break;                                                                 \
    }
        switch (res.tag)
        {
            case ResourceScheduler::Tag::ScalarReg:
            {
                int virtualId;
                auto it = sregMapTable.find(res.regId);
                if (it != sregMapTable.end()) 
                {
                    virtualId = it->second;
                    // std::cout << "read Sreg map " << res.regId << " to " << virtualId << std::endl;
                } 
                else 
                {
                    virtualId = sreg.size();
                    sreg.emplace_back(res, onMakeDependency);
                    sregMapTable.insert({res.regId, virtualId});
                    // std::cout << "read Sreg map " << res.regId << " to " << virtualId << std::endl;
                }
                pInst->setVirReg(res, virtualId);
                vSRegRefCount[virtualId]++;
                sreg.at(virtualId).read(pInst);
                break;
            }
            case ResourceScheduler::Tag::VectorReg:
            {
                int virtualId;
                auto it = vregMapTable.find(res.regId);
                if (it != vregMapTable.end()) 
                {
                    virtualId = it->second;
                    // std::cout << "read Vreg: map " << res.regId << " to " << virtualId << std::endl;
                } 
                else 
                {
                    virtualId = vreg.size();
                    vreg.emplace_back(res, onMakeDependency);
                    vregMapTable.insert({res.regId, virtualId});
                    // std::cout << "read Vreg: map " << res.regId << " to " << virtualId << std::endl;
                }
                pInst->setVirReg(res, virtualId);
                vVRegRefCount[virtualId]++;
                vreg.at(virtualId).read(pInst);
                break;
            }
            case ResourceScheduler::Tag::GSTF:
            {
                gstf[res.regId].read(pInst);
                res.serialId = gstf[res.regId].FiFOCount;
                gstf[res.regId].FiFOCount++;
                break;
            }
            case ResourceScheduler::Tag::GSNF:
            {
                gsnf[res.regId].read(pInst);
                res.serialId = gsnf[res.regId].FiFOCount;
                gsnf[res.regId].FiFOCount++;
                break;
            }
            CASE_MREG(PermitReg, permit)
            CASE_MREG(VMaskReg, vmask)
            CASE_MREG(PCR, pcr)
            CASE_MREG(SPR, spr)
            CASE_MREG(GMR, gmr)
            CASE_MREG(IAReg, ia)
            CASE_SREG(SMem, smem)
            CASE_SREG(CMem, cmem)
            CASE_SREG(HBM, hbm)
            CASE_MEM(VMem, vmem)
            CASE_MEM(SYNC_FLAG, sync)
            CASE_SREG(VSF, vsf)
            CASE_SREG(URF, urf)
            CASE_SREG(CRF, crf)
            CASE_MREG(MRF, mrf)
            // CASE_MREG(TRF, trf)
            case ResourceScheduler::Tag::TRF:
            {
                trf[res.regId].read(pInst);
                res.serialId = trf[res.regId].FiFOCount;
                // std::cout << "pop: " << res.serialId << std::endl;
                break;
            }
            CASE_MREG(NWS_LOCK, nws)
        }

#undef CASE_MREG
#undef CASE_SREG
#undef CASE_MEM
    }

    void
    write(ResourceScheduler &res, Inst233 *pInst)
    {
#define CASE_MREG(tag, lock)                                                   \
    case ResourceScheduler::Tag::tag:                                          \
    {                                                                          \
        (lock)[res.regId].write(pInst);                                        \
        break;                                                                 \
    }

#define CASE_SREG(tag, lock)                                                   \
    case ResourceScheduler::Tag::tag:                                          \
    {                                                                          \
        (lock).write(pInst);                                                   \
        break;                                                                 \
    }

#define CASE_MEM(tag, lock)                                                    \
    case ResourceScheduler::Tag::tag:                                          \
    {                                                                          \
        (lock).write(pInst, res.range.addr, res.range.size);                   \
        break;                                                                 \
    }
        switch (res.tag)
        {
            case ResourceScheduler::Tag::ScalarReg:
            {
                int virtualId = sreg.size();
                sreg.emplace_back(res, onMakeDependency);
                sregMapTable[res.regId] = virtualId;

                pInst->setVirReg(res, virtualId);
                vSRegRefCount[virtualId]++;
                sreg.at(virtualId).write(pInst);
                // std::cout << "write Sreg: map " << res.regId 
                //             << " to " << virtualId << std::endl;
                break;
            }
            case ResourceScheduler::Tag::VectorReg:
            {
                int virtualId = vreg.size();
                vreg.emplace_back(res, onMakeDependency);
                vregMapTable[res.regId] = virtualId;
                
                pInst->setVirReg(res, virtualId);
                vVRegRefCount[virtualId]++;
                vreg.at(virtualId).write(pInst);
                // std::cout << "write Vreg: map " << res.regId 
                //             << " to " << virtualId << std::endl;
                break;
            }
            case ResourceScheduler::Tag::GSTF: 
            {
                gstf[res.regId].write(pInst);
                res.serialId = gstf[res.regId].FiFOCount;
                // std::cout << "FiFOCount: " << res.serialId << std::endl;
                break;
            }
            case ResourceScheduler::Tag::GSNF:
            {
                gsnf[res.regId].write(pInst);
                res.serialId = gsnf[res.regId].FiFOCount;
                break;
            }
            CASE_MREG(PermitReg, permit)
            CASE_MREG(VMaskReg, vmask)
            CASE_MREG(PCR, pcr)
            CASE_MREG(SPR, spr)
            CASE_MREG(GMR, gmr)
            CASE_MREG(IAReg, ia)
            CASE_SREG(SMem, smem)
            CASE_SREG(CMem, cmem)
            CASE_MEM(VMem, vmem)
            CASE_SREG(VSF, vsf)
            CASE_SREG(URF, urf)
            CASE_SREG(CRF, crf)
            CASE_MREG(MRF, mrf)
            // CASE_MREG(TRF, trf)
            case ResourceScheduler::Tag::TRF:
            {
                trf[res.regId].write(pInst);
                res.serialId = trf[res.regId].FiFOCount;
                break;
            }
            case ResourceScheduler::Tag::NWS_LOCK:
            {
                nws[res.regId].write(pInst);
                trf[res.regId].FiFOCount++;
                break;
            }
            // CASE_MREG(NWS_LOCK, nws)
        }

#undef CASE_MREG
#undef CASE_SREG
#undef CASE_MEM
    }
};

DAG
buildDependency(std::vector<Inst233> &insts)
{
    DAG g;
    ResourcePool pool;
    bool makeEdged = false;
    pool.registerOnMakeDependency(
        [&g, &makeEdged](Inst233 *from, Inst233 *to, ResourceScheduler r)
        {
            if (from != to)
            {
                makeEdged = true;
                g.makeEdge(from, to, r);
            }
        });

    for (auto &inst : insts)
    {
        g.nodes.insert(&inst);
        makeEdged = false;
        for (auto &readResource : inst.readResource())
        {
            pool.read(readResource, &inst);
        }
        for (auto &writeResource : inst.writeResource())
        {
            pool.write(writeResource, &inst);
        }
        if (!makeEdged)
        {
            g.availableNodes.insert(&inst);
        }
    }

    return g;
}

constexpr int Slot0 = 1;
constexpr int Slot1 = 2;
constexpr int SlotBoth = 4;
const std::map<int, int> scalarSlot = {
    {S_NOOP, 0},
    {S_HALT, Slot0 | Slot1},
    {S_POP, Slot0 | Slot1},
    {S_DELAY, Slot0 | Slot1},
    {S_SMEM_LOAD, Slot1},
    {S_SMEM_LOAD_OFFSET, Slot1},
    {S_SMEM_STORE, Slot1},
    {S_SET, Slot0 | Slot1},
    {S_BRANCH, Slot0},
    {S_CALL_ABOSOLUTE, Slot0},
    {S_CALL_RELATIVE, Slot0},
    {S_CALL_REGISTER, Slot0},
    {S_FENCE, Slot0 | Slot1 | SlotBoth},
    {S_DMA, Slot1},
    {S_LOCAL_DMA, Slot0},
    {S_STRIDED_DMA, Slot0},
    {S_READ, Slot0 | Slot1 | SlotBoth},
    {S_CONVERT_S32_TO_F32, Slot0 | Slot1 | SlotBoth},
    {S_CONVERT_F32_TO_S32, Slot0 | Slot1 | SlotBoth},
    {S_S32_ADDITION, Slot0 | Slot1 | SlotBoth},
    {S_S32_SUBTRACTION, Slot0 | Slot1 | SlotBoth},
    {S_U32_AND, Slot0 | Slot1 | SlotBoth},
    {S_U32_OR, Slot0 | Slot1 | SlotBoth},
    {S_U32_XOR, Slot0 | Slot1 | SlotBoth},
    {S_U32_SHIFTLEFT, Slot1 | SlotBoth},
    {S_U32_SHIFTRIGHT, Slot1 | SlotBoth},
    {S_U32_MOVE, Slot0 | Slot1 | SlotBoth},
    {S_U32_COUNTLEADINGZEROES, Slot0 | Slot1 | SlotBoth},
    {S_U32_MULTIPLICATION, Slot0},
    {S_F32_ADDITION, Slot1},
    {S_F32_SUBTRACTION, Slot1},
    {S_F32_MULTIPLICATION, Slot0},
    {S_F32_MAX, Slot0 | Slot1 | SlotBoth},
    {S_F32_MIN, Slot0 | Slot1 | SlotBoth},
    {S_S32_EQUAL, Slot0 | Slot1 | SlotBoth},
    {S_S32_NOTEQUAL, Slot0 | Slot1 | SlotBoth},
    {S_S32_GREATER, Slot0 | Slot1 | SlotBoth},
    {S_S32_GREATEREQUAL, Slot0 | Slot1 | SlotBoth},
    {S_S32_LESSER, Slot0 | Slot1 | SlotBoth},
    {S_S32_LESSER_EQUAL, Slot0 | Slot1 | SlotBoth},
    {S_U32_CARRY, Slot0 | Slot1 | SlotBoth},
    {S_F32_EQUAL, Slot0 | Slot1 | SlotBoth},
    {S_F32_NOTEQUAL, Slot0 | Slot1 | SlotBoth},
    {S_F32_GREATER, Slot0 | Slot1 | SlotBoth},
    {S_F32_GREATEREQUAL, Slot0 | Slot1 | SlotBoth},
    {S_F32_LESSER, Slot0 | Slot1 | SlotBoth},
    {S_F32_LESSEREQUAL, Slot0 | Slot1 | SlotBoth},
    {S_F32_IS_INF_OR_NAN, Slot0 | Slot1 | SlotBoth},
    {S_PERMISSION_OR, Slot0 | Slot1 | SlotBoth},
    {S_ARITHMETIC_SHIFT_RIGHT, Slot1},
};
const std::map<int, int> vectorSlot = {
    {V_NOOP, 0},
    {V_HALF_FLOAT_PACK, Slot0 | Slot1 | SlotBoth},
    {V_TWO_LOWER_INT8_PACK, Slot0 | Slot1 | SlotBoth},
    {V_FOUR_INT8_PACK, Slot0 | Slot1 | SlotBoth},
    {V_SUBCORE_ROTATE, Slot0 | Slot1 | SlotBoth},
    {V_RNG_GENERATE_RANDOM_NUMBER, Slot1},
    {V_RNG_READ_SEED, Slot1},
    {V_RNG_RESEED, Slot1},
    {V_F32_SOFTSIGN, Slot0},
    {V_F32_LOG2, Slot0},
    {V_F32_SIGMOID, Slot0},
    {V_F32_RECIPROCAL, Slot0},
    {V_F32_SQUAREROOT_RECIPROCAL, Slot0},
    {V_F32_POWER, Slot0},
    {V_F32_SOFTPLUS, Slot0},
    {V_F32_EXPONENT, Slot0},
    {V_CONVERT_S32_TO_F32, Slot0 | Slot1 | SlotBoth},
    {V_CONVERT_F32_TO_S32, Slot0 | Slot1 | SlotBoth},
    {V_S32_ADDITION, Slot0 | Slot1 | SlotBoth},
    {V_S32_SUBTRACTION, Slot0 | Slot1 | SlotBoth},
    {V_U32_AND, Slot0 | Slot1 | SlotBoth},
    {V_U32_OR, Slot0 | Slot1 | SlotBoth},
    {V_U32_XOR, Slot0 | Slot1 | SlotBoth},
    {V_U32_SHIFTLEFT, Slot0 | Slot1 | SlotBoth},
    {V_U32_SHIFTRIGHT, Slot0 | Slot1 | SlotBoth},
    {V_U32_MOVE, Slot0 | Slot1 | SlotBoth},
    {V_U32_COUNTLEADINGZEROES, Slot0 | Slot1 | SlotBoth},
    {V_U32_MULTIPLICATION, Slot0 | Slot1 | SlotBoth},
    {V_F32_ADDITION, Slot1},
    {V_F32_SUBTRACTION, Slot1},
    {V_F32_MULTIPLICATION, Slot0},
    {V_F32_MAX, Slot0 | Slot1 | SlotBoth},
    {V_F32_MIN, Slot0 | Slot1 | SlotBoth},
    {V_S32_EQUAL, Slot0 | Slot1 | SlotBoth},
    {V_S32_NOTEQUAL, Slot0 | Slot1 | SlotBoth},
    {V_S32_GREATER, Slot0 | Slot1 | SlotBoth},
    {V_S32_GREATEREQUAL, Slot0 | Slot1 | SlotBoth},
    {V_S32_LESSER, Slot0 | Slot1 | SlotBoth},
    {V_S32_LESSER_EQUAL, Slot0 | Slot1 | SlotBoth},
    {V_U32_CARRY, Slot0 | Slot1 | SlotBoth},
    {V_F32_EQUAL, Slot0 | Slot1 | SlotBoth},
    {V_F32_NOTEQUAL, Slot0 | Slot1 | SlotBoth},
    {V_F32_GREATER, Slot0 | Slot1 | SlotBoth},
    {V_F32_GREATEREQUAL, Slot0 | Slot1 | SlotBoth},
    {V_F32_LESSER, Slot0 | Slot1 | SlotBoth},
    {V_F32_LESSEREQUAL, Slot0 | Slot1 | SlotBoth},
    {V_F32_IS_INF_OR_NAN, Slot0 | Slot1 | SlotBoth},
    {V_PERMISSION_OR, Slot0 | Slot1 | SlotBoth},
    {V_ARITHMETIC_SHIFT_RIGHT, Slot1},
    {V_ROUND_ARITHMETIC_SHIFT_RIGHT, Slot1},
    {V_SELECT_VMASK0, Slot0 | Slot1 | SlotBoth},
    {V_SELECT_VMASK1, Slot0 | Slot1 | SlotBoth},
    {V_SELECT_VMASK2, Slot0 | Slot1 | SlotBoth},
    {V_SELECT_VMASK3, Slot0 | Slot1 | SlotBoth},
    {V_SELECT_VMASK4, Slot0 | Slot1 | SlotBoth},
    {V_SELECT_VMASK5, Slot0 | Slot1 | SlotBoth},
    {V_SELECT_VMASK6, Slot0 | Slot1 | SlotBoth},
    {V_SELECT_VMASK7, Slot0 | Slot1 | SlotBoth},
    {V_GET_V_CORE_ID, Slot0 | Slot1 | SlotBoth},
    {V_SET_VMASK, Slot0 | Slot1 | SlotBoth},
    {V_EXTRACT, Slot0 | Slot1 | SlotBoth},
    {V_COMPOSE_FLOAT, Slot0 | Slot1 | SlotBoth},
    {V_COUNT_NUMBER_OF_ONE, Slot0 | Slot1 | SlotBoth},
    {V_RELUX, Slot0 | Slot1 | SlotBoth},
    {V_CLAMP, Slot0 | Slot1 | SlotBoth},
};

struct FifoHolder
{
    std::deque<Inst233 *> vsfHolder;
    std::deque<Inst233 *> crfHolder;
    std::deque<Inst233 *> urfHolder;
    std::deque<Inst233 *> mrfHolder[2];
    std::deque<Inst233 *> trfHolder[2];
    std::deque<Inst233 *> gsnfHolder[2];
    std::deque<Inst233 *> gstfHolder[2];

    std::deque<Inst233 *> &
    operator[](ResourceScheduler r)
    {
        switch (r.tag)
        {
        case ResourceScheduler::Tag::ScalarReg:
        case ResourceScheduler::Tag::VectorReg:
        case ResourceScheduler::Tag::PermitReg:
        case ResourceScheduler::Tag::VMaskReg:
        case ResourceScheduler::Tag::PCR:
        case ResourceScheduler::Tag::SPR:
        case ResourceScheduler::Tag::IAReg:
        case ResourceScheduler::Tag::SMem:
        case ResourceScheduler::Tag::VMem:
        case ResourceScheduler::Tag::CMem:
        case ResourceScheduler::Tag::NWS_LOCK:
            assert(false && "not fifo");
            break;
        case ResourceScheduler::Tag::VSF:
            return vsfHolder;
        case ResourceScheduler::Tag::URF:
            return urfHolder;
        case ResourceScheduler::Tag::CRF:
            return crfHolder;
        case ResourceScheduler::Tag::MRF:
            return mrfHolder[r.regId];
        case ResourceScheduler::Tag::TRF:
            return trfHolder[r.regId];
        case ResourceScheduler::Tag::GSNF:
            return gsnfHolder[r.regId];
        case ResourceScheduler::Tag::GSTF:
            return gstfHolder[r.regId];
        }
    }

    void
    printInfo()
    {
#define PRINT(who)                                                             \
    do                                                                         \
    {                                                                          \
        std::cout << "before " << #who ": ";                                                \
        for (auto &v : (who))                                                  \
        {                                                                      \
            std::cout << "@" << v->rawInstId << ", ";                          \
        }                                                                      \
        std::cout << "\n";                                                     \
    } while (false)
        PRINT(vsfHolder);
        PRINT(urfHolder);
        PRINT(crfHolder);
        PRINT(mrfHolder[0]);
        PRINT(mrfHolder[1]);
        PRINT(trfHolder[0]);
        PRINT(trfHolder[1]);
        PRINT(gsnfHolder[0]);
        PRINT(gsnfHolder[1]);
        PRINT(gstfHolder[0]);
        PRINT(gstfHolder[1]);
#undef PRINT
    }
};

struct InstructionBuilder
{
    using Res = ResourceScheduler::Tag;

    Inst233 *scalar0 = nullptr;
    Inst233 *scalar1 = nullptr;
    Inst233 *vector0 = nullptr;
    Inst233 *vector1 = nullptr;
    Inst233 *vectorLoad = nullptr;
    Inst233 *vectorStore = nullptr;
    Inst233 *mti = nullptr;
    Inst233 *mtr = nullptr;
    Inst233 *misc = nullptr;
    DAG *gPtr = nullptr;

    FifoHolder fifoHolder;
    std::unordered_set<Inst233* > GMRreaders[2];
    bool inTrans = false;
    bool GMRAvail[2] = {true, true};
    int usedImme = 0;

    std::set<Inst233* > priorScalarInst;

    void setPriorInst()
    {
        for (auto & inst : gPtr->availableNodes)
        {
            if (inst->tag == Inst233::Tag::Scalar)
            {
                const auto &sinst = inst->scalarInst;
                for (auto & res : sinst.writeResource)
                {
                    if (res.isReg())
                    {
                        for (auto & dep : gPtr->edge[inst])
                        {
                            const auto & ress = gPtr->edgeRes[std::make_tuple(inst, dep)];
                            if (std::any_of(ress.begin(), ress.end(), 
                                [res](ResourceScheduler depRes)->bool 
                                { return res == depRes; }) && gPtr->inDegree[dep] == 1)
                            {
                                int costNotChoose = 0;
                                for (auto & depRes : dep->readResource())
                                {
                                    if (!(res == depRes) 
                                        && depRes.tag == Res::VectorReg)
                                    {
                                        costNotChoose++;
                                    }
                                }
                                if (costNotChoose > 0)
                                {
                                    priorScalarInst.insert(inst);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::function<bool(Inst233 *)>
    peekScalar0()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::Scalar || scalar0 != nullptr)
            {
                return false;
            }
            const auto &sinst = inst->scalarInst;
            const auto slotInfo = scalarSlot.at(sinst.op);
            if (!priorScalarInst.empty() && priorScalarInst.find(inst) == priorScalarInst.end())
            {
                return false;
            }
            if ((slotInfo & Slot0) == 0)
            {
                return false;
            }
            if (scalar1 != nullptr && scalar1->scalarInst.op == sinst.op &&
                (slotInfo & SlotBoth) == 0)
            {
                return false;
            }
            if ((sinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= sinst.usedImme;
            scalar0 = inst;
            priorScalarInst.erase(inst);
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekScalar1()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::Scalar || scalar1 != nullptr ||
                (scalar0 != nullptr &&
                 (scalar0->scalarInst.op == S_LOCAL_DMA ||
                  scalar0->scalarInst.op == S_STRIDED_DMA)))
            {
                return false;
            }
            const auto &sinst = inst->scalarInst;
            const auto slotInfo = scalarSlot.at(sinst.op);
            if ((slotInfo & Slot1) == 0)
            {
                return false;
            }
            if (scalar0 != nullptr && scalar0->scalarInst.op == sinst.op &&
                (slotInfo & SlotBoth) == 0)
            {
                return false;
            }
            if ((sinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= sinst.usedImme;
            scalar1 = inst;
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekVector0()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::Vector || vector0 != nullptr)
            {
                return false;
            }
            const auto &vinst = inst->vectorInst;
            const auto slotInfo = vectorSlot.at(vinst.op);
            if ((slotInfo & Slot0) == 0)
            {
                return false;
            }
            if (vector1 != nullptr && vector1->scalarInst.op == vinst.op &&
                (slotInfo & SlotBoth) == 0)
            {
                return false;
            }
            if ((vinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= vinst.usedImme;
            vector0 = inst;
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekVector1()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::Vector || vector1 != nullptr)
            {
                return false;
            }
            const auto &vinst = inst->vectorInst;
            const auto slotInfo = vectorSlot.at(vinst.op);
            if ((slotInfo & Slot1) == 0)
            {
                return false;
            }
            if (vector0 != nullptr && vector0->scalarInst.op == vinst.op &&
                (slotInfo & SlotBoth) == 0)
            {
                return false;
            }
            if ((vinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= vinst.usedImme;
            vector1 = inst;
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekVectorLoad()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::VectorLoad || vectorLoad != nullptr)
            {
                return false;
            }
            const auto &vlinst = inst->vectorLoadInst;
            if ((vlinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= vlinst.usedImme;
            vectorLoad = inst;
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekVectorStore()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::VectorStore ||
                vectorStore != nullptr)
            {
                return false;
            }
            const auto &vsinst = inst->vectorStoreInst;
            if ((vsinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= vsinst.usedImme;
            vectorStore = inst;
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekMTI()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::MTI || mti != nullptr)
            {
                return false;
            }
            const auto &mtiinst = inst->mtiInst;
            for (auto &r : mtiinst.writeResource)
            {
                switch (r.tag)
                {
                case Res::TRF:
                case Res::GSNF:
                case Res::GSTF:
                    if (!fifoHolder[r].empty() && fifoHolder[r].front() != inst)
                    {
                        return false;
                    }
                    break;
                case Res::GMR:
                    if (!GMRAvail[r.regId]) 
                    {
                        return false;
                    }
                    break;
                case Res::MRF:
                    break;
                default:
                    break;
                }
            }
            for (auto &r : mtiinst.readResource)
            {
                switch (r.tag)
                {
                case Res::GMR:
                    if (!GMRAvail[r.regId] && 
                        GMRreaders[r.regId].find(inst) == GMRreaders[r.regId].end()) 
                    {
                        return false;
                    }
                    break;
                case Res::MRF:
                    if (fifoHolder[r].size() >= 16)
                    {
                        return false;
                    }
                    break;
                default:
                    break;
                }
            }
            if ((mtiinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= mtiinst.usedImme;
            mti = inst;
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekMTR()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::MTR || mtr != nullptr)
            {
                return false;
            }
            const auto &mtrinst = inst->mtrInst;
            for (auto &r : mtrinst.readResource)
            {
                if (r.isFifo())
                {
                    if (fifoHolder[r].empty() || fifoHolder[r].front() != inst)
                    {
                        return false;
                    }
                }
            }
            if ((mtrinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= mtrinst.usedImme;
            mtr = inst;
            return true;
        };
    }

    std::function<bool(Inst233 *)>
    peekMISC()
    {
        return [this](Inst233 *inst) -> bool
        {
            if (inst->tag != Inst233::Tag::MISC || misc != nullptr)
            {
                return false;
            }
            const auto &miscinst = inst->miscInst;
            if ((miscinst.usedImme & usedImme) != 0)
            {
                return false;
            }
            usedImme |= miscinst.usedImme;
            misc = inst;
            return true;
        };
    }

    static void
    FillImme(int usedImme, Instruction &dest, Instruction &src)
    {
        if (usedImme & IMME0)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE0,
                src.GetImmediateValue(Instruction::IMMEDIATE0));
        }
        if (usedImme & IMME1)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE1,
                src.GetImmediateValue(Instruction::IMMEDIATE1));
        }
        if (usedImme & IMME2)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE2,
                src.GetImmediateValue(Instruction::IMMEDIATE2));
        }
        if (usedImme & IMME3)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE3,
                src.GetImmediateValue(Instruction::IMMEDIATE3));
        }
        if (usedImme & IMME4)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE4,
                src.GetImmediateValue(Instruction::IMMEDIATE4));
        }
        if (usedImme & IMME5)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE5,
                src.GetImmediateValue(Instruction::IMMEDIATE5));
        }
        if (usedImme & VSIMME0)
        {
            dest.SetImmediateValue(
                Instruction::VECTORSCALARIMMEDIATE0,
                src.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0));
        }
        if (usedImme & VSIMME1)
        {
            dest.SetImmediateValue(
                Instruction::VECTORSCALARIMMEDIATE1,
                src.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1));
        }
        if (usedImme & VSIMME2)
        {
            dest.SetImmediateValue(
                Instruction::VECTORSCALARIMMEDIATE2,
                src.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE2));
        }
    }

    bool
    empty()
    {
        return scalar0 == nullptr && scalar1 == nullptr && vector0 == nullptr &&
               vector1 == nullptr && vectorLoad == nullptr &&
               vectorStore == nullptr && mti == nullptr && mtr == nullptr &&
               misc == nullptr;
    }

    void
    rename(uint16_t& operand,  
           const uint32_t& virtualId, 
           InstRegAlloc& allocator)
    {
        uint16_t newRegId = allocator.alloc(virtualId);
        // std::cout << "rename " << operand 
        //           << " to " << newRegId 
        //           << " serialId ID: " << virtualId << std::endl;
        operand = newRegId;
    }

    Instruction *
    build(bool coutInfo = false)
    {
        InstRegAlloc sregAllocator(&sregAllocBase);
        InstRegAlloc vregAllocator(&vregAllocBase);

        static int count = 0;
        int InstUsedImme = 0;

        if (scalar0 == nullptr && scalar1 == nullptr && vector0 == nullptr &&
            vector1 == nullptr && vectorLoad == nullptr &&
            vectorStore == nullptr && mti == nullptr && mtr == nullptr &&
            misc == nullptr)
        {
            std::cerr << COLOR::RED << "SCHEDULER PEEK INST FAIL" << std::endl;
            exit(-1);
        }
        auto pInst = new Instruction();
        Instruction &inst = *pInst;
        if (coutInfo)
        {
            // std::cout << "#" << count++ << ": " << std::endl;
        }
        if (scalar0 != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << scalar0->rawInstId << ", ";
            }
            auto &sinst = scalar0->scalarInst;
            FillImme(sinst.usedImme, inst, *sinst.pInst);
            for (int i = 0; i < sinst.regTable.size(); ++i)
            {
                if (sinst.regTable[i].flag) 
                {
                    uint32_t virtualId = sinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case ScalarInst::s_x:
                        rename(sinst.x, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_y:
                        rename(sinst.y, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_dest:
                        rename(sinst.dest, virtualId, sregAllocator);
                        break;
                    case ScalarInst::vs_imm_0:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        std::cout << "rename vs0 " 
                                  << inst.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0)
                                  << " to " << newRegId << std::endl;
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, newRegId);
                        break;
                    }
                    case ScalarInst::vs_imm_1:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        std::cout << "rename vs1 " 
                                  << inst.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1)
                                  << " to " << newRegId << std::endl;
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1, newRegId);
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
            if (sinst.op == S_LOCAL_DMA || sinst.op == S_STRIDED_DMA)
            {
                ScalarOperationState state(sinst.op,
                                           sinst.permit,
                                           sinst.src_addr,
                                           sinst.length,
                                           sinst.dest_addr,
                                           sinst.syncflag,
                                           sinst.misc);
                inst.SetOperationState(Instruction::SCALARONE, &state);
            }
            else
            {
                ScalarOperationState state(sinst.op,
                                           sinst.permit,
                                           sinst.x,
                                           sinst.y,
                                           sinst.dest);
                inst.SetOperationState(Instruction::SCALARONE, &state);
            }
            // std::cout << "vreg count: " << sinst.vRegCount << std::endl;
            scalar0 = nullptr;
        }
        if (scalar1 != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << scalar1->rawInstId << ", ";
            }
            auto &sinst = scalar1->scalarInst;
            FillImme(sinst.usedImme, inst, *sinst.pInst);
            for (int i = 0; i < sinst.regTable.size(); ++i)
            {
                if (sinst.regTable[i].flag) 
                {
                    uint32_t virtualId = sinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case ScalarInst::s_x:
                        rename(sinst.x, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_y:
                        rename(sinst.y, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_dest:
                        rename(sinst.dest, virtualId, sregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            ScalarOperationState state(sinst.op,
                                       sinst.permit,
                                       sinst.x,
                                       sinst.y,
                                       sinst.dest);
            inst.SetOperationState(Instruction::SCALARTWO, &state);
            // std::cout << "vreg count: " << sinst.vRegCount << std::endl;
            scalar1 = nullptr;
        }
        if (vector0 != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << vector0->rawInstId << ", ";
            }
            auto &vinst = vector0->vectorInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for(int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag)
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorInst::v_x:
                        rename(vinst.x, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_y:
                        rename(vinst.y, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_dest:
                        rename(vinst.dest, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            VectorOperationState state(vinst.op,
                                       vinst.permit,
                                       vinst.x,
                                       vinst.y,
                                       vinst.dest);
            inst.SetOperationState(Instruction::VECTORONE, &state);
            for (auto &r : vinst.writeResource)
            {
                if (r.isFifo())
                {
                    bool find = false;
                    for (auto &dep : gPtr->edge[vector0])
                    {
                        const auto &ress =
                            gPtr->edgeRes[std::make_tuple(vector0, dep)];
                        if (std::any_of(ress.begin(),
                                        ress.end(),
                                        [r](ResourceScheduler rr)
                                        { return rr == r; }))
                        {
                            for (auto &rr : dep->readResource())
                            {
                                if (rr == r)
                                {
                                    if (find)
                                    {
                                        assert(false && "unreachable!");
                                    }
                                    fifoHolder[r].push_back(dep);
                                    // std::cout << COLOR::RED << "push@"
                                    //           << dep->rawInstId << "\n"
                                    //           << COLOR::WHITE;
                                    find = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // std::cout << "vreg count: " << vinst.vRegCount << std::endl;
            vector0 = nullptr;
        }
        if (vector1 != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << vector1->rawInstId << ", ";
            }
            auto &vinst = vector1->vectorInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for(int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag)
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorInst::v_x:
                        rename(vinst.x, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_y:
                        rename(vinst.y, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_dest:
                        rename(vinst.dest, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            VectorOperationState state(vinst.op,
                                       vinst.permit,
                                       vinst.x,
                                       vinst.y,
                                       vinst.dest);
            inst.SetOperationState(Instruction::VECTORTWO, &state);
            for (auto &r : vinst.writeResource)
            {
                if (r.isFifo())
                {
                    bool find = false;
                    for (auto &dep : gPtr->edge[vector1])
                    {
                        const auto &ress =
                            gPtr->edgeRes[std::make_tuple(vector1, dep)];
                        if (std::any_of(ress.begin(),
                                        ress.end(),
                                        [r](ResourceScheduler rr)
                                        { return rr == r; }))
                        {
                            for (auto &rr : dep->readResource())
                            {
                                if (rr == r)
                                {
                                    if (find)
                                    {
                                        assert(false && "unreachable!");
                                    }
                                    fifoHolder[r].push_back(dep);
                                    // std::cout << COLOR::RED << "push@"
                                    //           << dep->rawInstId << "\n"
                                    //           << COLOR::WHITE;
                                    find = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // std::cout << "vreg count: " << vinst.vRegCount << std::endl;
            vector1 = nullptr;
        }
        if (vectorLoad != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << vectorLoad->rawInstId << ", ";
            }
            auto &vinst = vectorLoad->vectorLoadInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for (int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag) 
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorLoadInst::v_dest:
                    {
                        rename(vinst.dest, virtualId, vregAllocator);
                        break;
                    }
                    case VectorLoadInst::vs_imm_0:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, newRegId);
                        break;
                    }
                    case VectorLoadInst::vs_imm_1:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1, newRegId);
                        break;
                    }
                    case VectorLoadInst::vs_imm_2:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE2, newRegId);
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
            VectorLoadOperationState state(vinst.op,
                                           vinst.permit,
                                           vinst.dest,
                                           vinst.base,
                                           vinst.offset,
                                           vinst.stride,
                                           vinst.ia,
                                           vinst.mask);
            inst.SetOperationState(Instruction::VECTORLOAD, &state);
            // std::cout << "vreg count: " << vinst.vRegCount << std::endl;
            vectorLoad = nullptr;
        }
        if (vectorStore != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << vectorStore->rawInstId << ", ";
            }
            auto &vinst = vectorStore->vectorStoreInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for (int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag) 
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorStoreInst::v_x:
                    {
                        rename(vinst.x, virtualId, vregAllocator);
                        break;
                    }
                    case VectorStoreInst::vs_imm_0:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, newRegId);
                        break;
                    }
                    case VectorStoreInst::vs_imm_1:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1, newRegId);
                        break;
                    }
                    case VectorStoreInst::vs_imm_2:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE2, newRegId);
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
            VectorStoreOperationState state(vinst.op,
                                            vinst.permit,
                                            vinst.x,
                                            vinst.base,
                                            vinst.offset,
                                            vinst.stride,
                                            vinst.ia,
                                            vinst.mask);
            inst.SetOperationState(Instruction::VECTORSTORE, &state);
            // std::cout << "vreg count: " << vinst.vRegCount << std::endl;
            vectorStore = nullptr;
        }
        if (mti != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << mti->rawInstId << ", ";
            }
            auto &minst = mti->mtiInst;
            FillImme(minst.usedImme, inst, *minst.pInst);
            for (int i = 0; i < minst.regTable.size(); i++)
            {
                if (minst.regTable[i].flag)
                {
                    int virtualId = minst.regTable[i].virRegId;
                    switch (i)
                    {
                    case MTIInst::mti_x:
                        rename(minst.x, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            MTIOperationState state(minst.op,
                                    minst.permit,
                                    minst.x,
                                    minst.mask,
                                    minst.select);
            inst.SetOperationState(Instruction::MTI, &state);
            for (auto &r : minst.writeResource)
            {
                if (r.isFifo())
                {
                    bool find = false;
                    for (auto &dep : gPtr->edge[mti])
                    {
                        const auto &ress =
                            gPtr->edgeRes[std::make_tuple(mti, dep)];
                        if (std::any_of(ress.begin(),
                                        ress.end(),
                                        [r](ResourceScheduler rr)
                                        { return rr == r; }))
                        {
                            switch(r.tag) {
                            case Res::GSNF:
                            case Res::GSTF:
                                for (auto &rr : dep->writeResource())
                                {
                                    if (rr == r)
                                    {
                                        if (find)
                                        {
                                            assert(false && "unreachable!");
                                        }
                                        else
                                        {
                                            if (fifoHolder[r].empty()) 
                                            {
                                                fifoHolder[r].push_back(dep);
                                            }
                                            else 
                                            {
                                                fifoHolder[r].front() = dep;
                                            }
                                            // std::cout << COLOR::RED <<
                                            // "push@"
                                            //          << dep->rawInstId <<
                                            //          "\n"
                                            //           << COLOR::WHITE;
                                        }
                                        find = true;
                                        break;
                                    }
                                }
                                for (auto &rr : dep->readResource())
                                {
                                    if (find) {
                                        break;
                                    }
                                    if (rr == r)
                                    {
                                        if (find)
                                        {
                                            assert(false && "unreachable!");
                                        }
                                        else
                                        {
                                            if (fifoHolder[r].empty()) 
                                            {
                                                fifoHolder[r].push_back(dep);
                                            }
                                            else 
                                            {
                                                fifoHolder[r].front() = dep;
                                            }
                                            // std::cout << COLOR::RED <<
                                            // "pop@"
                                            //          << dep->rawInstId <<
                                            //          "\n"
                                            //           << COLOR::WHITE;
                                        }
                                        find = true;
                                        break;
                                    }
                                }
                                break;
                            case Res::MRF:
                                for (auto &rr : dep->readResource())
                                {
                                    if (rr == r)
                                    {
                                        if (find)
                                        {
                                            assert(false && "unreachable!");
                                        }
                                        else
                                        {
                                            fifoHolder[r].push_back(dep);
                                            // std::cout << COLOR::RED <<
                                            // "push@"
                                            //          << dep->rawInstId <<
                                            //          "\n"
                                            //           << COLOR::WHITE;
                                        }
                                        find = true;
                                        break;
                                    }
                                }
                                break;
                            case Res::TRF:
                                for (auto &rr : dep->writeResource())
                                {
                                    if (rr == r)
                                    {
                                        if (find)
                                        {
                                            assert(false && "unreachable!");
                                        }
                                        else
                                        {
                                            if (fifoHolder[r].empty()) 
                                            {
                                                fifoHolder[r].push_back(dep);
                                            }
                                            else 
                                            {
                                                fifoHolder[r].front() = dep;
                                            }
                                            // std::cout << COLOR::RED <<
                                            // "push@"
                                            //          << dep->rawInstId <<
                                            //          "\n"
                                            //           << COLOR::WHITE;
                                        }
                                        find = true;
                                        break;
                                    }
                                }
                                for (auto &rr : dep->readResource())
                                {
                                    if (find) break;
                                    if (rr == r)
                                    {
                                        if (find)
                                        {
                                            assert(false && "unreachable!");
                                        }
                                        else
                                        {
                                            if (fifoHolder[r].empty()) 
                                            {
                                                fifoHolder[r].push_back(dep);
                                            }
                                            else 
                                            {
                                                fifoHolder[r].front() = dep;
                                            }
                                            // std::cout << COLOR::RED <<
                                            // "push@"
                                            //          << dep->rawInstId <<
                                            //          "\n"
                                            //           << COLOR::WHITE;
                                        }
                                        find = true;
                                        break;
                                    }
                                }
                                break;
                            default: 
                                break;
                            }
                        }
                    }
                    if (!find) 
                    {
                        fifoHolder[r].pop_front();
                    }
                }
                if (r.isGMR()) {
                    assert(GMRAvail[r.regId] && "unreachable");
                    GMRAvail[r.regId] = false;
                    bool findMul = false;
                    for (auto &dep : gPtr->edge[mti])
                    {
                        for (auto &rr : dep->readResource())
                        {
                            if (rr == r)
                            {
                                GMRreaders[r.regId].insert(dep);
                                findMul = true;
                            }
                        }
                    }
                    if (!findMul)
                    {
                        GMRAvail[r.regId] = true;
                    }
                }
            }
            for (auto &r : minst.readResource)
            {
                switch (r.tag)
                {
                case Res::GSNF:
                case Res::GSTF:
                    fifoHolder[r].clear();
                    break;
                case Res::GMR:
                    if (!GMRreaders[r.regId].empty()) 
                    {
                        GMRreaders[r.regId].erase(mti);
                        if (GMRreaders[r.regId].empty()) 
                        {
                            GMRAvail[r.regId] = true;
                        }
                    }
                    break;
                default:
                    break;
                }
            }
            // std::cout << "vreg count: " << minst.vRegCount << std::endl;
            mti = nullptr;
        }
        if (mtr != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "@" << mtr->rawInstId << ", ";
            }
            auto &minst = mtr->mtrInst;
            FillImme(minst.usedImme, inst, *minst.pInst);
            for (int i = 0; i < minst.regTable.size(); i++)
            {
                if (minst.regTable[i].flag)
                {
                    int virtualId = minst.regTable[i].virRegId;
                    switch (i)
                    {
                    case MTRInst::v_dest:
                        rename(minst.dest, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            MTROperationState state(minst.op,
                                    minst.permit,
                                    minst.dest,
                                    minst.select);
            inst.SetOperationState(Instruction::MTR, &state);
            for (auto &r : minst.readResource)
            {
                if (r.isFifo())
                {
                    if (!fifoHolder[r].empty() && fifoHolder[r].front() == mtr)
                    {
                        bool find = false;
                        for (auto &dep : gPtr->edge[mtr])
                        {
                            const auto &ress =
                                gPtr->edgeRes[std::make_tuple(mtr, dep)];
                            if (std::any_of(ress.begin(),
                                            ress.end(),
                                            [r](ResourceScheduler rr)
                                            { return rr == r; }))
                            {
                                for (auto &rr : dep->readResource())
                                {
                                    if (rr == r)
                                    {
                                        if (find)
                                        {
                                            assert(false && "unreachable!");
                                        }
                                        find = true;
                                        // std::cout
                                        //     << COLOR::RED << "front@"
                                        //     <<
                                        //     fifoHolder[r].front()->rawInstId
                                        //     << "->@" << dep->rawInstId <<
                                        //     "\n"
                                        //     << COLOR::WHITE;
                                        fifoHolder[r].front() = dep;
                                        break;
                                    }
                                }
                            }
                        }
                        if (!find)
                        {
                            auto who = fifoHolder[r].front();
                            fifoHolder[r].pop_front();
                            // std::cout << COLOR::RED << "pop@" <<
                            // who->rawInstId
                            //           << "\n"
                            //           << COLOR::WHITE;
                        }
                    }
                }
            }
            // std::cout << "vreg count: " << minst.vRegCount << std::endl;
            mtr = nullptr;
        }
        if (misc != nullptr)
        {
            if (coutInfo)
            {
                // std::cout << "misc: " << std::endl;
                // std::cout << "@" << misc->rawInstId << ", ";
            }
            auto &minst = misc->miscInst;
            FillImme(minst.usedImme, inst, *minst.pInst);
            MiscOperationState state(minst.op,
                                     minst.permit,
                                     minst.operand,
                                     minst.cond,
                                     minst.target);
            inst.SetOperationState(Instruction::MISC, &state);
            misc = nullptr;
        }
        __CompleteInstruction(&inst);
        if (coutInfo)
        {
            inst.ToAssembly();
        }
        usedImme = 0;
        return pInst;
    }
};

std::vector<Instruction *>
fuse(DAG g, bool coutInfo = false)
{
    std::vector<Instruction *> res;
    InstructionBuilder builder;
    builder.gPtr = &g;
    DAGHeight.clear();
    g.getHeight();
    g.reSortAvailableNodes();
    // for (auto &h : DAGHeight)
    // {
    //     std::cout << "Inst ID: " << h.first->rawInstId
    //               << " height: " << h.second << std::endl;
    // }
    // for (auto &ref : vVRegRefCount)
    // {
    //     std::cout << "serialId Reg ID: " << ref.first
    //               << " ref count: " << ref.second << std::endl;
    // }
    const auto peekS0 = builder.peekScalar0();
    const auto peekS1 = builder.peekScalar1();
    const auto peekV0 = builder.peekVector0();
    const auto peekV1 = builder.peekVector1();
    const auto peekVL = builder.peekVectorLoad();
    const auto peekVS = builder.peekVectorStore();
    const auto peekMTI = builder.peekMTI();
    const auto peekMTR = builder.peekMTR();
    const auto peekMISC = builder.peekMISC();
    while (!g.empty())
    {
        g.peek(peekS0);
        g.peek(peekS1);
        g.resolvePeekedStage1();
        g.peek(peekV0);
        g.peek(peekV1);
        g.resolvePeekedStage1();
        g.peek(peekVS);
        g.peek(peekVL);
        g.peek(peekMTI);
        g.peek(peekMTR);
        g.peek(peekMISC);
        g.resolvePeekedStage1();
        if (coutInfo || builder.empty())
        {
            // std::cout << "sreg map size: " << sregAllocBase.size() << std::endl;
            // std::cout << "vreg map size: " << vregAllocBase.size() << std::endl;
            // std::cout << "avail: ";
            // for (auto &n : g.availableNodes)
            // {
            //     std::cout << "@" << n->rawInstId << ", ";
            // }
            // std::cout << "\n";
            // builder.fifoHolder.printInfo();
        }
        res.push_back(builder.build(coutInfo));
        g.resolvePeekedStage2();
    }
    return res;
}

InstHint
StaticAnalyze(Instruction &inst, const InstHint &old)
{
    InstHint next = old;
    auto sinst = reinterpret_cast<ScalarOperationState *>(
        inst.GetOperation(Instruction::SCALARONE));
    if (sinst != nullptr && sinst->GetOpCode() == S_U32_MOVE)
    {
        auto dest = sinst->GetIndexDest();
        auto sy = sinst->GetIndexY();
        if (sy >= 32 && sy <= 62)
        {
            next.sregValue[dest] = std::make_pair(true, GetImmeVal(inst, sy));
        }
        else
        {
            next.sregValue[dest].first = false;
        }
    }
    return next;
}

std::vector<Instruction *>
scheduleSingle(const std::vector<Instruction *> &bundle, bool coutInfo)
{
    if (bundle.empty())
    {
        return std::vector<Instruction *>{};
    }
    if (bundle.size() == 1)
    {
        std::vector<Instruction *> copy;
        copy.push_back(new Instruction(*bundle[0]));
        return copy;
    }
    std::vector<Inst233> insts;
    InstHint hint;
    int idCount = 0;
    for (int i = 0; i < bundle.size(); i++)
    {
        if (coutInfo)
        {
            std::cout << "I" << i << " [label=\""
                      << "I" << i << ": ";
            auto assm = HookCOut([&i, &bundle]() { bundle[i]->ToAssembly(); });
            assm.erase(assm.end() - 1);
            std::cout << assm << "\"]\n";
        }
        hint = StaticAnalyze(*bundle[i], hint);
        for (auto &j : read(*bundle[i], hint))
        {
            j.rawInstId = i;
            j.rawInst = bundle[i];
            if (!j.isNoop())
            {
                j.InstId = idCount++;
                j.checkResource();
                insts.emplace_back(j);
            }
        }
    }
    vSRegRefCount.clear();
    vVRegRefCount.clear();
    DAG g = buildDependency(insts);
    std::string dotGraph = Visualization::generateDotGraph(g, "Scheduler");
    Visualization::generateDotFile(dotGraph, "Scheduler.dot");
    if (coutInfo)
    {
        for (auto &n : g.nodes)
        {
            if (g.edge[n].empty())
            {
                continue;
            }
            // std::cout << "I" << n->rawInstId << " -> ";
            bool first = true;
            for (auto &t : g.edge[n])
            {
                // if (!first)
                // {
                //     std::cout << ", ";
                // }
                // std::cout << "I" << t->rawInstId;
                first = false;
            }
            // std::cout << "\n";
        }
    }
    
    auto res = fuse(g, coutInfo); 
    
    ConstrainScheduler::fuse(g, res.size() + 1, coutInfo);

    return res;
}

std::vector<int>
specificInstruction(Instruction *inst)
{
    auto s0op = inst->GetOperation(Instruction::SCALARONE)->GetOpCode();
    auto s1op = inst->GetOperation(Instruction::SCALARTWO)->GetOpCode();
    auto miscop = inst->GetOperation(Instruction::MISC)->GetOpCode();
    // std::cout << "opcode: " << inst->GetOperation(Instruction::MTI)->GetOpCode() << std::endl;
    switch (s0op)
    {
    case S_HALT:
    case S_DELAY:
    case S_FENCE:
    case S_LOCAL_DMA:
    case S_STRIDED_DMA:
        return {0, 1};
    case S_BRANCH:
    {
        assert(false && "branch is unsupported");
        auto op = static_cast<ScalarOperationState *>(
            inst->GetOperation(Instruction::SCALARONE));
        assert((op->GetIndexDest() == 0 || op->GetIndexDest() == 1) &&
               "b-abs, b-reg is unsupported");
        auto imm0 = (int16_t)inst->GetImmediateValue(Instruction::IMMEDIATE0);
        return {0, 1, imm0 - 1};
    }
    case S_CALL_ABOSOLUTE:
    case S_CALL_REGISTER:
    case S_CALL_RELATIVE:
        assert(false && "call is unsupported");
        break;
    default:;
    }
    switch (s1op)
    {
    case S_HALT:
    case S_DELAY:
    case S_FENCE:
    case S_DMA:
        return {0, 1};
    default:;
    }
    switch (miscop)
    {
    //case MISC_SET_SYNC_FLAG:
    case MISC_SYNC_FLAG_INCREMENT:
    case MISC_SYNC:
    case MISC_READ_SYNC_FLAG:
    case MISC_INTERRUPT:
    case MISC_VECTOR_DELAY_SHORT:
    case MISC_VECTOR_DELAY_LONG:
    case MISC_REMOTE_SET_SYNC_FLAG:
    case MISC_REMOTE_SYNC_FLAG_INCREMENT:
    case MISC_TRACE:
    case MISC_SET_TRACEMARK:
    case MISC_CFENCE:
        return {1, 0};
    default:;
    }
    return {};
}
} // namespace InstScheduler

std::vector<Instruction *>
schedule(const std::vector<Instruction *> &bundle, bool coutInfo)
{
    using namespace InstScheduler;

    std::set<size_t> splitPlaces;
    for (size_t i = 0; i < bundle.size(); i++)
    {
        auto s = specificInstruction(bundle[i]);
        for (auto p : s)
        {
            splitPlaces.insert(i + p); // some place need split
        }
    }
    size_t lp = 0;
    std::vector<Instruction *> ret;
    splitPlaces.insert(bundle.size());
    for (auto &p : splitPlaces)
    {
        // std::cout << "分割: " << p << std::endl;
        auto res =
            scheduleSingle({bundle.begin() + lp, bundle.begin() + p}, coutInfo);
        std::move(res.begin(), res.end(), std::back_inserter(ret));
        lp = p;
    }
    bool inTrans = false;
    for (auto &r : ret)
    {
        if (r->GetOperationState(Instruction::MTI) != nullptr)
        {
            const auto opcode =
                r->GetOperationState(Instruction::MTI)->GetOpCode();
            switch (opcode)
            {
            case MTI_TRANSPOSE_START:
            case MTI_TRANSPOSE_SEGMENT_START:
            case MTI_TRANSPOSE_PACKED_START:
            case MTI_TRANSPOSE_PACKED_SEGMENT_START:
                assert(!inTrans);
                inTrans = true;
                break;
            case MTI_TRANSPOSE:
            case MTI_TRANSPOSE_SEGMENT:
            case MTI_TRANSPOSE_PACKED:
            case MTI_TRANSPOSE_PACKED_SEGMENT:
                inTrans = true;
                break;
            case MTI_TRANSPOSE_END:
            case MTI_TRANSPOSE_SEGMENT_END:
            case MTI_TRANSPOSE_PACKED_END:
            case MTI_TRANSPOSE_PACKED_SEGMENT_END:
                inTrans = false;
                break;

            case MTI_PERMUTE:
            case MTI_PERMUTE_PACKED:
            case MTI_SET_PERMUTE:
            case MTI_SET_PERMUTE_SUBLANES:
            case MTI_SET_PERMUTE_BYTE:
            case MTI_SET_SPR:
            case MTI_REDUCTION_V_SUM:
            case MTI_REDUCTION_V_MAX:
            case MTI_REDUCTION_V_MIN:
            case MTI_REDUCTION_V_MAX_INDEX:
            case MTI_REDUCTION_V_MIN_INDEX:
            case MTI_REDUCTION_SEGMENTED_V_SUM:
            case MTI_REDUCTION_SEGMENTED_V_MAX:
            case MTI_REDUCTION_SEGMENTED_V_MIN:
            case MTI_REDUCTION_SEGMENTED_V_MAX_INDEX:
            case MTI_REDUCTION_SEGMENTED_V_MIN_INDEX:
            case MTI_REDUCTION_PACKED_V_SUM:
            case MTI_REDUCTION_PACKED_V_MAX:
            case MTI_REDUCTION_PACKED_V_MIN:
            case MTI_REDUCTION_PACKED_V_MAX_INDEX:
            case MTI_REDUCTION_PACKED_V_MIN_INDEX:
            case MTI_REDUCTION_PACKED_SEGMENTED_V_SUM:
            case MTI_REDUCTION_PACKED_SEGMENTED_V_MAX:
            case MTI_REDUCTION_PACKED_SEGMENTED_V_MIN:
            case MTI_REDUCTION_PACKED_SEGMENTED_V_MAX_INDEX:
            case MTI_REDUCTION_PACKED_SEGMENTED_V_MIN_INDEX:
            case MTI_ROTATE:
            case MTI_PACKED_ROTATE:
                assert(!inTrans);
                break;
            }
        }
    }
    return ret;
}

template struct Thief<ThiefHandle<Instruction, std::vector<bool>>,
                      &Instruction::_immediate_states>;

std::string
InstructionExports::NormalCppConstruction(Instruction *inst)
{
    std::stringstream out;
    out << "{\nInstruction* inst = new Instruction();" << std::endl;
    const auto &immeState =
        inst->*get(ThiefHandle<Instruction, std::vector<bool>>());
    const static std::string immeName[] = {
        "IMMEDIATE0",
        "IMMEDIATE1",
        "IMMEDIATE2",
        "IMMEDIATE3",
        "IMMEDIATE4",
        "IMMEDIATE5",
        "VECTORSCALARIMMEDIATE0",
        "VECTORSCALARIMMEDIATE1",
        "VECTORSCALARIMMEDIATE2",
    };
    const static std::unordered_map<int, std::string> opName = {
        {0, "S_NOOP"},
        {1, "S_HALT"},
        {2, "S_POP"},
        {3, "S_DELAY"},
        {4, "S_SMEM_LOAD"},
        {5, "S_SMEM_LOAD_OFFSET"},
        {6, "S_SMEM_STORE"},
        {7, "S_SET"},
        {8, "S_BRANCH"},
        {9, "S_CALL_ABOSOLUTE"},
        {10, "S_CALL_RELATIVE"},
        {11, "S_CALL_REGISTER"},
        {13, "S_FENCE"},
        {14, "S_DMA"},
        {15, "S_LOCAL_DMA"},
        {16, "S_STRIDED_DMA"},
        {17, "S_READ"},
        {18, "S_CONVERT_S32_TO_F32"},
        {19, "S_CONVERT_F32_TO_S32"},
        {20, "S_S32_ADDITION"},
        {21, "S_S32_SUBTRACTION"},
        {22, "S_U32_AND"},
        {23, "S_U32_OR"},
        {24, "S_U32_XOR"},
        {25, "S_U32_SHIFTLEFT"},
        {26, "S_U32_SHIFTRIGHT"},
        {27, "S_U32_MOVE"},
        {28, "S_U32_COUNTLEADINGZEROES"},
        {29, "S_U32_MULTIPLICATION"},
        {30, "S_F32_ADDITION"},
        {31, "S_F32_SUBTRACTION"},
        {32, "S_F32_MULTIPLICATION"},
        {33, "S_F32_MAX"},
        {34, "S_F32_MIN"},
        {35, "S_S32_EQUAL"},
        {36, "S_S32_NOTEQUAL"},
        {37, "S_S32_GREATER"},
        {38, "S_S32_GREATEREQUAL"},
        {39, "S_S32_LESSER"},
        {40, "S_S32_LESSER_EQUAL"},
        {41, "S_U32_CARRY"},
        {42, "S_F32_EQUAL"},
        {43, "S_F32_NOTEQUAL"},
        {44, "S_F32_GREATER"},
        {45, "S_F32_GREATEREQUAL"},
        {46, "S_F32_LESSER"},
        {47, "S_F32_LESSEREQUAL"},
        {48, "S_F32_IS_INF_OR_NAN"},
        {49, "S_PERMISSION_OR"},
        {50, "S_ARITHMETIC_SHIFT_RIGHT"},
        {51, "MAX_NUMBER_OF_SCALAR_INSTRUCTION"},
        {100, "V_NOOP"},
        {101, "V_HALF_FLOAT_PACK"},
        {102, "V_TWO_LOWER_INT8_PACK"},
        {103, "V_FOUR_INT8_PACK"},
        {104, "V_SUBCORE_ROTATE"},
        {106, "V_RNG_GENERATE_RANDOM_NUMBER"},
        {107, "V_RNG_READ_SEED"},
        {108, "V_RNG_RESEED"},
        {109, "V_F32_SOFTSIGN"},
        {110, "V_F32_LOG2"},
        {111, "V_F32_SIGMOID"},
        {112, "V_F32_RECIPROCAL"},
        {113, "V_F32_SQUAREROOT_RECIPROCAL"},
        {114, "V_F32_POWER"},
        {115, "V_F32_SOFTPLUS"},
        {116, "V_F32_EXPONENT"},
        {118, "V_CONVERT_S32_TO_F32"},
        {119, "V_CONVERT_F32_TO_S32"},
        {120, "V_S32_ADDITION"},
        {121, "V_S32_SUBTRACTION"},
        {122, "V_U32_AND"},
        {123, "V_U32_OR"},
        {124, "V_U32_XOR"},
        {125, "V_U32_SHIFTLEFT"},
        {126, "V_U32_SHIFTRIGHT"},
        {127, "V_U32_MOVE"},
        {128, "V_U32_COUNTLEADINGZEROES"},
        {129, "V_U32_MULTIPLICATION"},
        {130, "V_F32_ADDITION"},
        {131, "V_F32_SUBTRACTION"},
        {132, "V_F32_MULTIPLICATION"},
        {133, "V_F32_MAX"},
        {134, "V_F32_MIN"},
        {135, "V_S32_EQUAL"},
        {136, "V_S32_NOTEQUAL"},
        {137, "V_S32_GREATER"},
        {138, "V_S32_GREATEREQUAL"},
        {139, "V_S32_LESSER"},
        {140, "V_S32_LESSER_EQUAL"},
        {141, "V_U32_CARRY"},
        {142, "V_F32_EQUAL"},
        {143, "V_F32_NOTEQUAL"},
        {144, "V_F32_GREATER"},
        {145, "V_F32_GREATEREQUAL"},
        {146, "V_F32_LESSER"},
        {147, "V_F32_LESSEREQUAL"},
        {148, "V_F32_IS_INF_OR_NAN"},
        {149, "V_PERMISSION_OR"},
        {150, "V_ARITHMETIC_SHIFT_RIGHT"},
        {151, "V_ROUND_ARITHMETIC_SHIFT_RIGHT"},
        {152, "V_SELECT_VMASK0"},
        {153, "V_SELECT_VMASK1"},
        {154, "V_SELECT_VMASK2"},
        {155, "V_SELECT_VMASK3"},
        {156, "V_SELECT_VMASK4"},
        {157, "V_SELECT_VMASK5"},
        {158, "V_SELECT_VMASK6"},
        {159, "V_SELECT_VMASK7"},
        {160, "V_GET_V_CORE_ID"},
        {161, "V_SET_VMASK"},
        {162, "V_EXTRACT"},
        {163, "V_COMPOSE_FLOAT"},
        {164, "V_COUNT_NUMBER_OF_ONE"},
        {165, "V_RELUX"},
        {166, "V_CLAMP"},
        {167, "MAX_NUMBER_OF_V_INSTRUCTION"},
        {200, "V_STORE_NOOP"},
        {201, "V_STORE"},
        {202, "V_STORE_WITH_OFFSET"},
        {203, "V_STORE_WITH_VMASK0"},
        {204, "V_STORE_WITH_VMASK1"},
        {205, "V_STORE_WITH_VMASK2"},
        {206, "V_STORE_WITH_VMASK3"},
        {207, "V_STORE_WITH_VMASK4"},
        {208, "V_STORE_WITH_VMASK5"},
        {209, "V_STORE_WITH_VMASK6"},
        {210, "V_STORE_WITH_VMASK7"},
        {211, "V_STORE_INDEXED"},
        {212, "V_STORE_INDEXED_WITH_OFFSET"},
        {213, "V_STORE_INDEXED_WITH_VMASK0"},
        {214, "V_STORE_INDEXED_WITH_VMASK1"},
        {215, "V_STORE_INDEXED_WITH_VMASK2"},
        {216, "V_STORE_INDEXED_WITH_VMASK3"},
        {217, "V_STORE_INDEXED_WITH_VMASK4"},
        {218, "V_STORE_INDEXED_WITH_VMASK5"},
        {219, "V_STORE_INDEXED_WITH_VMASK6"},
        {220, "V_STORE_INDEXED_WITH_VMASK7"},
        {221, "V_STORE_SET_IA_OF_CORE"},
        {222, "V_STORE_SET_IA_OF_SUBCORE"},
        {223, "V_STORE_PUSH_TO_SCALAR_CORE"},
        {224, "V_STORE_FXC"},
        {225, "MAX_NUMBER_OF_V_STORE_INSTRUCTION"},
        {300, "V_LOAD_NOOP"},
        {301, "V_LOAD"},
        {302, "V_LOAD_WITH_OFFSET"},
        {303, "V_LOAD_WITH_VMASK0"},
        {304, "V_LOAD_WITH_VMASK1"},
        {305, "V_LOAD_WITH_VMASK2"},
        {306, "V_LOAD_WITH_VMASK3"},
        {307, "V_LOAD_WITH_VMASK4"},
        {308, "V_LOAD_WITH_VMASK5"},
        {309, "V_LOAD_WITH_VMASK6"},
        {310, "V_LOAD_WITH_VMASK7"},
        {311, "V_LOAD_INDEXED"},
        {312, "V_LOAD_INDEXED_WITH_OFFSET"},
        {313, "V_LOAD_INDEXED_WITH_VMASK0"},
        {314, "V_LOAD_INDEXED_WITH_VMASK1"},
        {315, "V_LOAD_INDEXED_WITH_VMASK2"},
        {316, "V_LOAD_INDEXED_WITH_VMASK3"},
        {317, "V_LOAD_INDEXED_WITH_VMASK4"},
        {318, "V_LOAD_INDEXED_WITH_VMASK5"},
        {319, "V_LOAD_INDEXED_WITH_VMASK6"},
        {320, "V_LOAD_INDEXED_WITH_VMASK7"},
        {321, "V_LOAD_WITH_SHUFFLE"},
        {322, "V_LOAD_INDEXED_WITH_SHUFFLE"},
        {323, "V_LOAD_FXC"},
        {324, "MAX_NUMBER_OF_V_LOAD_INSTRUCTION"},
        {400, "MTI_NOOP"},
        {401, "MTI_MUL_FLOAT_ROUNDED"},
        {402, "MTI_MUL_HIGHER_F16"},
        {403, "MTI_MUL_LOWER_F16"},
        {404, "MTI_MUL_F16_PACKED"},
        {405, "MTI_MUL_INT8_PACKED"},
        {406, "MTI_MUL_INT8_LOWER16_PACKED"},
        {407, "MTI_MUL_GSNF_ROUNDED"},
        {408, "MTI_MUL_GSNF_HIGHER16"},
        {409, "MTI_MUL_GSNF_LOWER16"},
        {410, "MTI_MUL_GSNF_PACKED_F16"},
        {411, "MTI_MUL_GSNF_PACKED_INT8"},
        {412, "MTI_MUL_GSNF_PACKED_INT8_LOWER16"},
        {413, "MTI_MUL_GSTF_ROUNDED"},
        {414, "MTI_MUL_GSTF_HIGHER16"},
        {415, "MTI_MUL_GSTF_LOWER16"},
        {416, "MTI_MUL_GSTF_PACKED_F16"},
        {417, "MTI_MUL_GSTF_PACKED_INT8"},
        {418, "MTI_MUL_GSTF_PACKED_INT8_LOWER16"},
        {419, "MTI_MUL_MASK_ROUNDED"},
        {420, "MTI_MUL_MASK_HIGER16"},
        {421, "MTI_MUL_MASK_LOWER16"},
        {422, "MTI_MUL_MASK_PACKED_F16"},
        {423, "MTI_MUL_MASK_PACKED_INT8"},
        {424, "MTI_MUL_MASK_PACKED_INT8_LOWER16"},
        {425, "MTI_MUL_MASK_GSNF_ROUNDED"},
        {426, "MTI_MUL_MASK_GSNF_HIGER16"},
        {427, "MTI_MUL_MASK_GSNF_LOWER16"},
        {428, "MTI_MUL_MASK_GSNF_PACKED_F16"},
        {429, "MTI_MUL_MASK_GSNF_PACKED_INT8"},
        {430, "MTI_MUL_MASK_GSNF_PACKED_INT8_LOWER16"},
        {431, "MTI_MUL_MASK_GSTF_ROUNDED"},
        {432, "MTI_MUL_MASK_GSTF_HIGER16"},
        {433, "MTI_MUL_MASK_GSTF_LOWER16"},
        {434, "MTI_MUL_MASK_GSTF_PACKED_F16"},
        {435, "MTI_MUL_MASK_GSTF_PACKED_INT8"},
        {436, "MTI_MUL_MASK_GSTF_PACKED_INT8_LOWER16"},
        {437, "MTI_LOAD_GSNF"},
        {438, "MTI_LOAD_GSTF"},
        {439, "MTI_PUSHGAIN_FLOAT_ROUNDED"},
        {440, "MTI_PUSHGAIN_HIGHER_F16"},
        {441, "MTI_PUSHGAIN_LOWER_F16"},
        {442, "MTI_PUSHGAIN_PACKED_INT8"},
        {444, "MTI_PUSHGAIN_TRANSPOSE_ROUND"},
        {445, "MTI_PUSHGAIN_TRANSPOSE_HIGHER_F16"},
        {446, "MTI_PUSHGAIN_TRANSPOSE_LOWER_F16"},
        {447, "MTI_PUSHGAIN_TRANSPOSE_PACKED_INT8"},
        {449, "MTI_PUSHGAIN_MASK_ROUNDED"},
        {450, "MTI_PUSHGAIN_MASK_HIGHER16"},
        {451, "MTI_PUSHGAIN_MASK_LOWER16"},
        {452, "MTI_PUSHGAIN_MASK_PACKED_INT8"},
        {454, "MTI_PUSHGAIN_MASK_TRANSPOSE_ROUND"},
        {455, "MTI_PUSHGAIN_MASK_TRANSPOSE_HIGHER16"},
        {456, "MTI_PUSHGAIN_MASK_TRANSPOSE_LOWER16"},
        {457, "MTI_PUSHGAIN_MASK_PACKED_TRANSPOSE_INT16"},
        {459, "MTI_TRANSPOSE_START"},
        {460, "MTI_TRANSPOSE"},
        {461, "MTI_TRANSPOSE_END"},
        {462, "MTI_TRANSPOSE_START_END"},
        {463, "MTI_TRANSPOSE_SEGMENT_START"},
        {464, "MTI_TRANSPOSE_SEGMENT"},
        {465, "MTI_TRANSPOSE_SEGMENT_END"},
        {466, "MTI_TRANSPOSE_SEGMENT_START_END"},
        {467, "MTI_TRANSPOSE_PACKED_START"},
        {468, "MTI_TRANSPOSE_PACKED"},
        {469, "MTI_TRANSPOSE_PACKED_END"},
        {470, "MTI_TRANSPOSE_PACKED_START_END"},
        {471, "MTI_TRANSPOSE_PACKED_SEGMENT_START"},
        {472, "MTI_TRANSPOSE_PACKED_SEGMENT"},
        {473, "MTI_TRANSPOSE_PACKED_SEGMENT_END"},
        {474, "MTI_TRANSPOSE_PACKED_SEGMENT_START_END"},
        {475, "MTI_PERMUTE"},
        {476, "MTI_PERMUTE_PACKED"},
        {477, "MTI_SET_PERMUTE"},
        {478, "MTI_SET_PERMUTE_SUBLANES"},
        {479, "MTI_SET_PERMUTE_BYTE"},
        {480, "MTI_SET_SPR"},
        {481, "MTI_REDUCTION_V_SUM"},
        {482, "MTI_REDUCTION_V_MAX"},
        {483, "MTI_REDUCTION_V_MIN"},
        {484, "MTI_REDUCTION_V_MAX_INDEX"},
        {485, "MTI_REDUCTION_V_MIN_INDEX"},
        {486, "MTI_REDUCTION_SEGMENTED_V_SUM"},
        {487, "MTI_REDUCTION_SEGMENTED_V_MAX"},
        {488, "MTI_REDUCTION_SEGMENTED_V_MIN"},
        {489, "MTI_REDUCTION_SEGMENTED_V_MAX_INDEX"},
        {490, "MTI_REDUCTION_SEGMENTED_V_MIN_INDEX"},
        {491, "MTI_REDUCTION_PACKED_V_SUM"},
        {492, "MTI_REDUCTION_PACKED_V_MAX"},
        {493, "MTI_REDUCTION_PACKED_V_MIN"},
        {494, "MTI_REDUCTION_PACKED_V_MAX_INDEX"},
        {495, "MTI_REDUCTION_PACKED_V_MIN_INDEX"},
        {496, "MTI_REDUCTION_PACKED_SEGMENTED_V_SUM"},
        {497, "MTI_REDUCTION_PACKED_SEGMENTED_V_MAX"},
        {498, "MTI_REDUCTION_PACKED_SEGMENTED_V_MIN"},
        {499, "MTI_REDUCTION_PACKED_SEGMENTED_V_MAX_INDEX"},
        {500, "MTI_REDUCTION_PACKED_SEGMENTED_V_MIN_INDEX"},
        {501, "MTI_ROTATE"},
        {502, "MTI_PACKED_ROTATE"},
        {503, "MAX_NUMBER_OF_MTI_INSTRUCTION"},
        {600, "MTR_NOOP"},
        {601, "MTR_READ_MATRIX_RESULT"},
        {602, "MTR_READ_TRANSPOSE_RESULT"},
        {603, "MTR_READ_UNARY_EXECUTION_RESULT"},
        {604, "MTR_READ_FUXI_CORD_RESULT"},
        {605, "MAX_NUMBER_OF_MTR_INSTRUCTION"},
        {700, "MISC_NOOP"},
        {701, "MISC_SET_SYNC_FLAG"},
        {702, "MISC_SYNC_FLAG_INCREMENT"},
        {703, "MISC_SYNC"},
        {704, "MISC_VMASK_OPERATION"},
        {705, "MISC_READ_SYNC_FLAG"},
        {706, "MISC_INTERRUPT"},
        {707, "MISC_CLEAR_RESULT_FIFO"},
        {708, "MISC_VECTOR_DELAY_SHORT"},
        {709, "MISC_VECTOR_DELAY_LONG"},
        {710, "MISC_REMOTE_SET_SYNC_FLAG"},
        {711, "MISC_REMOTE_SYNC_FLAG_INCREMENT"},
        {712, "MISC_TRACE"},
        {713, "MISC_SET_TRACEMARK"},
        {714, "MISC_CFENCE"},
    };
    for (int i = 0; i < immeState.size(); i++)
    {
        if (!immeState[i])
        {
            out << "inst->SetImmediateValue(Instruction::" << immeName[i]
                << ", "
                << inst->GetImmediateValue(
                       static_cast<Instruction::ImmediateValueType>(i))
                << ");\n";
        }
    }
    auto s0 = inst->GetOperation(Instruction::SCALARONE);
    if (s0 != nullptr && s0->GetOpCode() != S_NOOP)
    {
        auto opcode = s0->GetOpCode();
        auto ss0 = reinterpret_cast<ScalarOperationState *>(s0);
        if (opcode == S_LOCAL_DMA)
        {
            out << "ScalarOperationState s0(" << opName.at(opcode) << ", "
                << s0->GetPermissionValue() << ", " << ss0->GetIndexX() << ", "
                << ss0->GetIndexY() << ", " << ss0->GetIndexX1() << ", "
                << ss0->GetIndexY1() << ", " << ss0->GetIndexMisc() << ");\n"
                << "inst->SetOperationState(Instruction::SCALARONE, &s0);\n";
        }
        else if (opcode == S_STRIDED_DMA)
        {
            out << "ScalarOperationState s0(" << opName.at(opcode) << ", "
                << s0->GetPermissionValue() << ", " << ss0->GetIndexX() << ", "
                << ss0->GetIndexY() << ", " << ss0->GetIndexX1() << ", "
                << ss0->GetIndexY1() << ", " << ss0->GetIndexMisc() << ", "
                << ss0->GetVS0() << ", " << ss0->GetVS1() << ", "
                /*<< ss0->GetVS2()*/ << ");\n"
                << "inst->SetOperationState(Instruction::SCALARONE, &s0);\n";
        }
        else
        {
            out << "ScalarOperationState s0(" << opName.at(opcode) << ", "
                << s0->GetPermissionValue() << ", " << ss0->GetIndexX() << ", "
                << ss0->GetIndexY() << ", " << ss0->GetIndexDest() << ");\n"
                << "inst->SetOperationState(Instruction::SCALARONE, &s0);\n";
        }
    }
    auto s1 = inst->GetOperation(Instruction::SCALARTWO);
    if ((s0 != nullptr || (s0->GetOpCode() != S_LOCAL_DMA &&
                           s0->GetOpCode() != S_STRIDED_DMA)) &&
        s1 != nullptr && s1->GetOpCode() != S_NOOP)
    {
        auto ss1 = reinterpret_cast<ScalarOperationState *>(s1);
        auto opcode = s1->GetOpCode();
        out << "ScalarOperationState s1(" << opName.at(opcode) << ", "
            << s1->GetPermissionValue() << ", " << ss1->GetIndexX() << ", "
            << ss1->GetIndexY() << ", " << ss1->GetIndexDest() << ");\n"
            << "inst->SetOperationState(Instruction::SCALARTWO, &s1);\n";
    }
    auto v0 = inst->GetOperation(Instruction::VECTORONE);
    if (v0 != nullptr && v0->GetOpCode() != V_NOOP)
    {
        auto vv0 = reinterpret_cast<VectorOperationState *>(v0);
        auto opcode = v0->GetOpCode();
        out << "VectorOperationState v0(" << opName.at(opcode) << ", "
            << vv0->GetPermissionValue() << ", " << vv0->GetIndexX() << ", "
            << vv0->GetIndexY() << ", " << vv0->GetIndexDest() << ");\n"
            << "inst->SetOperationState(Instruction::VECTORONE, &v0);\n";
    }
    auto v1 = inst->GetOperation(Instruction::VECTORTWO);
    if (v1 != nullptr && v1->GetOpCode() != V_NOOP)
    {
        auto vv1 = reinterpret_cast<VectorOperationState *>(v1);
        auto opcode = v1->GetOpCode();
        out << "VectorOperationState v1(" << opName.at(opcode) << ", "
            << vv1->GetPermissionValue() << ", " << vv1->GetIndexX() << ", "
            << vv1->GetIndexY() << ", " << vv1->GetIndexDest() << ");\n"
            << "inst->SetOperationState(Instruction::VECTORTWO, &v1);\n";
    }
    auto st = inst->GetOperation(Instruction::VECTORSTORE);
    if (st != nullptr && st->GetOpCode() != V_STORE_NOOP)
    {
        auto stst = reinterpret_cast<VectorStoreOperationState *>(st);
        auto opcode = st->GetOpCode();
        out << "VectorStoreOperationState st(" << opName.at(opcode) << ", "
            << stst->GetPermissionValue() << ", " << stst->GetIndexX() << ", "
            << stst->GetBase() << ", " << stst->GetOffset() << ", "
            << stst->GetStride() << ", " << stst->GetIA() << ", "
            << stst->GetMask() << ");\n"
            << "inst->SetOperationState(Instruction::VECTORSTORE, &st);\n";
    }
    auto ld = inst->GetOperation(Instruction::VECTORLOAD);
    if (ld != nullptr && ld->GetOpCode() != V_LOAD_NOOP)
    {
        auto ldld = reinterpret_cast<VectorLoadOperationState *>(ld);
        auto opcode = ld->GetOpCode();
        out << "VectorLoadOperationState ld(" << opName.at(opcode) << ", "
            << ldld->GetPermissionValue() << ", " << ldld->GetIndexDest()
            << ", " << ldld->GetBase() << ", " << ldld->GetOffset() << ", "
            << ldld->GetStride() << ", " << ldld->GetIA() << ", "
            << ldld->GetMask() << ");\n"
            << "inst->SetOperationState(Instruction::VECTORLOAD, &ld);\n";
    }
    auto mti = inst->GetOperation(Instruction::MTI);
    if (mti != nullptr && mti->GetOpCode() != MTI_NOOP)
    {
        auto mtimti = reinterpret_cast<MTIOperationState *>(mti);
        auto opcode = mti->GetOpCode();
        out << "MTIOperationState mti(" << opName.at(opcode) << ", "
            << mtimti->GetPermissionValue() << ", " << mtimti->GetIndexX()
            << ", " << mtimti->GetMask() << ", " << mtimti->GetSelect()
            << ");\n"
            << "inst->SetOperationState(Instruction::MTI, &mti);\n";
    }
    auto mtr = inst->GetOperation(Instruction::MTR);
    if (mtr != nullptr && mtr->GetOpCode() != MTR_NOOP)
    {
        auto mtrmtr = reinterpret_cast<MTROperationState *>(mtr);
        auto opcode = mtr->GetOpCode();
        out << "MTROperationState mtr(" << opName.at(opcode) << ", "
            << mtrmtr->GetPermissionValue() << ", " << mtrmtr->GetIndexDest()
            << ", " << mtrmtr->GetSelect() << ");\n"
            << "inst->SetOperationState(Instruction::MTR, &mtr);\n";
    }
    auto misc = inst->GetOperation(Instruction::MISC);
    if (misc != nullptr && misc->GetOpCode() != MISC_NOOP)
    {
        auto miscmisc = reinterpret_cast<MiscOperationState *>(misc);
        auto opcode = misc->GetOpCode();
        out << "MiscOperationState misc(" << opName.at(opcode) << ", "
            << miscmisc->GetPermissionValue() << ", "
            << miscmisc->GetMiscOperand() << ", "
            << miscmisc->GetMiscCondition() << ", " << miscmisc->GetMiscTarget()
            << ");\n"
            << "inst->SetOperationState(Instruction::MISC, &misc);\n";
    }
    out << "CompleteInstruction(inst);\ninstruction_list.push_back(inst);\n}";
    return out.str();
}

namespace Visualization {
    std::unordered_map<int, std::string> opName = {
        {0, "S_NOOP"},
        {1, "S_HALT"},
        {2, "S_POP"},
        {3, "S_DELAY"},
        {4, "S_SMEM_LOAD"},
        {5, "S_SMEM_LOAD_OFFSET"},
        {6, "S_SMEM_STORE"},
        {7, "S_SET"},
        {8, "S_BRANCH"},
        {9, "S_CALL_ABOSOLUTE"},
        {10, "S_CALL_RELATIVE"},
        {11, "S_CALL_REGISTER"},
        {13, "S_FENCE"},
        {14, "S_DMA"},
        {15, "S_LOCAL_DMA"},
        {16, "S_STRIDED_DMA"},
        {17, "S_READ"},
        {18, "S_CONVERT_S32_TO_F32"},
        {19, "S_CONVERT_F32_TO_S32"},
        {20, "S_S32_ADDITION"},
        {21, "S_S32_SUBTRACTION"},
        {22, "S_U32_AND"},
        {23, "S_U32_OR"},
        {24, "S_U32_XOR"},
        {25, "S_U32_SHIFTLEFT"},
        {26, "S_U32_SHIFTRIGHT"},
        {27, "S_U32_MOVE"},
        {28, "S_U32_COUNTLEADINGZEROES"},
        {29, "S_U32_MULTIPLICATION"},
        {30, "S_F32_ADDITION"},
        {31, "S_F32_SUBTRACTION"},
        {32, "S_F32_MULTIPLICATION"},
        {33, "S_F32_MAX"},
        {34, "S_F32_MIN"},
        {35, "S_S32_EQUAL"},
        {36, "S_S32_NOTEQUAL"},
        {37, "S_S32_GREATER"},
        {38, "S_S32_GREATEREQUAL"},
        {39, "S_S32_LESSER"},
        {40, "S_S32_LESSER_EQUAL"},
        {41, "S_U32_CARRY"},
        {42, "S_F32_EQUAL"},
        {43, "S_F32_NOTEQUAL"},
        {44, "S_F32_GREATER"},
        {45, "S_F32_GREATEREQUAL"},
        {46, "S_F32_LESSER"},
        {47, "S_F32_LESSEREQUAL"},
        {48, "S_F32_IS_INF_OR_NAN"},
        {49, "S_PERMISSION_OR"},
        {50, "S_ARITHMETIC_SHIFT_RIGHT"},
        {51, "MAX_NUMBER_OF_SCALAR_INSTRUCTION"},
        {100, "V_NOOP"},
        {101, "V_HALF_FLOAT_PACK"},
        {102, "V_TWO_LOWER_INT8_PACK"},
        {103, "V_FOUR_INT8_PACK"},
        {104, "V_SUBCORE_ROTATE"},
        {106, "V_RNG_GENERATE_RANDOM_NUMBER"},
        {107, "V_RNG_READ_SEED"},
        {108, "V_RNG_RESEED"},
        {109, "V_F32_SOFTSIGN"},
        {110, "V_F32_LOG2"},
        {111, "V_F32_SIGMOID"},
        {112, "V_F32_RECIPROCAL"},
        {113, "V_F32_SQUAREROOT_RECIPROCAL"},
        {114, "V_F32_POWER"},
        {115, "V_F32_SOFTPLUS"},
        {116, "V_F32_EXPONENT"},
        {118, "V_CONVERT_S32_TO_F32"},
        {119, "V_CONVERT_F32_TO_S32"},
        {120, "V_S32_ADDITION"},
        {121, "V_S32_SUBTRACTION"},
        {122, "V_U32_AND"},
        {123, "V_U32_OR"},
        {124, "V_U32_XOR"},
        {125, "V_U32_SHIFTLEFT"},
        {126, "V_U32_SHIFTRIGHT"},
        {127, "V_U32_MOVE"},
        {128, "V_U32_COUNTLEADINGZEROES"},
        {129, "V_U32_MULTIPLICATION"},
        {130, "V_F32_ADDITION"},
        {131, "V_F32_SUBTRACTION"},
        {132, "V_F32_MULTIPLICATION"},
        {133, "V_F32_MAX"},
        {134, "V_F32_MIN"},
        {135, "V_S32_EQUAL"},
        {136, "V_S32_NOTEQUAL"},
        {137, "V_S32_GREATER"},
        {138, "V_S32_GREATEREQUAL"},
        {139, "V_S32_LESSER"},
        {140, "V_S32_LESSER_EQUAL"},
        {141, "V_U32_CARRY"},
        {142, "V_F32_EQUAL"},
        {143, "V_F32_NOTEQUAL"},
        {144, "V_F32_GREATER"},
        {145, "V_F32_GREATEREQUAL"},
        {146, "V_F32_LESSER"},
        {147, "V_F32_LESSEREQUAL"},
        {148, "V_F32_IS_INF_OR_NAN"},
        {149, "V_PERMISSION_OR"},
        {150, "V_ARITHMETIC_SHIFT_RIGHT"},
        {151, "V_ROUND_ARITHMETIC_SHIFT_RIGHT"},
        {152, "V_SELECT_VMASK0"},
        {153, "V_SELECT_VMASK1"},
        {154, "V_SELECT_VMASK2"},
        {155, "V_SELECT_VMASK3"},
        {156, "V_SELECT_VMASK4"},
        {157, "V_SELECT_VMASK5"},
        {158, "V_SELECT_VMASK6"},
        {159, "V_SELECT_VMASK7"},
        {160, "V_GET_V_CORE_ID"},
        {161, "V_SET_VMASK"},
        {162, "V_EXTRACT"},
        {163, "V_COMPOSE_FLOAT"},
        {164, "V_COUNT_NUMBER_OF_ONE"},
        {165, "V_RELUX"},
        {166, "V_CLAMP"},
        {167, "MAX_NUMBER_OF_V_INSTRUCTION"},
        {200, "V_STORE_NOOP"},
        {201, "V_STORE"},
        {202, "V_STORE_WITH_OFFSET"},
        {203, "V_STORE_WITH_VMASK0"},
        {204, "V_STORE_WITH_VMASK1"},
        {205, "V_STORE_WITH_VMASK2"},
        {206, "V_STORE_WITH_VMASK3"},
        {207, "V_STORE_WITH_VMASK4"},
        {208, "V_STORE_WITH_VMASK5"},
        {209, "V_STORE_WITH_VMASK6"},
        {210, "V_STORE_WITH_VMASK7"},
        {211, "V_STORE_INDEXED"},
        {212, "V_STORE_INDEXED_WITH_OFFSET"},
        {213, "V_STORE_INDEXED_WITH_VMASK0"},
        {214, "V_STORE_INDEXED_WITH_VMASK1"},
        {215, "V_STORE_INDEXED_WITH_VMASK2"},
        {216, "V_STORE_INDEXED_WITH_VMASK3"},
        {217, "V_STORE_INDEXED_WITH_VMASK4"},
        {218, "V_STORE_INDEXED_WITH_VMASK5"},
        {219, "V_STORE_INDEXED_WITH_VMASK6"},
        {220, "V_STORE_INDEXED_WITH_VMASK7"},
        {221, "V_STORE_SET_IA_OF_CORE"},
        {222, "V_STORE_SET_IA_OF_SUBCORE"},
        {223, "V_STORE_PUSH_TO_SCALAR_CORE"},
        {224, "V_STORE_FXC"},
        {225, "MAX_NUMBER_OF_V_STORE_INSTRUCTION"},
        {300, "V_LOAD_NOOP"},
        {301, "V_LOAD"},
        {302, "V_LOAD_WITH_OFFSET"},
        {303, "V_LOAD_WITH_VMASK0"},
        {304, "V_LOAD_WITH_VMASK1"},
        {305, "V_LOAD_WITH_VMASK2"},
        {306, "V_LOAD_WITH_VMASK3"},
        {307, "V_LOAD_WITH_VMASK4"},
        {308, "V_LOAD_WITH_VMASK5"},
        {309, "V_LOAD_WITH_VMASK6"},
        {310, "V_LOAD_WITH_VMASK7"},
        {311, "V_LOAD_INDEXED"},
        {312, "V_LOAD_INDEXED_WITH_OFFSET"},
        {313, "V_LOAD_INDEXED_WITH_VMASK0"},
        {314, "V_LOAD_INDEXED_WITH_VMASK1"},
        {315, "V_LOAD_INDEXED_WITH_VMASK2"},
        {316, "V_LOAD_INDEXED_WITH_VMASK3"},
        {317, "V_LOAD_INDEXED_WITH_VMASK4"},
        {318, "V_LOAD_INDEXED_WITH_VMASK5"},
        {319, "V_LOAD_INDEXED_WITH_VMASK6"},
        {320, "V_LOAD_INDEXED_WITH_VMASK7"},
        {321, "V_LOAD_WITH_SHUFFLE"},
        {322, "V_LOAD_INDEXED_WITH_SHUFFLE"},
        {323, "V_LOAD_FXC"},
        {324, "MAX_NUMBER_OF_V_LOAD_INSTRUCTION"},
        {400, "MTI_NOOP"},
        {401, "MTI_MUL_FLOAT_ROUNDED"},
        {402, "MTI_MUL_HIGHER_F16"},
        {403, "MTI_MUL_LOWER_F16"},
        {404, "MTI_MUL_F16_PACKED"},
        {405, "MTI_MUL_INT8_PACKED"},
        {406, "MTI_MUL_INT8_LOWER16_PACKED"},
        {407, "MTI_MUL_GSNF_ROUNDED"},
        {408, "MTI_MUL_GSNF_HIGHER16"},
        {409, "MTI_MUL_GSNF_LOWER16"},
        {410, "MTI_MUL_GSNF_PACKED_F16"},
        {411, "MTI_MUL_GSNF_PACKED_INT8"},
        {412, "MTI_MUL_GSNF_PACKED_INT8_LOWER16"},
        {413, "MTI_MUL_GSTF_ROUNDED"},
        {414, "MTI_MUL_GSTF_HIGHER16"},
        {415, "MTI_MUL_GSTF_LOWER16"},
        {416, "MTI_MUL_GSTF_PACKED_F16"},
        {417, "MTI_MUL_GSTF_PACKED_INT8"},
        {418, "MTI_MUL_GSTF_PACKED_INT8_LOWER16"},
        {419, "MTI_MUL_MASK_ROUNDED"},
        {420, "MTI_MUL_MASK_HIGER16"},
        {421, "MTI_MUL_MASK_LOWER16"},
        {422, "MTI_MUL_MASK_PACKED_F16"},
        {423, "MTI_MUL_MASK_PACKED_INT8"},
        {424, "MTI_MUL_MASK_PACKED_INT8_LOWER16"},
        {425, "MTI_MUL_MASK_GSNF_ROUNDED"},
        {426, "MTI_MUL_MASK_GSNF_HIGER16"},
        {427, "MTI_MUL_MASK_GSNF_LOWER16"},
        {428, "MTI_MUL_MASK_GSNF_PACKED_F16"},
        {429, "MTI_MUL_MASK_GSNF_PACKED_INT8"},
        {430, "MTI_MUL_MASK_GSNF_PACKED_INT8_LOWER16"},
        {431, "MTI_MUL_MASK_GSTF_ROUNDED"},
        {432, "MTI_MUL_MASK_GSTF_HIGER16"},
        {433, "MTI_MUL_MASK_GSTF_LOWER16"},
        {434, "MTI_MUL_MASK_GSTF_PACKED_F16"},
        {435, "MTI_MUL_MASK_GSTF_PACKED_INT8"},
        {436, "MTI_MUL_MASK_GSTF_PACKED_INT8_LOWER16"},
        {437, "MTI_LOAD_GSNF"},
        {438, "MTI_LOAD_GSTF"},
        {439, "MTI_PUSHGAIN_FLOAT_ROUNDED"},
        {440, "MTI_PUSHGAIN_HIGHER_F16"},
        {441, "MTI_PUSHGAIN_LOWER_F16"},
        {442, "MTI_PUSHGAIN_PACKED_INT8"},
        {444, "MTI_PUSHGAIN_TRANSPOSE_ROUND"},
        {445, "MTI_PUSHGAIN_TRANSPOSE_HIGHER_F16"},
        {446, "MTI_PUSHGAIN_TRANSPOSE_LOWER_F16"},
        {447, "MTI_PUSHGAIN_TRANSPOSE_PACKED_INT8"},
        {449, "MTI_PUSHGAIN_MASK_ROUNDED"},
        {450, "MTI_PUSHGAIN_MASK_HIGHER16"},
        {451, "MTI_PUSHGAIN_MASK_LOWER16"},
        {452, "MTI_PUSHGAIN_MASK_PACKED_INT8"},
        {454, "MTI_PUSHGAIN_MASK_TRANSPOSE_ROUND"},
        {455, "MTI_PUSHGAIN_MASK_TRANSPOSE_HIGHER16"},
        {456, "MTI_PUSHGAIN_MASK_TRANSPOSE_LOWER16"},
        {457, "MTI_PUSHGAIN_MASK_PACKED_TRANSPOSE_INT16"},
        {459, "MTI_TRANSPOSE_START"},
        {460, "MTI_TRANSPOSE"},
        {461, "MTI_TRANSPOSE_END"},
        {462, "MTI_TRANSPOSE_START_END"},
        {463, "MTI_TRANSPOSE_SEGMENT_START"},
        {464, "MTI_TRANSPOSE_SEGMENT"},
        {465, "MTI_TRANSPOSE_SEGMENT_END"},
        {466, "MTI_TRANSPOSE_SEGMENT_START_END"},
        {467, "MTI_TRANSPOSE_PACKED_START"},
        {468, "MTI_TRANSPOSE_PACKED"},
        {469, "MTI_TRANSPOSE_PACKED_END"},
        {470, "MTI_TRANSPOSE_PACKED_START_END"},
        {471, "MTI_TRANSPOSE_PACKED_SEGMENT_START"},
        {472, "MTI_TRANSPOSE_PACKED_SEGMENT"},
        {473, "MTI_TRANSPOSE_PACKED_SEGMENT_END"},
        {474, "MTI_TRANSPOSE_PACKED_SEGMENT_START_END"},
        {475, "MTI_PERMUTE"},
        {476, "MTI_PERMUTE_PACKED"},
        {477, "MTI_SET_PERMUTE"},
        {478, "MTI_SET_PERMUTE_SUBLANES"},
        {479, "MTI_SET_PERMUTE_BYTE"},
        {480, "MTI_SET_SPR"},
        {481, "MTI_REDUCTION_V_SUM"},
        {482, "MTI_REDUCTION_V_MAX"},
        {483, "MTI_REDUCTION_V_MIN"},
        {484, "MTI_REDUCTION_V_MAX_INDEX"},
        {485, "MTI_REDUCTION_V_MIN_INDEX"},
        {486, "MTI_REDUCTION_SEGMENTED_V_SUM"},
        {487, "MTI_REDUCTION_SEGMENTED_V_MAX"},
        {488, "MTI_REDUCTION_SEGMENTED_V_MIN"},
        {489, "MTI_REDUCTION_SEGMENTED_V_MAX_INDEX"},
        {490, "MTI_REDUCTION_SEGMENTED_V_MIN_INDEX"},
        {491, "MTI_REDUCTION_PACKED_V_SUM"},
        {492, "MTI_REDUCTION_PACKED_V_MAX"},
        {493, "MTI_REDUCTION_PACKED_V_MIN"},
        {494, "MTI_REDUCTION_PACKED_V_MAX_INDEX"},
        {495, "MTI_REDUCTION_PACKED_V_MIN_INDEX"},
        {496, "MTI_REDUCTION_PACKED_SEGMENTED_V_SUM"},
        {497, "MTI_REDUCTION_PACKED_SEGMENTED_V_MAX"},
        {498, "MTI_REDUCTION_PACKED_SEGMENTED_V_MIN"},
        {499, "MTI_REDUCTION_PACKED_SEGMENTED_V_MAX_INDEX"},
        {500, "MTI_REDUCTION_PACKED_SEGMENTED_V_MIN_INDEX"},
        {501, "MTI_ROTATE"},
        {502, "MTI_PACKED_ROTATE"},
        {503, "MAX_NUMBER_OF_MTI_INSTRUCTION"},
        {600, "MTR_NOOP"},
        {601, "MTR_READ_MATRIX_RESULT"},
        {602, "MTR_READ_TRANSPOSE_RESULT"},
        {603, "MTR_READ_UNARY_EXECUTION_RESULT"},
        {604, "MTR_READ_FUXI_CORD_RESULT"},
        {605, "MAX_NUMBER_OF_MTR_INSTRUCTION"},
        {700, "MISC_NOOP"},
        {701, "MISC_SET_SYNC_FLAG"},
        {702, "MISC_SYNC_FLAG_INCREMENT"},
        {703, "MISC_SYNC"},
        {704, "MISC_VMASK_OPERATION"},
        {705, "MISC_READ_SYNC_FLAG"},
        {706, "MISC_INTERRUPT"},
        {707, "MISC_CLEAR_RESULT_FIFO"},
        {708, "MISC_VECTOR_DELAY_SHORT"},
        {709, "MISC_VECTOR_DELAY_LONG"},
        {710, "MISC_REMOTE_SET_SYNC_FLAG"},
        {711, "MISC_REMOTE_SYNC_FLAG_INCREMENT"},
        {712, "MISC_TRACE"},
        {713, "MISC_SET_TRACEMARK"},
        {714, "MISC_CFENCE"},
    };

    std::string setDotEdge(const std::string& from, const std::string& to, const std::string& note = "") {
        std::string edge = "  \"" + from + "\" -> \"" + to + "\"";
        if (note != "") {
            edge += " [label=\"" + note + "\"]";
        }
        edge += ";\n";
        return edge;
    }

    int getNodeOp(InstScheduler::Inst233* inst) {
        using InstScheduler::Inst233;
        switch (inst->tag)
        {
        case Inst233::Tag::Scalar:
            return inst->scalarInst.op;
        case Inst233::Tag::Vector:
            return inst->vectorInst.op;
        case Inst233::Tag::VectorLoad:
            return inst->vectorLoadInst.op;
        case Inst233::Tag::VectorStore:
            return inst->vectorStoreInst.op;;
        case Inst233::Tag::MTI:
            return inst->mtiInst.op;
        case Inst233::Tag::MTR:
            return inst->mtrInst.op;
        case Inst233::Tag::MISC:
            return inst->miscInst.op;
        default:
            break;
        }
        return 0;
    }

    std::string getNodeLabel(InstScheduler::Inst233* inst) {
        using InstScheduler::Inst233;
        std::string label = "InstId:" + std::to_string(inst->rawInstId) + "\\n";
        uint16_t opcode = getNodeOp(inst);
        label += opName[opcode];
        return label;
    }

    std::string getEdgeLabel(InstScheduler::ResourceScheduler r) {
        using InstScheduler::ResourceScheduler;
        std::string label;
        switch(r.tag)
        {
        case ResourceScheduler::Tag::ScalarReg:
            label += "ScalarReg";
            break;
        case ResourceScheduler::Tag::VectorReg:
            label += "VectorReg";
            break;
        case ResourceScheduler::Tag::PermitReg:
            label += "PermitReg";
            break;
        case ResourceScheduler::Tag::VMaskReg:
            label += "VMaskReg";
            break;
        case ResourceScheduler::Tag::PCR:
            label += "PCR";
            break;
        case ResourceScheduler::Tag::SPR:
            label += "SPR";
            break;
        case ResourceScheduler::Tag::IAReg:
            label += "IAReg";
            break;
        case ResourceScheduler::Tag::SMem:
            label += "SMem";
            break;
        case ResourceScheduler::Tag::VMem:
            label += "VMem";
            break;
        case ResourceScheduler::Tag::CMem:
            label += "CMem";
            break;
        case ResourceScheduler::Tag::NWS_LOCK:
            label += "NWS";
            break;
        case ResourceScheduler::Tag::SYNC_FLAG:
            label += "SYNC_FLAG";
            break;
        case ResourceScheduler::Tag::HBM:
            label += "HBM";
            break;
        case ResourceScheduler::Tag::GMR:
            label += "GMR";
            break;
        case ResourceScheduler::Tag::VSF:
            label += "VSF";
            break;
        case ResourceScheduler::Tag::URF:
            label += "URF";
            break;
        case ResourceScheduler::Tag::CRF:
            label += "CRF";
            break;
        case ResourceScheduler::Tag::MRF:
            label += "MRF";
            break;
        case ResourceScheduler::Tag::TRF:
            label += "TRF";
            break;
        case ResourceScheduler::Tag::GSNF:
            label += "GSNF";
            break;
        case ResourceScheduler::Tag::GSTF:
            label += "GSTF";
            break;
        }
        return label;
    }

    std::string generateSubGraph(const InstScheduler::DAG& g) {
        enum NodeType {Scalar, Vector, VectorLoad, VectorStore, 
                        MTI, MTR, MISC};
        std::unordered_set<InstScheduler::Inst233* > subNodes[7];
        auto edges = g.edge;
        for(auto& each_from : edges) {
            for(auto& to : each_from.second) {
                auto from = each_from.first;
                subNodes[static_cast<int>(from->tag)].insert(from);
                subNodes[static_cast<int>(to->tag)].insert(to);
            }
        }
        std::string subGraph;
        if (!subNodes[Scalar].empty()) {
            subGraph += "subgraph cluster_Scalar {\n";
            for(auto& node : subNodes[Scalar]) {
                subGraph += "\"" + getNodeLabel(node) + "\";\n";
            }
            subGraph += "}\n";
        }
        if (!subNodes[Vector].empty()) {
            subGraph += "subgraph cluster_Vector {\n";
            for(auto& node : subNodes[Vector]) {
                subGraph += "\"" + getNodeLabel(node) + "\";\n";
            }
            subGraph += "}\n";
        }
        if (!subNodes[VectorLoad].empty()) {
            subGraph += "subgraph cluster_VectorLoad {\n";
            for(auto& node : subNodes[VectorLoad]) {
                subGraph += "\"" + getNodeLabel(node) + "\";\n";
            }
            subGraph += "}\n";
        }
        if (!subNodes[VectorStore].empty()) {
            subGraph += "subgraph cluster_VectorStore {\n";
            for(auto& node : subNodes[VectorStore]) {
                subGraph += "\"" + getNodeLabel(node) + "\";\n";
            }
            subGraph += "}\n";
        }
        if (!subNodes[MTI].empty()) {
            subGraph += "subgraph cluster_MTI {\n";
            for(auto& node : subNodes[MTI]) {
                subGraph += "\"" + getNodeLabel(node) + "\";\n";
            }
            subGraph += "}\n";
        }
        if (!subNodes[MTR].empty()) {
            subGraph += "subgraph cluster_MTR {\n";
            for(auto& node : subNodes[MTR]) {
                subGraph += "\"" + getNodeLabel(node) + "\";\n";
            }
            subGraph += "}\n";
        }
        if (!subNodes[MISC].empty()) {
            subGraph += "subgraph cluster_MISC {\n";
            for(auto& node : subNodes[MISC]) {
                subGraph += "\"" + getNodeLabel(node) + "\";\n";
            }
            subGraph += "}\n";
        }
        return subGraph;
    }

    std::string generateDotGraph(const InstScheduler::DAG& g, const std::string& graphName) {
        std::string dotGraph = "digraph " + graphName + " {\n";
        // dotGraph += generateSubGraph(g);
        auto edges = g.edge;
        for(auto& each_from : edges) {
            for(auto& to : each_from.second) {
                auto from = each_from.first;
                std::string fromNodeLabel = getNodeLabel(from); 
                std::string toNodeLabel = getNodeLabel(to);
                std::string resInfo;
                for(auto& r : g.edgeRes.at(std::make_tuple(from, to))) {
                    resInfo += getEdgeLabel(r) + " ";
                }
                dotGraph += setDotEdge(fromNodeLabel, toNodeLabel, resInfo);
            }
        }
        dotGraph += "}\n";
        return dotGraph;
    }

    // generate dot file
    void generateDotFile(const std::string& dotGraph, const std::string& fileName) {
        std::ofstream file(fileName, std::ios::app);
        if (file.is_open()) {
            file << dotGraph;
            file.close();
        } else {
            std::cerr << "Unable to open file for writing: " << fileName << std::endl;
        }
    }
}

namespace InstScheduler 
{
namespace ConstrainScheduler
{
// TODO: compute ASAP and ALAP cycle (completed)
// TODO: build mul/pop constrains
// TODO: build register def constrains
// TODO: build immediate constrains
using namespace operations_research::sat;

// slot constrains
// -> 指令槽的数量
const int kScalarSlotCount = 2;
const int kScalarOneSlotCount = 1;
const int kScalarTwoSlotCount = 1;

const int kVectorSlotCount = 2;
const int kVectorOneSlotCount = 1;
const int kVectorTwoSlotCount = 1;

const int kVectorLoadSlotCount = 1;
const int kVectorStoreSlotCount = 1;

const int kMTISlotCount = 1;
const int kMTRSlotCount = 1;
const int kMISCSlotCount = 1;

// -> 处理FIFO策略的处理器
class FifoProcessor {
    void makeConstrains() 
    {

    }

};


// -> 提供了管理处理器中指令所需的所有信息的方式
struct InstIR {
    Inst233* inst;
    /*
        · 指令的唯一标识符
        · 在调度周期内的上界和下界
        · 调度结果
        · 定义列表以及使用列表: 可能分别表示了指令的操作数和结果
    */
    int instId;
    int upperBound;
    int lowerBound;
    int schedResult = -1;
    std::vector<int> defList;
    std::vector<int> useList;
};

// -> 用于将输入的指令包装成InstIR对象的
class InstWrapper {
public:
    using InstGraph = std::unordered_map<Inst233*, std::unordered_set<Inst233*>>;
    
    // -> 接受一副DAG和一个最大调度周期
    InstWrapper(DAG* g, int maxCycle) : g(g), maxScheduleCycle(maxCycle) {} 

    DAG* g;
    int maxScheduleCycle;
    
    enum Slot 
    { 
        Scalar = 0, 
        ScalarOne, 
        ScalarTwo, 
        Vector,
        VectorOne,
        VectorTwo,
        VectorLoad,
        VectorStore,
        MTI,
        MTR,
        MISC
    };


    std::unordered_map<std::tuple<ResourceScheduler::Tag, int, int>, std::vector<int>> getDefiner()
    {
        return fifodefiner;
    }

    std::unordered_map<std::tuple<ResourceScheduler::Tag, int, int>, std::tuple<int, int>> getsecDefiner()
    {
        return secdefiner;
    }

    /*
        wrap函数使用所有的节点调用wrap函数，将指令包装成InstIR实例。函数首先计算DAG边的反向图，
    然后遍历所有的节点，使用wrap函数包装节点。最后，返回包含InstIR实例的vector
    
    */
    std::vector<InstIR> wrap()
    {
        reversedEdges = edgesReverse();
        std::vector<InstIR> insts;
        for (auto & node : g->nodes)
        {
            // std::cout << "node: " << node->rawInstId << std::endl;
            wrap(insts, node);        
        }
        
        for (const auto& pair : fifodefiner)
        {
            if (std::get<0>(pair.first) == ResourceScheduler::Tag::GSTF ||
                std::get<0>(pair.first) == ResourceScheduler::Tag::GSNF ||
                std::get<0>(pair.first) == ResourceScheduler::Tag::TRF)
            {
                int minVal = std::numeric_limits<int>::max();
                int minlocal = 0;
                const std::vector<int>& values = pair.second;
                for (int index : values)
                {
                    if (insts[index].inst->rawInstId < minVal)  
                    {
                        minVal = insts[index].inst->rawInstId;
                        minlocal = index;
                    }
                }
                secdefiner[pair.first] = std::make_tuple(minlocal, 0);
                std::cout << "minlocal: " << std::get<0>(secdefiner[pair.first]) << std::endl;
            }
        }

        for (const auto& pair : user)
        {
            if (std::get<0>(pair.first) == ResourceScheduler::Tag::TRF)
            {
                int maxVal = std::numeric_limits<int>::min();
                int maxlocal = 0;
                const std::vector<int> values = pair.second;
                for (int index : values)
                {
                    if (insts[index].inst->rawInstId > maxVal)
                    {
                        maxVal = insts[index].inst->rawInstId;
                        maxlocal = index;
                    }
                }
                std::get<1>(secdefiner[pair.first]) = maxlocal;
                std::cout << "maxlocal: " << std::get<1>(secdefiner[pair.first]) << std::endl;
            }
        }
        return insts;
    }

private:
    // -> 检查方法。检查这个实例在DAG中是否有后继节点。如果没有，表示这个指令是DAG的结束节点，返回true
    bool isEnd(Inst233* inst) {
        auto it = g->edge.find(inst);
        if (it == g->edge.end() || it->second.size() == 0) 
        {
            return true;
        }
        return false;
    }

    /*
     计算并返回一个给定前驱实例集合所需的每一个插槽类型的延迟。这个函数会根据具体的
    指令类型，分别计算每种插槽的使用次数，然后将其除以对应插槽的最大数量(也就是类中
    定义的常量)，得出每种插槽的延迟
    */
    std::array<int, 11> getSlotDelay(const std::set<Inst233*>& predecessors)
    {
        std::array<int, 11> slotUsedCount = {0};
        for (auto & inst : predecessors)
        {
            switch (inst->tag)
            {
            case Inst233::Tag::Scalar:
            {
                slotUsedCount[Slot::Scalar]++;
                auto scalarOp = inst->scalarInst.op;
                if (scalarSlot.at(scalarOp) == Slot0) 
                {
                    slotUsedCount[Slot::ScalarOne]++;
                } 
                else if (scalarSlot.at(scalarOp) == Slot1) 
                {
                    slotUsedCount[Slot::ScalarTwo]++;
                }
                break;
            }
            case Inst233::Tag::Vector:
            {
                slotUsedCount[Slot::Vector]++;
                auto vectorOp = inst->vectorInst.op;
                if (vectorSlot.at(vectorOp) == Slot0) 
                {
                    slotUsedCount[Slot::VectorOne]++;
                } 
                else if (vectorSlot.at(vectorOp) == Slot1) 
                {
                    slotUsedCount[Slot::VectorTwo]++;
                }
                break;
            }
            case Inst233::Tag::VectorLoad:
                slotUsedCount[Slot::VectorLoad]++;
                break;
            case Inst233::Tag::VectorStore:
                slotUsedCount[Slot::VectorStore]++;
                break;
            case Inst233::Tag::MTI:
                slotUsedCount[Slot::MTI]++;
                break;
            case Inst233::Tag::MTR:
                slotUsedCount[Slot::MTR]++;
                break;  
            case Inst233::Tag::MISC:
                slotUsedCount[Slot::MISC]++;
                break;
            default:
                break;
            }
        }

        /* 
            将每个具体插槽使用数量除以对应插槽的最大数量，实现所谓的向上取整操作。
        目的是通过比例计算出每种类型插槽的延迟。这样，如果某一类型插槽的使用数量
        超过最大数量，那么其对应的插槽延迟就会增加。
        
        */
        auto intCeil = [](int& dividend, const int& divistor) {   
            dividend /= divistor; 
        };

        intCeil(slotUsedCount[Scalar], kScalarSlotCount);
        intCeil(slotUsedCount[ScalarOne], kScalarOneSlotCount);
        intCeil(slotUsedCount[ScalarTwo], kScalarTwoSlotCount);
        
        intCeil(slotUsedCount[Vector], kVectorSlotCount);
        intCeil(slotUsedCount[VectorOne], kVectorOneSlotCount);
        intCeil(slotUsedCount[VectorTwo], kVectorTwoSlotCount);
        
        intCeil(slotUsedCount[VectorLoad], kVectorLoadSlotCount);
        intCeil(slotUsedCount[VectorStore], kVectorStoreSlotCount);
        intCeil(slotUsedCount[MTI], kMTISlotCount);
        intCeil(slotUsedCount[MTR], kMTRSlotCount);
        intCeil(slotUsedCount[MISC], kMISCSlotCount);
    
        return slotUsedCount;
    }

    /*
        首先它尝试在一个已知的上界映射(upperBound)中找到给定指令(inst)的上界。如果找到，
    他会返回存储的上界。如果没有找到，并且给定指令没有前驱指令(即其入度为0)，函数会为该
    指令存储上界为1，并返回。如果以上两种情况都不满足，函数遍历所有输入指令的前驱，计算他们
    的上界，同时更新与给定指令相关的所有前驱的集合(Predecessors)。然后求出插槽的延迟，最大
    的前驱延迟，以及最大的插槽延迟。最后，函数返回以上三种延迟中的最大值加一。
    */
    int getUpperBound(Inst233* inst)
    {
        if (upperBound.find(inst) != upperBound.end())
        {
            return upperBound[inst];
        }
        else if (g->inDegree[inst] == 0)
        {
            upperBound[inst] = 1;
            return 1;
        }
        else
        {   // find all predecessors
            std::vector<int> fatherDelay;
            for (auto & father : reversedEdges[inst])
            {
                int count = getUpperBound(father);
                fatherDelay.push_back(count);
                if (count != 0)
                {
                    if (Predecessors.find(father) != Predecessors.end()) 
                    {
                        std::set_union(Predecessors[inst].begin(), Predecessors[inst].end(),
                                       Predecessors[father].begin(), Predecessors[father].end(), 
                                       std::inserter(Predecessors[inst], Predecessors[inst].begin()));
                    }
                }
            }
            std::array<int, 11> slotDelay = getSlotDelay(Predecessors[inst]);
            int maxFatherDelay = *(std::max_element(
                                   fatherDelay.begin(), fatherDelay.end()));
            int maxSlotDelay = *(std::max_element(
                                 slotDelay.begin(), slotDelay.end()));
            return 1 + std::max(maxFatherDelay, maxSlotDelay);
        }
    }

    /*
        工作原理与getUpperBound类似，但是其方向相反，计算的是后继节点(所有那些被给定指令直接
    指向的节点)的上界。它首先查找存储的上界，然后查找那些没有后继节点的指令(即这个指令是否是图
    的末端节点)。如果以上都没有找到，那么他会找到所有后继节点并计算他们的上界，同时更新与给定
    指令相关的所有后继的集合(Decessors)。接着，函数会求出插槽的延迟，最大的后继延迟，以及最大的插槽延迟。
    
        反向上界为了获取某个任务能够开始的最晚时间，与之相关的是他们的最终任务及其所有先导任务的完成时间(对应这
    里的fatherDelay)，以及完成该任务所需的资源延迟时间(对应这里的slotDelay)。
    */
    int getReverseUpperBound(Inst233* inst)
    {
        if (reverseUpperBound.find(inst) != reverseUpperBound.end())
        {
            return reverseUpperBound[inst];
        }
        else if (isEnd(inst))
        {
            reverseUpperBound[inst] = 1;
            return 1;
        }
        else
        {   // find all decessors
            // -> 寻找指令的所有父指令(执行在他之前的指令)并计算他们的上界
            std::vector<int> fatherDelay;
            for (auto & father : g->edge[inst])
            {
                int count = getReverseUpperBound(father);
                fatherDelay.push_back(count);
                if (count != 0)
                {
                    if (Decessors.find(father) != Decessors.end()) 
                    {
                        std::set_union(Decessors[inst].begin(), Decessors[inst].end(),
                                       Decessors[father].begin(), Decessors[father].end(), 
                                       std::inserter(Decessors[inst], Decessors[inst].begin()));
                    }
                }
            }
            // -> 计算在满足该指令的所有需要执行的任务已经完成的情况下，完成该任务所需的资源延迟时间
            std::array<int, 11> slotDelay = getSlotDelay(Decessors[inst]);
            int maxFatherDelay = *(std::max_element(
                                   fatherDelay.begin(), fatherDelay.end()));
            int maxSlotDelay = *(std::max_element(
                                 slotDelay.begin(), slotDelay.end()));
            return 1 + std::max(maxFatherDelay, maxSlotDelay);
        }
    }
    
    /*
        此函数用于计算一个指令的调度下界。函数使用lowerBound查找表来保存计算好的调度下界。
    如果给定指令的带哦都下界已知，则直接返回。如果给定指令是结束指令，其下界被设定为最大调度
    周期maxScheduleCycle。其他情况下，调度下界被计算为最大调度周期maxScheduleCycle减去该
    指令的逆向调度上界reverseUpperBound再加1.
    */
    int getLowerBound(Inst233* inst)
    {
        if (lowerBound.find(inst) != lowerBound.end())
        {
            return lowerBound[inst];
        }
        else if (isEnd(inst))
        {
            lowerBound[inst] = maxScheduleCycle;
            return maxScheduleCycle;
        }
        else
        { 
            lowerBound[inst] = maxScheduleCycle - getReverseUpperBound(inst) + 1;
            return lowerBound[inst];
        }
    }

    /*
        用于计算出当前指令图的反向图。
        反向图仅仅是将原图的边反转过来。如此，原图中从node到neighbor的边，在反向图中就会变成
    一条从neighbor到node的边。
    */
    InstGraph edgesReverse()
    {
        InstGraph newEdges;
        for (const auto & pair : g->edge) 
        {
            const auto & node = pair.first;
            const auto & neighbors = pair.second;
            for (const auto & neighbor : neighbors) 
            {
                std::cout << "edges: " << node->rawInstId  << " -> " << neighbor->rawInstId << std::endl;
                newEdges[neighbor].insert(node);
            }
        }
        return newEdges;
    }

    /*
        负责将原始的Inst233指令转换为InstIR格式，并将其添加到结果列表中。该函数首先创建一个新的
    InstIR实例，并设置其指令ID(等于现有指令列表的大小)、包装的指令、以及调度的上下界(调用上面的
    getLowerBound和getUpperBound方法来获取)。然后，他将包装好的新指令添加到输入指令列表isnts的
    末尾
    */
    void wrap(std::vector<InstIR>& insts, Inst233* inst) 
    {
        InstIR wrappedInst;
        wrappedInst.instId = insts.size();
        std::cout << "wrapped: " << inst->rawInstId << ": " <<  inst->InstId << std::endl;
        wrappedInst.inst = inst;
        inst->InstId = wrappedInst.instId;
        wrappedInst.lowerBound = getLowerBound(inst) - 1;
        wrappedInst.upperBound = getUpperBound(inst) - 1;
        insts.push_back(wrappedInst);

        for (auto & res : inst-> writeResource())
        {
            if (res.isFifo() && (res.tag == ResourceScheduler::Tag::GSTF ||
                                res.tag == ResourceScheduler::Tag::GSNF ||
                                res.tag == ResourceScheduler::Tag::TRF))
            {
                std::cout << "Write TRF ID: " << res.serialId << std::endl;
                if (fifodefiner.count({res.tag, res.regId, res.serialId}) == 0) {
                    std::vector<int> Id;
                    Id.push_back(inst->InstId);
                    fifodefiner[{res.tag, res.regId, res.serialId}] = Id;
                    // std::cout << "GSTF ID: " <<  res.serialId << " - " << inst->rawInstId << std::endl;
                } else {
                    fifodefiner[{res.tag, res.regId, res.serialId}].push_back(inst->InstId);
                    // std::cout << "GSTF ID: " << res.serialId << " - " << inst->rawInstId << std::endl;
                }
            }
        }

        for (auto & res : inst->readResource())
        {
            if (res.isFifo())
            {
                if (res.tag == ResourceScheduler::Tag::GSTF ||
                    res.tag == ResourceScheduler::Tag::GSNF ||
                    res.tag == ResourceScheduler::Tag::TRF)
                {
                    std::cout << "Read GSNF/GSTF ID: " << res.serialId << std::endl;
                    if(user.count({res.tag, res.regId, res.serialId}) == 0)
                    {
                        std::vector<int> Id;
                        Id.push_back(inst->InstId);
                        user[{res.tag, res.regId, res.serialId}] = Id;
                    } else {
                        user[{res.tag, res.regId, res.serialId}].push_back(inst->InstId);
                    }
                }
            }

        }
    }

private:
    /*
        · 指令图的反向边
        · 指令的前驱
        · 指令的后继
        · 指令的上界
        · 指令的下界
        · 指令的逆向调度上界
    */
    InstGraph reversedEdges;
    std::unordered_map<Inst233*, std::set<Inst233*>> Predecessors;
    std::unordered_map<Inst233*, std::set<Inst233*>> Decessors;
    std::unordered_map<Inst233*, int> upperBound;
    std::unordered_map<Inst233*, int> lowerBound;
    std::unordered_map<Inst233*, int> reverseUpperBound;
    std::unordered_map<std::tuple<ResourceScheduler::Tag, int, int>, std::vector<int>> fifodefiner;
    std::unordered_map<std::tuple<ResourceScheduler::Tag, int, int>, std::tuple<int, int>> secdefiner;
    std::unordered_map<std::tuple<ResourceScheduler::Tag, int, int>, std::vector<int>> user;
};

struct CpInstructionBuilder 
{
    Inst233 *scalar0 = nullptr;
    Inst233 *scalar1 = nullptr;
    Inst233 *vector0 = nullptr;
    Inst233 *vector1 = nullptr;
    Inst233 *vectorLoad = nullptr;
    Inst233 *vectorStore = nullptr;
    Inst233 *mti = nullptr;
    Inst233 *mtr = nullptr;
    Inst233 *misc = nullptr;

    int usedImme = 0;

    void peek(const InstIR& ir)
    {
        const auto & inst = ir.inst;
        switch (inst->tag)
        {
        case Inst233::Tag::Scalar:
        {
            auto scalarOp = inst->scalarInst.op;
            if (scalarSlot.at(scalarOp) == Slot0) 
            {
                scalar0 = inst;
            } 
            else if (scalarSlot.at(scalarOp) == Slot1) 
            {
                scalar1 = inst;    
            }
            else if (scalar0 == nullptr)
            {
                scalar0 = inst;
            }
            else
            {
                scalar1 = inst;
            }
            return;
        }
        case Inst233::Tag::Vector:
        {
            auto vectorOp = inst->vectorInst.op;
            if (vectorSlot.at(vectorOp) == Slot0) 
            {
                vector0 = inst;
            } 
            else if (vectorSlot.at(vectorOp) == Slot1) 
            {
                vector1 = inst;
            }
            else if (vector0 == nullptr)
            {
                vector0 = inst;
            }
            else
            {
                vector1 = inst;
            }
            return;
        }
        case Inst233::Tag::VectorLoad:
            vectorLoad = inst;
            return;
        case Inst233::Tag::VectorStore:
            vectorStore = inst;
            return;
        case Inst233::Tag::MTI:
            mti = inst;
            return;
        case Inst233::Tag::MTR:
            mtr = inst;
            return;
        case Inst233::Tag::MISC:
            misc = inst;
            return;
        default:
            break;
        }  
    } 

    static void
    FillImme(int operationImme, Instruction &dest, Instruction &src)
    {
        if (operationImme & IMME0)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE0,
                src.GetImmediateValue(Instruction::IMMEDIATE0));
        }
        if (operationImme & IMME1)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE1,
                src.GetImmediateValue(Instruction::IMMEDIATE1));
        }
        if (operationImme & IMME2)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE2,
                src.GetImmediateValue(Instruction::IMMEDIATE2));
        }
        if (operationImme & IMME3)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE3,
                src.GetImmediateValue(Instruction::IMMEDIATE3));
        }
        if (operationImme & IMME4)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE4,
                src.GetImmediateValue(Instruction::IMMEDIATE4));
        }
        if (operationImme & IMME5)
        {
            dest.SetImmediateValue(
                Instruction::IMMEDIATE5,
                src.GetImmediateValue(Instruction::IMMEDIATE5));
        }
        if (operationImme & VSIMME0)
        {
            dest.SetImmediateValue(
                Instruction::VECTORSCALARIMMEDIATE0,
                src.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0));
        }
        if (operationImme & VSIMME1)
        {
            dest.SetImmediateValue(
                Instruction::VECTORSCALARIMMEDIATE1,
                src.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1));
        }
        if (operationImme & VSIMME2)
        {
            dest.SetImmediateValue(
                Instruction::VECTORSCALARIMMEDIATE2,
                src.GetImmediateValue(Instruction::VECTORSCALARIMMEDIATE2));
        }
    }

    void
    rename(uint16_t& operand, const uint32_t& virtualId, InstRegAlloc& allocator)
    {
        uint16_t newRegId = allocator.alloc(virtualId);
        std::cout << "rename " << operand 
                  << " to " << newRegId 
                  << " serialId ID: " << virtualId << std::endl;
        operand = newRegId;
    }

    Instruction* 
    build(bool coutInfo = false) 
    {
        InstRegAlloc sregAllocator(&sregAllocBase);
        InstRegAlloc vregAllocator(&vregAllocBase);

        static int count = 0;

        if (scalar0 == nullptr && scalar1 == nullptr && vector0 == nullptr &&
            vector1 == nullptr && vectorLoad == nullptr &&
            vectorStore == nullptr && mti == nullptr && mtr == nullptr &&
            misc == nullptr)
        {
            std::cerr << COLOR::RED << "SCHEDULER PEEK INST FAIL" << std::endl;
            exit(-1);
        }
        auto pInst = new Instruction();
        Instruction &inst = *pInst;
        if (coutInfo)
        {
            std::cout << "#" << count++ << ": " << std::endl;
        }
        if (scalar0 != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << scalar0->rawInstId << ", ";
            }
            auto &sinst = scalar0->scalarInst;
            FillImme(sinst.usedImme, inst, *sinst.pInst);
            for (int i = 0; i < sinst.regTable.size(); ++i)
            {
                if (sinst.regTable[i].flag) 
                {
                    uint32_t virtualId = sinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case ScalarInst::s_x:
                        rename(sinst.x, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_y:
                        rename(sinst.y, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_dest:
                        rename(sinst.dest, virtualId, sregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            if (sinst.op == S_LOCAL_DMA || sinst.op == S_STRIDED_DMA)
            {
                ScalarOperationState state(sinst.op,
                                           sinst.permit,
                                           sinst.src_addr,
                                           sinst.length,
                                           sinst.dest_addr,
                                           sinst.syncflag,
                                           sinst.misc);
                inst.SetOperationState(Instruction::SCALARONE, &state);
            }
            else
            {
                ScalarOperationState state(sinst.op,
                                           sinst.permit,
                                           sinst.x,
                                           sinst.y,
                                           sinst.dest);
                inst.SetOperationState(Instruction::SCALARONE, &state);
            }
            scalar0 = nullptr;
        }
        if (scalar1 != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << scalar1->rawInstId << ", ";
            }
            auto &sinst = scalar1->scalarInst;
            FillImme(sinst.usedImme, inst, *sinst.pInst);
            for (int i = 0; i < sinst.regTable.size(); ++i)
            {
                if (sinst.regTable[i].flag) 
                {
                    uint32_t virtualId = sinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case ScalarInst::s_x:
                        rename(sinst.x, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_y:
                        rename(sinst.y, virtualId, sregAllocator);
                        break;
                    case ScalarInst::s_dest:
                        rename(sinst.dest, virtualId, sregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            ScalarOperationState state(sinst.op,
                                       sinst.permit,
                                       sinst.x,
                                       sinst.y,
                                       sinst.dest);
            inst.SetOperationState(Instruction::SCALARTWO, &state);
            scalar1 = nullptr;
        }
        if (vector0 != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << vector0->rawInstId << ", ";
            }
            auto &vinst = vector0->vectorInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for(int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag)
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorInst::v_x:
                        rename(vinst.x, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_y:
                        rename(vinst.y, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_dest:
                        rename(vinst.dest, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            VectorOperationState state(vinst.op,
                                       vinst.permit,
                                       vinst.x,
                                       vinst.y,
                                       vinst.dest);
            inst.SetOperationState(Instruction::VECTORONE, &state);
            vector0 = nullptr;
        }
        if (vector1 != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << vector1->rawInstId << ", ";
            }
            auto &vinst = vector1->vectorInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for(int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag)
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorInst::v_x:
                        rename(vinst.x, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_y:
                        rename(vinst.y, virtualId, vregAllocator);
                        break;
                    case VectorInst::v_dest:
                        rename(vinst.dest, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            VectorOperationState state(vinst.op,
                                       vinst.permit,
                                       vinst.x,
                                       vinst.y,
                                       vinst.dest);
            inst.SetOperationState(Instruction::VECTORTWO, &state);
            vector1 = nullptr;
        }
        if (vectorLoad != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << vectorLoad->rawInstId << ", ";
            }
            auto &vinst = vectorLoad->vectorLoadInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for (int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag) 
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorLoadInst::v_dest:
                    {
                        rename(vinst.dest, virtualId, vregAllocator);
                        break;
                    }
                    case VectorLoadInst::vs_imm_0:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, newRegId);
                        break;
                    }
                    case VectorLoadInst::vs_imm_1:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1, newRegId);
                        break;
                    }
                    case VectorLoadInst::vs_imm_2:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE2, newRegId);
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
            VectorLoadOperationState state(vinst.op,
                                           vinst.permit,
                                           vinst.dest,
                                           vinst.base,
                                           vinst.offset,
                                           vinst.stride,
                                           vinst.ia,
                                           vinst.mask);
            inst.SetOperationState(Instruction::VECTORLOAD, &state);
            vectorLoad = nullptr;
        }
        if (vectorStore != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << vectorStore->rawInstId << ", ";
            }
            auto &vinst = vectorStore->vectorStoreInst;
            FillImme(vinst.usedImme, inst, *vinst.pInst);
            for (int i = 0; i < vinst.regTable.size(); ++i)
            {
                if (vinst.regTable[i].flag) 
                {
                    uint32_t virtualId = vinst.regTable[i].virRegId;
                    switch (i)
                    {
                    case VectorStoreInst::v_x:
                    {
                        rename(vinst.x, virtualId, vregAllocator);
                        break;
                    }
                    case VectorStoreInst::vs_imm_0:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE0, newRegId);
                        break;
                    }
                    case VectorStoreInst::vs_imm_1:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE1, newRegId);
                        break;
                    }
                    case VectorStoreInst::vs_imm_2:
                    {
                        uint16_t newRegId = sregAllocator.alloc(virtualId);
                        inst.SetImmediateValue(Instruction::VECTORSCALARIMMEDIATE2, newRegId);
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
            VectorStoreOperationState state(vinst.op,
                                            vinst.permit,
                                            vinst.x,
                                            vinst.base,
                                            vinst.offset,
                                            vinst.stride,
                                            vinst.ia,
                                            vinst.mask);
            inst.SetOperationState(Instruction::VECTORSTORE, &state);
            vectorStore = nullptr;
        }
        if (mti != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << mti->rawInstId << ", ";
            }
            auto &minst = mti->mtiInst;
            FillImme(minst.usedImme, inst, *minst.pInst);
            for (int i = 0; i < minst.regTable.size(); i++)
            {
                if (minst.regTable[i].flag)
                {
                    int virtualId = minst.regTable[i].virRegId;
                    switch (i)
                    {
                    case MTIInst::mti_x:
                        rename(minst.x, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            MTIOperationState state(minst.op,
                                    minst.permit,
                                    minst.x,
                                    minst.mask,
                                    minst.select);
            inst.SetOperationState(Instruction::MTI, &state);
            mti = nullptr;
        }
        if (mtr != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << mtr->rawInstId << ", ";
            }
            auto &minst = mtr->mtrInst;
            FillImme(minst.usedImme, inst, *minst.pInst);
            for (int i = 0; i < minst.regTable.size(); i++)
            {
                if (minst.regTable[i].flag)
                {
                    int virtualId = minst.regTable[i].virRegId;
                    switch (i)
                    {
                    case MTRInst::v_dest:
                        rename(minst.dest, virtualId, vregAllocator);
                        break;
                    default:
                        break;
                    }
                }
            }
            MTROperationState state(minst.op,
                                    minst.permit,
                                    minst.dest,
                                    minst.select);
            inst.SetOperationState(Instruction::MTR, &state);
            mtr = nullptr;
        }
        if (misc != nullptr)
        {
            if (coutInfo)
            {
                std::cout << "@" << misc->rawInstId << ", ";
            }
            auto &minst = misc->miscInst;
            FillImme(minst.usedImme, inst, *minst.pInst);
            MiscOperationState state(minst.op,
                                     minst.permit,
                                     minst.operand,
                                     minst.cond,
                                     minst.target);
            inst.SetOperationState(Instruction::MISC, &state);
            misc = nullptr;
        }
        __CompleteInstruction(&inst);
        if (coutInfo)
        {
            inst.ToAssembly();
        }
        usedImme = 0;
        return pInst;
    }
};

// -> fuse函数的目标是找出可能的指令调度，以减少程序的运行时间 upper_bound作为调度的上界
std::vector<Instruction *>
fuse(InstScheduler::DAG& g, int upper_bound, bool coutInfo) 
{
    // -> 创建约束规划模型的实例
    CpModelBuilder cp_model;
    // -> 定义了调度的最大周期和指令的总数
    const int maxScheduleCycle = upper_bound;
    const int instTotalCount = g.nodes.size();

    std::cout << "maxScheduleCycle: " << maxScheduleCycle << std::endl;
    std::cout << "instTotalCount: " << instTotalCount << std::endl;

    /* 
        首先使用InstWrapper类将指令转换成InstIR对象，这些对象包含调度的上下界。
    然后构造一个二维的布尔变量矩阵x，表示在每个调度时间段某个特定指令是否被执行。
    */
    InstWrapper wrapper(&g, maxScheduleCycle);
    std::vector<InstIR> instIRs = wrapper.wrap();
    std::unordered_map<std::tuple<ResourceScheduler::Tag, int, int>, std::vector<int>> fifodefiner = wrapper.getDefiner();
    std::unordered_map<std::tuple<ResourceScheduler::Tag, int, int>, std::tuple<int, int>> secdefiner = wrapper.getsecDefiner();
    for (auto & inst : instIRs)
    {
        std::cout << "ID: " << inst.instId << "  " 
                  << "upper: " << inst.upperBound << "  "
                  << "lower: " << inst.lowerBound << std::endl; 
    }

    std::vector<std::vector<BoolVar>> x;
    // -> 使得每个指令在每个调度周期都有一个对应的布尔变量
    for (int i = 0; i < instTotalCount; i++) 
    {
        std::vector<BoolVar> row;
        for (int j = 0; j < maxScheduleCycle; j++) 
        {
            row.push_back(cp_model.NewBoolVar());
        }
        x.push_back(row);
    }

    /*
        通过约束编程模型(cp_model),函数添加以下类型的约束:
            · 每一条指令只能被调度一次
            · 指令必须在其调度窗口(上下界)内被调度
            · 设备资源的使用不能超过每个时间周期的最大容量
            · 按依赖关系严格调度指令，即，只有当所有输入已经被处理时，指令才会被执行
            · IMMEDIATE VALUE 的资源限制
    */
    // schedule once constraint
    for (int i = 0; i < instTotalCount; i++) 
    {  
        cp_model.AddExactlyOne(x[i]);
        for (int j = 0; j < maxScheduleCycle; j++)
        {
            if (j < instIRs[i].upperBound || j > instIRs[i].lowerBound)
            {
                cp_model.FixVariable(x[i][j], 0);
            }
        }
    }

    // device constraint 因为设备资源时有限的
    for (int j = 0; j < maxScheduleCycle; j++) 
    {
        LinearExpr scalarSum;
        LinearExpr scalarOneSum;
        LinearExpr scalarTwoSum;
        LinearExpr vectorSum;
        LinearExpr vectorOneSum;
        LinearExpr vectorTwoSum;
        
        LinearExpr vloadSum;
        LinearExpr vstoreSum;
        LinearExpr mtiSum;
        LinearExpr mtrSum;
        LinearExpr miscSum;

        for (auto & inst : g.nodes) 
        {
            switch (inst->tag) 
            {
            case Inst233::Tag::Scalar:
            {
                scalarSum += x[inst->InstId][j];
                auto scalarOp = inst->scalarInst.op;
                if (scalarSlot.at(scalarOp) == Slot0) 
                {
                    scalarOneSum += x[inst->InstId][j];
                } 
                else if (scalarSlot.at(scalarOp) == Slot1) 
                {
                    scalarTwoSum += x[inst->InstId][j];
                }
                break;
            }
            case Inst233::Tag::Vector:
            {
                vectorSum += x[inst->InstId][j];
                auto vectorOp = inst->vectorInst.op;
                if (vectorSlot.at(vectorOp) == Slot0) 
                {
                    vectorOneSum += x[inst->InstId][j];
                } 
                else if (vectorSlot.at(vectorOp) == Slot1) 
                {
                    vectorTwoSum += x[inst->InstId][j];
                }
                break;
            }
            case Inst233::Tag::VectorLoad:
                vloadSum += x[inst->InstId][j];
                break;
            case Inst233::Tag::VectorStore:
                vstoreSum += x[inst->InstId][j];
                break;
            case Inst233::Tag::MTI:
                mtiSum += x[inst->InstId][j];
                break;
            case Inst233::Tag::MTR:
                mtrSum += x[inst->InstId][j];
                break;
            case Inst233::Tag::MISC:
                miscSum += x[inst->InstId][j];
                break;
            default:
                break;
            }
        }

        cp_model.AddLessOrEqual(scalarSum, kScalarSlotCount);
        cp_model.AddLessOrEqual(scalarOneSum, kScalarOneSlotCount);
        cp_model.AddLessOrEqual(scalarTwoSum, kScalarTwoSlotCount);
        cp_model.AddLessOrEqual(vectorSum, kVectorSlotCount);
        cp_model.AddLessOrEqual(vectorOneSum, kVectorOneSlotCount);
        cp_model.AddLessOrEqual(vectorTwoSum, kVectorTwoSlotCount);

        cp_model.AddLessOrEqual(vloadSum, kVectorLoadSlotCount);
        cp_model.AddLessOrEqual(vstoreSum, kVectorSlotCount);
        cp_model.AddLessOrEqual(mtiSum, kMTISlotCount);
        cp_model.AddLessOrEqual(mtrSum, kMTRSlotCount);
        cp_model.AddLessOrEqual(miscSum, kMISCSlotCount);
    }

    // dependency constraint
    // -> 指遵循所有的依赖关系，始终保证对所有的依赖指令，他们总是在当前指令之间被调度
    for (auto& from : g.nodes) 
    {
        for (auto& to : g.edge[from]) 
        {
            LinearExpr lhs;
            LinearExpr rhs;
            for (int j = 0; j < maxScheduleCycle; j++) 
            {
                lhs += j * x[from->InstId][j];
                rhs += j * x[to->InstId][j];
            }
            cp_model.AddLessThan(lhs, rhs); 
        }
    }

    for (int sel = 0; sel < 2; sel++)
    {
        // GSTF constraint
        std::vector<IntervalVar> intervalgstf;
        for (int Id = 0; secdefiner.count({ResourceScheduler::Tag::GSTF, sel, Id}); Id++)
        {
            IntVar startTime = cp_model.NewIntVar({0, maxScheduleCycle});
            LinearExpr startTimeExpr;
            int definerId = std::get<0>(secdefiner[{ResourceScheduler::Tag::GSTF, sel, Id}]);
            std::cout << "definerId: " << definerId << std::endl;
            for (int j = 0; j < maxScheduleCycle; j++)
            {
                startTimeExpr += j * x[definerId][j];
            }
            cp_model.AddEquality(startTime, startTimeExpr);

            IntVar endTime = cp_model.NewIntVar({0, maxScheduleCycle + 1});
            LinearExpr endTimeExpr;
            int userId = std::get<1>(secdefiner[{ResourceScheduler::Tag::GSTF, sel, Id}]);
            std::cout << "userId: " << userId << std::endl;
            for (int j = 0; j < maxScheduleCycle; j++)
            {
                endTimeExpr += j * x[userId][j];
            }
            cp_model.AddEquality(endTime, endTimeExpr);

            IntVar sz = cp_model.NewIntVar({0, maxScheduleCycle});
            cp_model.AddEquality(sz, endTime - startTime);
            IntervalVar lifeTime = cp_model.NewIntervalVar(startTime, sz, endTime);
            intervalgstf.push_back(lifeTime);
        }
        cp_model.AddNoOverlap(intervalgstf);

        // GSNF constraint
        std::vector<IntervalVar> intervalgsnf;
        for (int Id = 0; secdefiner.count({ResourceScheduler::Tag::GSNF, sel, Id}); Id++)
        {
            IntVar startTime = cp_model.NewIntVar({0, maxScheduleCycle});
            LinearExpr startTimeExpr;
            int definerId = std::get<0>(secdefiner[{ResourceScheduler::Tag::GSNF, sel, Id}]);
            std::cout << "definerId: " << definerId << std::endl;
            for (int j = 0; j < maxScheduleCycle; j++)
            {
                startTimeExpr += j * x[definerId][j];
            }
            cp_model.AddEquality(startTime, startTimeExpr);

            IntVar endTime = cp_model.NewIntVar({0, maxScheduleCycle + 1});
            LinearExpr endTimeExpr;
            int userId = std::get<1>(secdefiner[{ResourceScheduler::Tag::GSNF, sel, Id}]);
            std::cout << "userId: " << userId << std::endl;
            for (int j = 0; j < maxScheduleCycle; j++)
            {
                endTimeExpr += j * x[userId][j];
            }
            cp_model.AddEquality(endTime, endTimeExpr);

            IntVar sz = cp_model.NewIntVar({0, maxScheduleCycle});
            cp_model.AddEquality(sz, endTime - startTime);
            IntervalVar lifeTime = cp_model.NewIntervalVar(startTime, sz, endTime);
            intervalgsnf.push_back(lifeTime);
        }
        cp_model.AddNoOverlap(intervalgsnf);

        // TRF constraint
        std::vector<IntervalVar> intervaltrf;
        for (int Id = 1; secdefiner.count({ResourceScheduler::Tag::TRF, sel, Id}); Id++)
        {
            IntVar startTime = cp_model.NewIntVar({0, maxScheduleCycle});
            LinearExpr startTimeExpr;
            int definerId = std::get<0>(secdefiner[{ResourceScheduler::Tag::TRF, sel, Id}]);
            std::cout << "definerId: " << definerId << std::endl;
            for (int j = 0; j < maxScheduleCycle; j++)
            {
                startTimeExpr += j * x[definerId][j];
            }
            cp_model.AddEquality(startTime, startTimeExpr);

            IntVar endTime = cp_model.NewIntVar({0, maxScheduleCycle + 1});
            LinearExpr endTimeExpr;
            int userId = std::get<1>(secdefiner[{ResourceScheduler::Tag::TRF, sel, Id}]);
            std::cout << "userId: " << userId << std::endl;
            for (int j = 0; j < maxScheduleCycle; j++)
            {
                endTimeExpr += j * x[userId][j];
            }
            cp_model.AddEquality(endTime, endTimeExpr + 1);

            IntVar sz = cp_model.NewIntVar({0, maxScheduleCycle});
            cp_model.AddEquality(sz, endTime - startTime);
            IntervalVar lifeTime = cp_model.NewIntervalVar(startTime, sz, endTime);
            intervaltrf.push_back(lifeTime);
        }
        cp_model.AddNoOverlap(intervaltrf);
    }

    // imme constraint
    // -> 针对每个周期，限制了立即数的最大使用量
    for (int j = 0; j < maxScheduleCycle; j++)
    {
        LinearExpr immSum[9];
        for (int i = 0; i < instTotalCount; i++)
        {
            auto usedImme = instIRs[i].inst->usedImme();

            for (int k = 0; k < 9; k++)
            {
                if (usedImme & (1 << k))
                {
                    immSum[k] += x[i][j];
                }
            }
        }
        for (int k = 0; k < 9; k++)
        {
            cp_model.AddLessOrEqual(immSum[k], 1);
        }
    }

    // target
    // -> 计算最后一条指令的最大调度时间，目标时最小化这个时间。
    std::vector<LinearExpr> scheduleTime;
    for (auto& inst : g.nodes)
    {
        if (DAGHeight[inst] == 1)
        {
            LinearExpr expr;
            for (int j = 0; j < maxScheduleCycle; j++) 
            {
                expr += j * x[inst->InstId][j];
            }
            scheduleTime.push_back(expr);
        }
    }

    // -> 函数定义目标函数为所有结束指令的最大完成时间，目标是最小化这个最大完成时间
    IntVar obj_var = cp_model.NewIntVar({0, maxScheduleCycle}).WithName("makespan");
    cp_model.AddMaxEquality(obj_var, scheduleTime);
    cp_model.Minimize(obj_var);

    const CpSolverResponse response = Solve(cp_model.Build());

    // rebuild instruction list
    /*
        接下来，将构建好的模型传给CP-SAT求解器进行求解，求解结果将用于指令调度。根据求解器返回的结果，
    使用CpInstructionBuilder类将结果转换为最终的指令调度，并返回结果列表。
    */
    CpInstructionBuilder builder;
    std::vector<Instruction*> res;

    if (response.status() == CpSolverStatus::OPTIMAL ||
        response.status() == CpSolverStatus::FEASIBLE) 
    {
        int resultCycle = SolutionIntegerValue(response, obj_var);
        for (int j = 0; j <= resultCycle; j++) 
        {
            for (int i = 0; i < instTotalCount; i++) 
            {
                if (SolutionBooleanValue(response, x[i][j])) 
                {
                    builder.peek(instIRs[i]);
                }
            }
            res.push_back(builder.build(coutInfo));
        }
    } 
    else 
    {
        std::cout << "No solution found." << std::endl;
    }

    return res;
}

} // namespace ConstrainScheduler
} // namespace InstScheduler 


