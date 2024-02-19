#pragma once
#ifndef __DATA_H__
#    define __DATA_H__

#    include "GPT2Define.h"
#    include <cstdint>
#    include <numeric>
#    include <string>
#    include <queue>

template <uint32_t N>
struct data;

template <uint32_t U>
void
Fill(Inst2 &inst, const data<U> &dest, float value)
{
    dest.asVMem(inst) = value;
}

template <uint32_t U>
data<U>
Alloc(Inst2 &inst, const std::array<uint32_t, U> &dims)
{
    data<U> res(-1, dims);
    res.addr = inst.Alloc(res.size(), 1024).startAddr;
    return res;
}

template <uint32_t U>
data<U>
Alloc(Inst2 &inst, const std::array<uint32_t, U> &dims, uint32_t cap)
{
    data<U> res(-1, dims);
    cap      = std::max(res.size(), cap);
    res.addr = inst.Alloc(cap, 1024).startAddr;
    res.cap  = cap;
    return res;
}

template <uint32_t N>
struct data
{
    uint32_t addr;
    uint32_t hbmaddr;
    std::array<uint32_t, N> dims;
    uint32_t cap = 0;
    bool own     = true;

    explicit data(uint32_t addr = -1) : addr(addr)
    {
        dims.fill(1);
        dims.back() = 0;
        hbmaddr = -1;
    }

    data(uint32_t addr, const std::array<uint32_t, N> &dims)
        : addr(addr), dims(dims), own(true)
    {
        cap = size();
    }

    data(const data &) = default;
    data &operator=(const data &) = default;

    data(data &&d) noexcept : addr(d.addr), dims(d.dims), cap(d.cap)
    {
        d.own == false;
    }

    data &
    operator=(data &&d) noexcept
    {
        own   = true;
        addr  = d.addr;
        dims  = d.dims;
        cap   = d.cap;
        d.own = false;
        hbmaddr = d.hbmaddr;
        return *this;
    }

    uint32_t
    size() const
    {
        assert(own);

        return std::accumulate(dims.begin(),
                               dims.end(),
                               1u,
                               std::multiplies<uint32_t>());
    }

    void
    concat(Inst2 &inst2, const data<N> &d)
    {
        assert(own);

        auto newSize = size() + d.size();

        for (int i = 1; i < N; i++)
        {
            // assert(dims[i] == d.dims[i]);
        }

        auto newDims = dims;
        newDims[0] += d.dims[0];

        if (cap >= size() + d.size())
        {
            Memcopy(d.asVMem(inst2),
                    asVMem(inst2)[Range(size(), size() + d.size())]);
            resize(newDims);
        }
        else
        {
            data<N> newMem = Alloc<N>(inst2, newDims, newSize * 2);
            Memcopy(asVMem(inst2), newMem.asVMem(inst2)[Range(0, size())]);
            Memcopy(d.asVMem(inst2),
                    newMem.asVMem(inst2)[Range(size(), newSize)]);
            *this = std::move(newMem);
        }
    }

    void
    concat(Inst2 &inst2, const data<N - 1> &d)
    {
        assert(own);

        auto newSize = size() + d.size();

        for (int i = 1; i < N; i++)
        {
            assert(dims[i] == d.dims[i - 1]);
        }

        auto newDims = dims;
        newDims[0]++;

        if (cap >= size() + d.size())
        {
            Memcopy(d.asVMem(inst2),
                    asVMem(inst2)[Range(size(), size() + d.size())]);
            resize(newDims);
        }
        else
        {
            data<N> newMem = Alloc<N>(inst2, newDims, newSize * 2);
            Memcopy(asVMem(inst2), newMem.asVMem(inst2)[Range(0, size())]);
            Memcopy(d.asVMem(inst2),
                    newMem.asVMem(inst2)[Range(size(), newSize)]);
            *this = std::move(newMem);
        }
    }

    template <uint32_t U>
    data<U>
    as() const
    {
        assert(own);
        static_assert(U >= N, "");

        std::array<uint32_t, U> d;
        d.fill(1);
        std::copy(dims.begin(), dims.end(), d.begin() + U - N);
        return data<U>(addr, d);
    }

    data<N>
    slice(uint32_t begin, int32_t end = -1) const
    {
        assert(own);

        if (end == -1 || end > dims[0])
        {
            end = dims[0];
        }
        if (begin > dims[0])
        {
            begin = dims[0];
        }
        auto d = dims;
        d[0]   = end - begin;
        return data<N>(addr + (d[0] == 0 ? 0 : (begin * size() / d[0])), d);
    }

    data<N - 1>
    operator[](uint32_t idx) const
    {
        assert(own);
        std::array<uint32_t, N - 1> d;
        std::copy(dims.begin() + 1, dims.end(), d.begin());
        return data<N - 1>(addr + idx * size() / dims[0], d);
    }

    void
    resize(const std::array<uint32_t, N> &newDims)
    {
        assert(own);

        dims = newDims;
    }

    VMem
    asVMem(Inst2 &inst2) const
    {
        return VMem(addr, std::max(size(), cap), true, &inst2);
    }

    uint32_t
    endAddr() const
    {
        return addr + std::max(size(), cap);
    }
};

#endif