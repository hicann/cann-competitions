/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file selu.h
 * \brief
 */
#ifndef SELU_H
#define SELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "selu_tiling_data.h"
#include "selu_tiling_key.h"

namespace NsSelu {

using namespace AscendC;

#define ALPHA 1.67326324235f
#define SCALE 1.05070098736f
#define SCALE_ALPHA_PRODUCT 1.75809934085f

constexpr int32_t BUFFER_NUM = 2;
constexpr float  CONST_ZERO = 0.0f;
constexpr float  SCALAR_NEGATIVE_ONE = -1.0f;

template <typename T>
class Selu {
public:
    __aicore__ inline Selu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SeluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void SeluComputeV2(LocalTensor<T>& x, LocalTensor<T>& y, 
                                        LocalTensor<float>& xLocal_f, 
                                        LocalTensor<float>& res, 
                                        LocalTensor<float>& res2,
                                        LocalTensor<float>& p1,
                                        LocalTensor<uint8_t>& bits);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<QuePosition::VECCALC>  compare_bits;
    TBuf<QuePosition::VECCALC>  tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int32_t coreDataNum = 0;
    int32_t tileNum = 0;
    int32_t tileDataNum = 0;
    int32_t tailDataNum = 0;
    int32_t processDataNum = 0;
};

template <typename T>
__aicore__ inline void Selu<T>::Init(GM_ADDR x, GM_ADDR y, const SeluTilingData* tilingData)
{   
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero"); 
    uint32_t blockIdx = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;

    if (blockIdx <  tilingData->tailBlockNum){
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    }
    else{
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (GetBlockIdx() - tilingData->tailBlockNum);
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));  //p1 
    pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));  //res
    pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));  //res2
    pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(float));  //xLocal_f
    pipe.InitBuffer(compare_bits, this->tileDataNum * sizeof(uint8_t));   //bits
    
    if constexpr (std::is_same_v<T, int8_t>){
        pipe.InitBuffer(tmp5, this->tileDataNum * sizeof(int32_t));  //p2
        pipe.InitBuffer(tmp6, this->tileDataNum * sizeof(int32_t));  //p3
        pipe.InitBuffer(tmp7, this->tileDataNum * sizeof(half));  //xLocal_h
        pipe.InitBuffer(tmp8, this->tileDataNum * sizeof(half));  //xLocal_h2
        pipe.InitBuffer(tmp9, this->tileDataNum * sizeof(half));  //p4
    }
}

template <typename T>
__aicore__ inline void Selu<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Selu<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->tileDataNum], yLocal, this->processDataNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Selu<T>::SeluComputeV2(LocalTensor<T>& x, LocalTensor<T>& y, LocalTensor<float>& xLocal_f, LocalTensor<float>& res, LocalTensor<float>& res2, LocalTensor<float>& p1, LocalTensor<uint8_t>& bits)
{
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>){ 
        Cast(xLocal_f, x, RoundMode::CAST_NONE, this->processDataNum);
        Muls(res, xLocal_f, SCALE, this->processDataNum);  //res =slhs
        Exp(res2, xLocal_f, this->processDataNum);     //res2 = exp_res
        Adds(res2, res2, SCALAR_NEGATIVE_ONE, this->processDataNum);
        Muls(res2, res2, SCALE_ALPHA_PRODUCT, this->processDataNum);   //res2 = srhs
        Duplicate(p1, static_cast<float>(0.0), this->processDataNum); 
        Compare(bits, xLocal_f, p1, CMPMODE::GT, this->processDataNum);
        Select(xLocal_f, bits, res, res2, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        Cast(y, xLocal_f, RoundMode::CAST_RINT, this->processDataNum); 
    }
    else{
        Muls(res, x, SCALE, this->processDataNum);  //res =slhs
        Exp(res2, x, this->processDataNum);     //res2 = exp_res
        Adds(res2, res2, SCALAR_NEGATIVE_ONE, this->processDataNum);
        Muls(res2, res2, SCALE_ALPHA_PRODUCT, this->processDataNum);   //res2 = srhs
        Duplicate(p1, static_cast<float>(0.0), this->processDataNum); 
        Compare(bits, x, p1, CMPMODE::GT, this->processDataNum);
        Select(y, bits, res, res2, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    } 

}

template <typename T>
__aicore__ inline void Selu<T>::Compute(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<int32_t> p2, p3;
    LocalTensor<half> xLocal_h, xLocal_h2, p4;

    auto p1 = tmp1.Get<float>();
    auto res = tmp2.Get<float>();
    auto res2 = tmp3.Get<float>();
    auto xLocal_f = tmp4.Get<float>();
    auto bits = compare_bits.Get<uint8_t>();
    if constexpr (std::is_same_v<T, int8_t>){
        p2 = tmp5.Get<int32_t>();
        p3 = tmp6.Get<int32_t>();
        xLocal_h = tmp7.Get<half>();
        xLocal_h2 = tmp8.Get<half>();
        p4 = tmp9.Get<half>();
    }

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>){ 
            SeluComputeV2(xLocal, yLocal, xLocal_f, res, res2, p1, bits);
            outputQueueY.EnQue<T>(yLocal);
            inputQueueX.FreeTensor(xLocal);
            return; 
    }
    else if constexpr (std::is_same_v<T, int32_t>){
        Cast(xLocal_f, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    }
    else if constexpr (std::is_same_v<T, int8_t>){
        Cast(xLocal_h, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(xLocal_f, xLocal_h, RoundMode::CAST_NONE, this->processDataNum);
    }

    Muls(p1, xLocal_f, static_cast<float>(0.0f), this->processDataNum); 
    Min(res, xLocal_f, p1, this->processDataNum);   //res  negative_res
    Max(res2, xLocal_f, p1,  this->processDataNum); //res2 positive_res
    Exp(res, res, this->processDataNum);
    Adds(res, res, SCALAR_NEGATIVE_ONE, this->processDataNum);
    Muls(res, res, SCALE_ALPHA_PRODUCT, this->processDataNum);
    if constexpr (std::is_same_v<T, int8_t> ){
        Cast(p2, res, RoundMode::CAST_CEIL, this->processDataNum);
        Cast(res, p2, RoundMode::CAST_NONE, this->processDataNum);
    }

    Muls(res2, res2, SCALE, this->processDataNum);
    Add(res, res, res2, this->processDataNum);
    if constexpr (std::is_same_v<T, int8_t>){
        Cast(p2, res, RoundMode::CAST_TRUNC, this->processDataNum); // F32 → S32 truncate toward zero（对应 F32_S32Z）
        Duplicate(p3, static_cast<int32_t>(255), this->processDataNum); 
        And(p2, p2, p3, this->processDataNum); 
        SetDeqScale(static_cast<half>(1.0));
        Cast(xLocal_h, p2, RoundMode::CAST_NONE, this->processDataNum);
        Adds(xLocal_h, xLocal_h, static_cast<half>(128.0), this->processDataNum);         //ProcessUint8Int8Overflow
        Muls(xLocal_h2, xLocal_h, static_cast<half>(1.0 / 256.0), this->processDataNum);
        Cast(p2, xLocal_h2, RoundMode::CAST_FLOOR, this->processDataNum);
        Duplicate(p3, static_cast<int32_t>(256), this->processDataNum);
        Mul(p2, p2, p3, this->processDataNum);
        SetDeqScale(static_cast<half>(1.0));                                           // int32 -> half
        Cast(p4, p2, RoundMode::CAST_NONE, this->processDataNum);
        Sub(xLocal_h, xLocal_h, p4, this->processDataNum);
        Adds(xLocal_h, xLocal_h, static_cast<half>(-128.0), this->processDataNum);
        Cast(yLocal, xLocal_h, RoundMode::CAST_NONE, this->processDataNum);

        outputQueueY.EnQue<T>(yLocal);
        inputQueueX.FreeTensor(xLocal);
        return;
    }

    if constexpr (std::is_same_v<T, int32_t>){ 
        Cast(yLocal, res, RoundMode::CAST_TRUNC, this->processDataNum);
    }
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Selu<T>::Process()
{
    int32_t loopCount = this->tileNum ;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if( i == loopCount - 1){
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsSelu
#endif // SELU_H
