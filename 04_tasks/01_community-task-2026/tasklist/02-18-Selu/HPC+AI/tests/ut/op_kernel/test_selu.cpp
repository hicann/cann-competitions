/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <cstdint>
#include <iostream>
#include <string>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#endif

#include "../../../op_kernel/selu.cpp"
#include "../../../op_kernel/selu_tiling_data.h"

using namespace std;

class SeluTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SeluTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "SeluTest TearDown" << endl;
    }
};

TEST_F(SeluTest, test_case_0)
{
    constexpr uint32_t dataNum = 32 * 4 * 4 * 4;
    size_t xByteSize = dataNum * sizeof(float);
    size_t yByteSize = dataNum * sizeof(float);
    size_t tilingDataSize = sizeof(SeluTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xByteSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yByteSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024 * 1024 * 16));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    SeluTilingData* tilingData = reinterpret_cast<SeluTilingData*>(tiling);

    tilingData->smallCoreDataNum = 2048;
    tilingData->bigCoreDataNum = 2112;
    tilingData->tileDataNum = 4032;
    tilingData->smallTailDataNum = 2048;
    tilingData->bigTailDataNum = 2112;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    auto KernelSelu = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::selu<0>(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSelu, blockDim, x, y, workspace, reinterpret_cast<uint8_t*>(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}


