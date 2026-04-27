# 需求背景（required）

## 需求来源

## 背景介绍

### Cross算子实现优化

基于Cross算子历史TBE版本使用Ascend C编程语言进行优化。

Cross算子（TBE）实现路径和相关API路径

Cross算子实现路径为：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl

Cross算子实现中的API路径：/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl

### Cross算子TBE实现现状分析

通过对Cross算子TBE版本的功能分析，当前支持的能力如下：
① input_x1和input_x2支持float16、float、int32、int8、uint8、int16六种格式的输入。
② 算子不支持广播。
③ 计算表达式（叉积公式）：
对于两个三维向量 $\mathbf{a} = (a_1, a_2, a_3)$ 和 $\mathbf{b} = (b_1, b_2, b_3)$，叉积结果为：
$$
\mathbf{a} \times \mathbf{b} = (a_2b_3 - a_3b_2,\; a_3b_1 - a_1b_3,\; a_1b_2 - a_2b_1)
$$
即：
- $output_i = left_j \cdot right_k - right_j \cdot left_k$
- $output_j = right_i \cdot left_k - left_i \cdot right_k$
- $output_k = left_i \cdot right_j - right_i \cdot left_j$

④ 对于float16输入，在计算时转换为float32进行高精度计算，再转换回float16；对于float32和其他整数类型，直接计算。
⑤ 使用vmul、vsub等接口实现计算。

Cross算子TBE版本的整体流程图如下图所示：
![tbe2.png](https://raw.gitcode.com/user-images/assets/7649531/3275f14e-1a62-435f-9d56-74191cf407b1/tbe2.png 'tbe2.png')

算子原型

| 名称 | 类别 | dtype | format | shape | 介绍 |
|------|------|-------|--------|-------|------|
| x1 | 输入 | float16/float/int32/int8/uint8/int16 | ND | all | 第一个输入张量 |
| x2 | 输入 | float16/float/int32/int8/uint8/int16 | ND | all | 第二个输入张量 |
| dim | 属性 | int | - | - | 进行叉积计算的维度，默认值为-65530 |
| y | 输出 | float16/float/int32/int8/uint8/int16 | ND | 同输入 | 输出张量，叉积计算结果 |

相关约束
Atlas A2 训练系列产品/Atlas 800I A2推理产品支持float16、float32、int32、int8、uint8、int16
x1、x2、y 三个张量的shape和dtype必须完全一致
张量最大维度数为8

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现Cross算子，支持float16/float/int32/int8/uint8/int16数据类型

## 需求拆解

1. 支持float16/float/int32/int8/uint8/int16数据类型
1. 性能不低于TBE版本

# 详细设计（required）

## 算子分析

### 数学公式

对于两个三维向量 $\mathbf{a} = (a_1, a_2, a_3)$ 和 $\mathbf{b} = (b_1, b_2, b_3)$，叉积结果为：
$$
\mathbf{a} \times \mathbf{b} = (a_2b_3 - a_3b_2,\; a_3b_1 - a_1b_3,\; a_1b_2 - a_2b_1)
$$
即：
- $output_i = left_j \cdot right_k - right_j \cdot left_k$
- $output_j = right_i \cdot left_k - left_i \cdot right_k$
- $output_k = left_i \cdot right_j - right_i \cdot left_j$

### 支持数据类型

float16/float/int32/int8/uint8/int16

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

由于输入 self 和 other 形状相同，当前实现不支持广播场景下的 kernel 侧特殊排布处理，算子计算过程依赖输入 shape 和 dim 信息。Host 侧首先根据输入 shape 和 dim 计算：

intervalNum：cross 维度之后的连续元素个数
loopTimes：cross 三元组的组数
算子本质上是对长度为 3 的向量做叉积计算。当前实现将输入展平后，按照 dim 对应的 cross 轴进行分组：

当 intervalNum > 1 时，按 stride=intervalNum 访问同一组三个分量
当 intervalNum == 1 时，说明每组三个分量在内存中天然连续，可切换为 group mode 处理
任务划分采用 tile 化方式。Host 侧根据 UB 大小、数据类型长度、kernel 临时 buffer 数量等信息计算 tileDataNum，表示单次搬运/计算的数据量。随后根据 totalTileCount 和可用 coreNum 设置 blockDim，并将 tile 尽量均匀分配到各个 core。尾块不单独拆出复杂调度信息，而是统一交由 kernel 侧通过 DataCopyPad 处理，以减少 host 侧切分复杂度。


##### 1. 分核策略：
当前实现遵循“优先使用满核、尽量均分”的原则。

首先通过平台信息获取：
AIV 核数 coreNum
UB 大小 ubSize

然后根据输入 shape、dtype 和 dim 计算总工作量。对不同模式分别处理：
normal mode：总 tile 数为 loopTimes * ceil(intervalNum / tileDataNum)
group mode：总 tile 数为 ceil(loopTimes / tileDataNum)

最终分核策略为：
blockDim = min(coreNum, totalTileCount)
每个 core 处理的 tile 数尽量均分
若不能整除，前面的若干 core 多分配 1 个 tile
这种方式可以避免某些 core 空跑，同时保持 host 逻辑简单稳定。当前实现没有单独维护“大核/小核数据量”字段，而是直接基于 tile 数均分。

##### 2. 数据分块和内存优化策略：
遵循“尽量使用 UB、同时保证 kernel 中间 buffer 可容纳”的原则。

Host 侧通过平台接口获取 UB 大小后，结合 block size 和不同 dtype 所需的 buffer 数量，估算单次 tile 能处理的数据规模。不同 dtype 的 UB 预算不同：

float / int32 / int16：除输入输出外，还需一个临时 buffer
float16：由于计算过程中需要先转 fp32 再计算，因此需要更多 fp32 临时 buffer
group mode：只需要连续搬运两路输入和一路输出，buffer 数相对更少
切分策略如下：
normal mode 下，tileDataNum 表示单次处理多少个 interval 元素
group mode 下，tileDataNum 表示单次处理多少组三元组，实际搬运大小为 tileDataNum * 3
为避免 UB 超限，tileDataNum 会受以下因素约束：
UB 总大小
block 对齐大小
dtype 字节数
kernel 临时 buffer 数量
当前模式下的最大可处理 work unit 数

尾块场景下，若单次 tile 不能整除 block 对齐大小，kernel 使用 DataCopyPad 对尾部不足整块的数据进行补齐搬运和写回，从而保证计算流程统一，不额外增加 host 侧复杂的尾块调度逻辑。

##### 3. tilingkey规划策略：

tilingkey固定为0，无需分支感知,在kernel侧做运行时分支。

#### 3.2.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

对于float16数据类型，转换为float32进行计算，计算完成后再转换回float16；

对于int8、uint8数据类型，先转换为int32，计算完成后再转回int8、uint8；

对于float32和整数类型（int32、int16），直接计算。

无广播填充需求，直接在Compute阶段执行元素级运算。

根据tilingkey（固定）执行核函数。

Ascend C的Cross算子流程见下图。
![SoC Version Input-2026-03-25-050038.png](https://raw.gitcode.com/user-images/assets/7649531/7cb3a43d-11e3-4612-a1ca-681e349a094b/SoC_Version_Input-2026-03-25-050038.png 'SoC Version Input-2026-03-25-050038.png')

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |

## 算子约束限制

不支持广播。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 不低于TBE版本 |  |
| 性能标准 | 不低于TBE版本 |  |

## 兼容性分析

新算子，不涉及兼容性分析