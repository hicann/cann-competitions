# 需求背景（required）

## 需求来源

## 背景介绍

### SoftMarginLossGrad算子实现优化

基于SoftMarginLossGrad算子历史TBE版本使用Ascend C编程语言进行优化。

SoftMarginLossGrad算子（TBE）实现路径和相关API路径

SoftMarginLossGrad算子实现路径为：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl

SoftMarginLossGrad算子实现中的API路径：/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl

### SoftMarginLossGrad算子TBE实现现状分析

通过对SoftMarginLossGrad算子TBE版本的功能分析，当前支持的能力如下：
①input_predict、input_label、input_dout支持float16、float32、bfloat16三种格式输入。
② 算子支持自动广播机制，predict、label、dout三个输入的形状可以不同，系统会自动将较小维度的输入广播到最大形状（shape_max）进行计算。同时支持维度补齐，当predict维度高于label或dout时，会在低维输入的形状前自动补1。
③ 计算表达式：
$$
\frac{\partial \text{Loss}}{\partial x} = 
\begin{cases} 
\dfrac{-y \cdot e^{-xy}}{1 + e^{-xy}} \cdot \text{dout}, & \text{reduction}=\text{"none"} \\[8pt]
\dfrac{-y \cdot e^{-xy}}{(1 + e^{-xy}) \cdot N} \cdot \text{dout}, & \text{reduction}=\text{"mean"}
\end{cases}
$$

其中 $N$ 为predict的元素总数（$N = \prod_{i} \text{predict\_shape}_i$），当reduction为"mean"时需要除以元素个数进行归一化。

④ 算子限制：单tensor大小限制为2147483648（2^31）个元素。
⑤ 对于 float16、bfloat16 输入，转换为float32后计算，
⑥ 使用 vexp、vdiv、vmuls等接口实现计算。

SoftMarginLossGrad算子TBE版本的整体流程图如下图所示：
![tbes.png](https://raw.gitcode.com/user-images/assets/7665709/d7dd4cbd-7f78-4707-9356-2500bf051256/tbes.png 'tbes.png')

算子原型

| 名称 | 类别 | dtype | format | shape | 介绍 |
|------|------|-------|--------|-------|------|
| input_predict | 输入 | fp16/fp32/bf16 | ND | all | 输入张量 x，即前向传播的predict |
| input_label | 输入 | fp16/fp32/bf16 | ND | all | 	标签张量 y，与input_predict进行广播 |
| input_dout | 输入 | fp16/fp32/bf16 | ND | all | 	反向传播梯度输入，即上一层的梯度 |
| reduction | 属性 | string | - | - | 规约模式，可选"none"或"mean"或"sum"，默认"mean" |
| output_gdient | 输出 | fp16/fp32/bf16 | ND | 广播后shape | 输出梯度，SoftMarginLoss对predict的梯度 |

相关约束
Atlas A2 训练系列产品/Atlas 800I A2推理产品支持float16、float32、bfloat16
张量最大维度数为 8

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现SoftMarginLossGrad算子，支持float16、float32、bfloat16数据类型

## 需求拆解

1. 支持float16、float32、bfloat16数据类型
1. 性能不低于TBE版本

# 详细设计（required）

## 算子分析

### 数学公式

$$
\frac{\partial \text{Loss}}{\partial x} = 
\begin{cases} 
\dfrac{-y \cdot e^{-xy}}{1 + e^{-xy}} \cdot \text{dout}, & \text{reduction}=\text{"none"} \\[8pt]
\dfrac{-y \cdot e^{-xy}}{(1 + e^{-xy}) \cdot N} \cdot \text{dout}, & \text{reduction}=\text{"mean"}
\end{cases}
$$

其中 $N$ 为predict的元素总数（$N = \prod_{i} \text{predict\_shape}_i$），当reduction为"mean"时需要除以元素个数进行归一化。

### 支持数据类型

float16、float32、bfloat16

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

由于输入形状相同且不支持广播，算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。

任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：tileBlockNum 和 tileDataNum 计算单次搬运的数据量，通过 finalSmallTileNum 和 finalBigTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。


##### 1. 分核策略：
优先使用满核的原则。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；

如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

##### 2. 数据分块和内存优化策略：
充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。

Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM，计算出每个Tile块的数据数量。

数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。

设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到SoftMarginLossGradTilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

##### 3. tilingkey规划策略：

tilingkey固定为0，无需分支感知,在kernel侧做运行时分支。

#### 3.2.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。
对于 bfloat16，float16 数据类型，统一转换为 float32 进行计算；
对于 float32 数据类型，直接计算。
无广播填充需求，直接在Compute阶段执行元素级运算。
根据tilingkey（固定）执行核函数。

Ascend C的SoftMarginLossGrad算子流程见下图。
![asc1.png](https://raw.gitcode.com/user-images/assets/7665709/27a6ec3a-1dc3-45f9-a6e9-208efe4cb0df/asc1.png 'asc1.png')

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