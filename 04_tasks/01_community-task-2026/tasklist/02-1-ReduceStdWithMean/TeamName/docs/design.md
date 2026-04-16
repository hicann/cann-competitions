# ReduceStdWithMean 算子设计文档

## 1. 算子功能
ReduceStdWithMean 算子用于计算输入张量在指定维度上的标准差和均值，返回两个输出张量。

### 数学公式
给定输入张量 $x$，在维度 $d$ 上计算：

**均值**：
$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

**标准差**：
$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$

当 `unbiased=true` 时，使用无偏估计：
$$\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2}$$

## 2. 算子设计

### 2.1 整体架构
```
┌─────────────────────────────────────────────────────────┐
│                    Host 端                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ 参数校验    │ -> │ 形状推导    │ -> │ Tiling策略  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Device 端 (AI Core)                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ 数据搬运    │ -> │ 并行计算    │ -> │ 结果写回    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Tiling 策略
根据输入张量的形状和数据量，选择不同的 Tiling 策略：

| 场景 | Tiling 策略 | 说明 |
|------|-------------|------|
| 小数据量 | 单核处理 | 数据量小于 UB 容量 |
| 中等数据量 | 多核均分 | 按维度均分到各核 |
| 大数据量 | 流水处理 | 分批次处理数据 |

### 2.3 内存管理
- **Unified Buffer (UB)**：存储中间计算结果
- **Global Memory**：存储输入输出张量
- **临时内存**：存储部分和结果

## 3. 算子实现

### 3.1 Host 端实现
```cpp
// 参数校验
int CheckParams(const aclTensor* self, int64_t dim,
                const aclTensor* stdOut, const aclTensor* meanOut);

// 形状推导
void InferShape(const std::vector<int64_t>& inShape,
                int64_t dim, bool keepdim,
                std::vector<int64_t>& outShape);

// Tiling 策略选择
TilingData SelectTilingStrategy(const aclTensor* self, int64_t dim);
```

### 3.2 Kernel 端实现
```cpp
// 核函数
__global__ void ReduceStdWithMeanKernel(
    GM_ADDR x, GM_ADDR std_out, GM_ADDR mean_out,
    TilingData tiling);

// 计算逻辑
void ComputeReduceStdWithMean(
    LocalTensor<float>& xLocal,
    LocalTensor<float>& stdLocal,
    LocalTensor<float>& meanLocal,
    const TilingData& tiling);
```

## 4. 性能优化

### 4.1 向量化计算
- 使用 Ascend C 内置向量指令
- 充分利用 AI Core 计算能力

### 4.2 内存优化
- 减少内存搬运次数
- 优化数据布局
- 复用 UB 空间

### 4.3 并行优化
- 多核并行计算
- 计算与搬运流水

## 5. 测试方案

### 5.1 功能测试
- 不同维度测试
- 不同数据类型测试
- 边界条件测试

### 5.2 精度测试
- 与 CPU 参考实现对比
- 相对误差 < 1e-5

### 5.3 性能测试
- 不同数据规模性能
- 与基准实现对比

## 6. 约束与限制
- 最大支持 8 维张量
- 单次计算数据量不超过 2GB
- 仅支持连续内存张量
