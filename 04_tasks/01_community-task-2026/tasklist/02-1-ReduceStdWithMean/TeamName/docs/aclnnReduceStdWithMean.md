# aclnnReduceStdWithMean API 文档

## 功能描述
计算输入张量在指定维度上的标准差和均值。

## 函数原型
```cpp
aclnnStatus aclnnReduceStdWithMean(
    aclTensor* self,
    int64_t dim,
    bool unbiased,
    bool keepdim,
    aclTensor* std_out,
    aclTensor* mean_out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

## 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| self | aclTensor* | 输入张量 |
| dim | int64_t | 计算标准差的维度 |
| unbiased | bool | 是否使用无偏估计 |
| keepdim | bool | 是否保留缩减维度 |
| std_out | aclTensor* | 输出标准差张量 |
| mean_out | aclTensor* | 输出均值张量 |
| workspaceSize | uint64_t* | 工作空间大小 |
| executor | aclOpExecutor** | 算子执行器 |

## 返回值
- `ACLNN_SUCCESS`：执行成功
- 其他错误码：执行失败

## 支持的数据类型
- ACL_FLOAT
- ACL_FLOAT16
- ACL_DOUBLE

## 约束条件
1. 输入张量维度必须大于0
2. dim 必须在有效范围内 [-rank, rank-1]
3. 输出张量形状必须与计算结果形状匹配

## 示例代码
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_reduce_std_with_mean.h"

// 创建输入张量
std::vector<int64_t> selfShape = {3, 4};
aclTensor* self = nullptr;
// ... 创建张量代码

// 创建输出张量
std::vector<int64_t> outShape = {3};  // keepdim=false
aclTensor* stdOut = nullptr;
aclTensor* meanOut = nullptr;
// ... 创建张量代码

// 计算工作空间大小
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnReduceStdWithMeanGetWorkspaceSize(
    self, 1, true, false, stdOut, meanOut, &workspaceSize, &executor);

// 执行算子
void* workspace = nullptr;
if (workspaceSize > 0) {
    workspace = aclrtMalloc(workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
ret = aclnnReduceStdWithMean(workspace, workspaceSize, executor, stream);
```

## 性能说明
- 算子针对昇腾 AI Core 进行了优化
- 支持多核并行计算
- 内存访问模式优化

## 版本支持
- CANN 8.0 及以上版本
