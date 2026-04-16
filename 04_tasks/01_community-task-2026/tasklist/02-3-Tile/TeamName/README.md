# Tile

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：沿指定维度复制扩展张量。

- 计算公式：

$$
y = Tile(x, multiples)
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>  
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_tile.cpp](./examples/test_aclnn_tile.cpp) | 通过[test_aclnn_tile](./docs/aclnnTile.md)接口方式调用Tile算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| TeamName | 待填写 | Tile | 待填写 | Tile算子适配开源仓 |
