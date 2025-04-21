# 日志摘要评估工具

这个工具用于评估日志摘要的质量。它通过分析原始日志和摘要之间的信息保留程度，以及摘要的简洁性来给出评分。原始思路出自 [Ragas 开源项目的 SummarizationScore](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/summarization_score/)。

## 功能特点

- 自动提取日志中的关键信息（IP地址、用户名、错误类型等，定义来自[LoFI 论文](https://arxiv.org/pdf/2409.13561)的 FIP/FID 概念）
- 生成基于关键信息的问题集
- 评估摘要是否包含这些关键信息的答案
- 计算信息保留分数和简洁度分数
- 生成评估报告

## 使用方法

```bash
python3 evaluate_summary.py --log <日志文件路径> --summary <摘要文件路径> --output <输出文件路径>
```

### 参数说明

- `--log`: 原始日志文件路径，默认为 `ssh.log`
- `--summary`: 摘要文件路径，默认为 `dpp_summary.txt`
- `--output`: 评估结果输出文件路径，默认为 `evaluation_result.txt`

### 示例

```bash
python3 evaluate_summary.py --log data/ssh.log --summary res/trae_summary.txt --output res/trae_eval_result.txt
```

## 评分标准

评分由两部分组成：

1. **问答评分 (qa_score)**: 衡量摘要保留原始日志关键信息的程度，范围0-1
2. **简洁度评分 (conciseness_score)**: 衡量摘要相对于原始日志的简洁程度，范围0-1

最终评分是这两个分数的加权平均，默认权重为：
- 问答评分权重: 0.5
- 简洁度评分权重: 0.5

## 输出结果

评估结果包含以下内容：

- 总评分
- 问答评分
- 简洁度评分
- 用于评估的问题列表及其答案（1表示摘要包含该信息，0表示不包含）

### 示例输出
```
总评分: 0.82
问答评分: 0.85
简洁度评分: 0.79

评估问题：
1. 日志中是否包含来自173.234.31.186的登录尝试？ [1]
2. 是否有使用webmaster用户名的登录尝试？ [1]
3. 是否记录到亚马逊AWS中国区域的IP地址？ [1]
4. 是否提及SSH协议版本？ [0]
5. 是否包含时间范围信息？ [1]
```

## 模型评估对比

| 模型 | 问答评分 | 简洁度评分 | 综合评分 |
|------|---------|-----------|--------|
| DPP | 0.78    | 0.82      | 0.80   |
| Minimax | 0.81   | 0.75      | 0.78   |
| Trae | 0.85    | 0.79      | 0.82   |

*评估基于ssh.log测试数据集，权重设置为默认值（问答0.5，简洁度0.5）*

## 与dpp.py的集成

大模型直接总结的文本，可以直接保存。
聚类采样总结方案，可参考目录下的 dpp.py 文件。运行dpp.py生成摘要并保存到res/dpp_summary.txt后，可以直接运行evaluate_summary.py来评估摘要质量。

## TODO

后续考虑验证 [LoFI 项目中标注数据](https://github.com/Jun-jie-Huang/LoFI/tree/main/data)的摘要质量，在本方法下的效果。