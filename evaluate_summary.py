#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import typing as t
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv
import re
import time
import json
import asyncio
import aiohttp
from openai import OpenAI

# 加载.env文件中的环境变量
load_dotenv()

from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SummaryEvaluator')

# 通用 Dashscope API 调用函数（改为 openai 兼容接口）
def _call_dashscope_api(api_key, messages, default_return=None):
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    try:
        completion = client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
            response_format={"type": "json_object"}
        )
        result = completion.choices[0].message.content.strip()
        try:
            return json.loads(result)
        except Exception as e:
            logger.error(f"Dashscope返回内容无法解析为JSON: {result}")
            return default_return
    except Exception as e:
        logger.error(f"Dashscope API 调用失败: {e}")
        return default_return

# 从原始ragas_summarization.py导入的类和函数
class StringIO(BaseModel):
    text: str

class ExtractedKeyphrases(BaseModel):
    keyphrases: List[str]

class QuestionsGenerated(BaseModel):
    questions: List[str]

class AnswersGenerated(BaseModel):
    answers: List[int]

class GenerateQuestionsPromptInput(BaseModel):
    text: str
    keyphrases: List[str]

class SummaryAndQuestions(BaseModel):
    summary: str
    questions: List[str]

class ExtractKeyphrasePrompt:
    def __init__(self):
        self.name = "extract_keyphrases"
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
        
    def dashscope_extract(self, content):
        messages = [
            {
                'role': 'system',
                'content': """You are an IT operations and security expert.
Extract keyphrases from the following log text that are crucial for understanding IT operations and security events.
Identify entities of the following types:
- Error Message: directly describes a failed action or an exception raised from a software stack.
- Missing Component: means some components are unavailable such as devices, tasks and hosts.
- Abnormal Behavior: indicates the degraded performance of an application e.g., HTTP timeout, slow response time.
- Wrong Status: means a specific response code is incorporated to explain the wrong event, e.g., status code, error flags.
- Address: includes a concrete URL of HTTP requests, IP address or paths to a folder.
- Component ID: records the index for a system component e.g., job ID, task ID, service ID.
- Parameter Name: shows the key and value for a parameter e.g., data name, user name.


Return only a JSON list of keyphrases.
"""
            },
            {
                'role': 'user',
                'content': content
            }
        ]
        return _call_dashscope_api(self.DASHSCOPE_API_KEY, messages, default_return=[])
    
    async def generate(self, text):
        # 使用Qwen大模型API提取关键短语
        keyphrases = self.dashscope_extract(text)
        if not keyphrases:
            logger.warning('关键短语提取失败')
        return ExtractedKeyphrases(keyphrases=keyphrases)

class GenerateQuestionsPrompt:
    def __init__(self):
        self.name = "generate_questions"
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
    
    def dashscope_generate_questions(self, text, keyphrases):
        messages = [
            {
                'role': 'system',
                'content': 'You are an IT operations expert. Based on the given log text and keyphrases, generate 10 closed-ended questions that can be answered with 1 (present in text) or 0 (not present). Return only a JSON list of questions.'
            },
            {
                'role': 'user',
                'content': f"Log text: {text}\nKeyphrases: {', '.join(keyphrases)}"
            }
        ]
        return _call_dashscope_api(self.DASHSCOPE_API_KEY, messages, default_return=[])
    
    async def generate(self, text, keyphrases):
        questions = self.dashscope_generate_questions(text, keyphrases)
        if not questions:
            logger.warning('问题生成失败')
        # 修复点：从字典中提取question字段的值，并限制10个问题
        valid_questions = [q.get("question") for q in questions if isinstance(q, dict)]
        return QuestionsGenerated(questions=valid_questions[:10])

class GenerateAnswersPrompt:
    def __init__(self):
        self.name = "generate_answers"
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
    
    def dashscope_generate_answers(self, summary, questions):
        messages = [
            {
                'role': 'system',
                'content': 'You are an IT operations expert. For each question, answer 1 if the summary contains sufficient information to answer it, otherwise 0. Return only a JSON list of answers.'
            },
            {
                'role': 'user',
                'content': f"Summary: {summary}\nQuestions: {', '.join(questions)}"
            }
        ]
        default_return = ['0'] * len(questions)
        return _call_dashscope_api(self.DASHSCOPE_API_KEY, messages, default_return=default_return)
    
    async def generate(self, summary, questions):
        answers = self.dashscope_generate_answers(summary, questions)
        return AnswersGenerated(answers=answers)

@dataclass
class SummarizationScore:
    name: str = "summary_score"
    max_retries: int = 1
    length_penalty: bool = True
    coeff: float = 0.5
    
    def __init__(self):
        self.extract_keyphrases_prompt = ExtractKeyphrasePrompt()
        self.question_generation_prompt = GenerateQuestionsPrompt()
        self.answer_generation_prompt = GenerateAnswersPrompt()
    
    async def score(self, text, summary):
        # 提取关键短语
        keyphrases_response = await self.extract_keyphrases_prompt.generate(text)
        keyphrases = keyphrases_response.keyphrases
        
        # 动态生成问题
        questions_gen = await self.question_generation_prompt.generate(text, keyphrases)
        questions = questions_gen.questions[:10]
        
        # 获取答案
        answers_gen = await self.answer_generation_prompt.generate(summary, questions)
        answers = answers_gen.answers

        # 计算分数
        scores = {}
        qa_score = self._compute_qa_score(answers)
        scores["qa_score"] = qa_score
        if self.length_penalty:
            conciseness_score = self._compute_conciseness_score(text, summary)
            scores["conciseness"] = conciseness_score # 使用 'conciseness' 作为键
        
        # 计算最终评分
        final_score = self._compute_score(scores)
        
        return final_score, scores, questions, answers
    
    def _compute_score(self, scores):
        # 与 ragas_summarization.py 保持一致
        return (
            scores["qa_score"] * (1 - self.coeff)
            + scores.get("conciseness", 0) * self.coeff
        )
    
    def _compute_qa_score(self, answers):
        correct = sum([1 for a in answers if a == 1])
        return correct / len(answers) if answers else 0
    
    def _compute_conciseness_score(self, text, summary):
        return 1 - min(len(summary), len(text)) / (len(text) + 1e-10)

def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败: {e}")
        return ""

def write_file(file_path, content):
    """写入文件内容"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"写入文件 {file_path} 失败: {e}")
        return False

def format_results(score, scores, questions, answers):
    """格式化评分结果"""
    result = f"总评分: {score:.4f}\n"
    result += f"问答评分 (qa_score): {scores.get('qa_score', 0):.4f}\n" # 使用 get 以防万一
    if 'conciseness' in scores:
        result += f"简洁度评分 (conciseness): {scores['conciseness']:.4f}\n"
    
    result += "\n问题和答案:\n"
    for i, (q, a) in enumerate(zip(questions, answers)):
        result += f"{i+1}. {q} -> {a}\n"
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估日志摘要质量')
    parser.add_argument('--log', default='data/ssh.log', help='原始日志文件路径')
    parser.add_argument('--summary', default='dpp_summary.txt', help='摘要文件路径')
    parser.add_argument('--output', default='evaluation_result.txt', help='评估结果输出文件路径')
    args = parser.parse_args()
    
    # 读取文件
    logger.info(f"读取原始日志文件: {args.log}")
    log_text = read_file(args.log)
    if not log_text:
        logger.error("原始日志文件为空或无法读取")
        return
    
    logger.info(f"读取摘要文件: {args.summary}")
    summary_text = read_file(args.summary)
    if not summary_text:
        logger.error("摘要文件为空或无法读取")
        return
    
    # 评估摘要
    logger.info("开始评估摘要...")
    evaluator = SummarizationScore()
    # 使用 asyncio.run 来运行异步的 score 方法
    score, scores, questions, answers = asyncio.run(evaluator.score(log_text, summary_text))
    
    # 格式化结果
    result = format_results(score, scores, questions, answers)
    
    # 输出结果
    logger.info(f"评估完成，总评分: {score:.4f}")
    logger.info(f"将结果写入: {args.output}")
    if write_file(args.output, result):
        logger.info(f"结果已保存到: {args.output}")
    
    # 打印结果摘要
    print("\n评估结果摘要:")
    print(f"总评分: {score:.4f}")
    print(f"问答评分 (qa_score): {scores.get('qa_score', 0):.4f}")
    if 'conciseness' in scores:
        print(f"简洁度评分 (conciseness): {scores['conciseness']:.4f}")

if __name__ == "__main__":
    main()