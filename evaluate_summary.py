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
- **Network Identifiers:** IP Address (e.g., 192.168.1.100), Hostname/Domain (e.g., server01.local), URL, Port Number (e.g., 443), Protocol (e.g., TCP).
- **System Identifiers:** Username (e.g., root, svc_app), Process ID (PID) (e.g., 12345), Service Name (e.g., sshd), Device/Host ID.
- **Resource Identifiers:** File Path (e.g., /var/log/auth.log), Job/Task/Component ID (e.g., job_123, disk_sda1).
- **Status & Codes:** Log Level (e.g., ERROR, WARN), Error Code/Status Code (e.g., 500, 404, 0xc0000005), Event ID (e.g., 4625).
- **Security Artifacts:** CVE ID (e.g., CVE-2023-1234), Malware Name, Alert Type/Signature (e.g., 'SQL Injection Attempt', 'Brute Force Login').
- **Event Description / Issue Type:** Key terms or phrases describing the event, error, or behavior (e.g., 'Connection timed out', 'Authentication failed', 'Disk full', 'Service stopped', 'Component unavailable', 'HTTP timeout', 'Slow response time', 'Missing device').
- **Key Parameters / Values:** Specific configuration settings or important data values mentioned (e.g., 'threshold=90%', 'user_role=admin', 'request_size=10MB').

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
    
    # 注意：原来的score方法已被移除，相关逻辑已移至main函数
    # 这样可以确保所有摘要使用相同的关键短语和问题进行评估
    
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
        # 计算摘要简洁度评分：1 - (摘要与原文较短者的长度 / 原文长度)，值越大表示摘要越简洁
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

def format_results(results):
    """格式化多个摘要的评分结果"""
    result = "日志摘要评估结果对比\n" + "="*30 + "\n\n"
    
    # 按照评分从高到低排序
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    
    # 添加摘要评分对比表格
    result += "摘要评分对比:\n"
    result += "{:<20} {:<10} {:<15} {:<15}\n".format("摘要文件", "总评分", "问答评分", "简洁度评分")
    result += "-"*60 + "\n"
    
    for score, scores, _, _, summary_file in sorted_results:
        file_name = os.path.basename(summary_file)
        qa_score = scores.get('qa_score', 0)
        conciseness = scores.get('conciseness', 0)
        result += "{:<20} {:<10.4f} {:<15.4f} {:<15.4f}\n".format(
            file_name, score, qa_score, conciseness
        )
    
    # 为每个摘要添加详细信息
    result += "\n" + "="*30 + "\n"
    for score, scores, questions, answers, summary_file in sorted_results:
        file_name = os.path.basename(summary_file)
        result += f"\n摘要文件: {file_name}\n"
        result += f"总评分: {score:.4f}\n"
        result += f"问答评分 (qa_score): {scores.get('qa_score', 0):.4f}\n"
        if 'conciseness' in scores:
            result += f"简洁度评分 (conciseness): {scores['conciseness']:.4f}\n"
        
        result += "\n问题和答案:\n"
        for i, (q, a) in enumerate(zip(questions, answers)):
            result += f"{i+1}. {q} -> {a}\n"
        result += "\n" + "-"*30 + "\n"
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估日志摘要质量')
    parser.add_argument('--log', default='data/ssh.log', help='原始日志文件路径')
    parser.add_argument('--summaries', nargs='+', required=True, help='摘要文件路径列表')
    parser.add_argument('--output', default='evaluation_result.txt', help='评估结果输出文件路径')
    args = parser.parse_args()
    
    # 读取文件
    logger.info(f"读取原始日志文件: {args.log}")
    log_text = read_file(args.log)
    if not log_text:
        logger.error("原始日志文件为空或无法读取")
        return
    
    # 创建评估器
    evaluator = SummarizationScore()
    
    # 首先提取关键短语和生成问题，确保所有摘要使用相同的评估标准
    logger.info("提取关键短语和生成问题...")
    keyphrases_response = asyncio.run(evaluator.extract_keyphrases_prompt.generate(log_text))
    keyphrases = keyphrases_response.keyphrases
    
    questions_gen = asyncio.run(evaluator.question_generation_prompt.generate(log_text, keyphrases))
    questions = questions_gen.questions[:10]
    
    logger.info(f"已提取 {len(keyphrases)} 个关键短语，生成 {len(questions)} 个问题")
    
    # 评估所有摘要文件
    results = []
    
    for summary_file in args.summaries:
        logger.info(f"读取摘要文件: {summary_file}")
        summary_text = read_file(summary_file)
        if not summary_text:
            logger.error(f"摘要文件 {summary_file} 为空或无法读取")
            continue
            
        # 评估摘要
        logger.info(f"开始评估摘要: {summary_file}...")
        
        # 只获取答案，使用相同的问题
        answers_gen = asyncio.run(evaluator.answer_generation_prompt.generate(summary_text, questions))
        answers = answers_gen.answers
        
        # 计算分数
        scores = {}
        qa_score = evaluator._compute_qa_score(answers)
        scores["qa_score"] = qa_score
        if evaluator.length_penalty:
            conciseness_score = evaluator._compute_conciseness_score(log_text, summary_text)
            scores["conciseness"] = conciseness_score
        
        # 计算最终评分
        final_score = evaluator._compute_score(scores)
        
        results.append((final_score, scores, questions, answers, summary_file))
        logger.info(f"摘要 {summary_file} 评估完成，总评分: {final_score:.4f}")
    
    if not results:
        logger.error("没有成功评估任何摘要文件")
        return
    
    # 格式化结果
    result = format_results(results)
    
    # 输出结果
    logger.info(f"所有摘要评估完成，共评估了 {len(results)} 个摘要文件")
    logger.info(f"将结果写入: {args.output}")
    if write_file(args.output, result):
        logger.info(f"结果已保存到: {args.output}")
    
    # 打印结果摘要
    print("\n评估结果摘要:")
    print(f"共评估了 {len(results)} 个摘要文件")
    
    # 按评分排序并显示前三名
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    for i, (score, scores, _, _, summary_file) in enumerate(sorted_results[:3]):
        print(f"\n第 {i+1} 名: {os.path.basename(summary_file)}")
        print(f"总评分: {score:.4f}")
        print(f"问答评分: {scores.get('qa_score', 0):.4f}")
        if 'conciseness' in scores:
            print(f"简洁度评分: {scores['conciseness']:.4f}")

if __name__ == "__main__":
    main()