#!python3
import os
import sys
import requests
import json
import time
import numpy as np
import concurrent.futures
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DPPHandler')

# 模拟原有的Table类
class Table:
    def __init__(self):
        self.rows = []
        self.fields = []
    
    def add_row(self, row_dict):
        self.rows.append(row_dict)
    
    def get_rows(self):
        return self.rows

class DPPHandler:
    k = "3"
    QIANFAN_TOKEN_FILE = '/tmp/spl_dpp_access_token.json'
    QIANFAN_API_KEY = os.getenv('QIANFAN_API_KEY', '填你的百度千帆应用apikey')
    QIANFAN_SECRET_KEY = os.getenv('QIANFAN_SECRET_KEY', '填你的百度千帆应用secretkey')
    DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '填你的阿里云灵积apikey')

    def __init__(self, k=3, by_field=None):
        self.access_token = None
        self.by = by_field
        self.k = str(k)
        logger.info(f"初始化DPP处理器，k={self.k}, by={self.by}")

    def get_access_token(self):
        # 检查文件中是否有有效的access_token
        if os.path.exists(self.QIANFAN_TOKEN_FILE):
            with open(self.QIANFAN_TOKEN_FILE, 'r') as f:
                token_data = json.load(f)
                if time.time() < token_data['expires_at']:
                    logger.info(time.time())
                    self.access_token = token_data['access_token']
                    return

        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.QIANFAN_API_KEY}&client_secret={self.QIANFAN_SECRET_KEY}"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        logger.info('call get qianfan access token')
        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        self.access_token = response_json.get("access_token")
        expires_in = response_json.get("expires_in", 3600)
        expires_at = time.time() + expires_in

        # 将新的access_token和过期时间写入文件
        with open(self.QIANFAN_TOKEN_FILE, 'w') as f:
            json.dump({'access_token': self.access_token, 'expires_at': expires_at}, f)

    def qianfan_summarize(self, content):
        self.get_access_token()
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token={self.access_token}"
        logger.info(url)
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }, ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json'
        }
        logger.info('call get qianfan ernie-speed llm')
        response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
        logger.info(response.json())
        return response.json().get('result','')

    def dashscope_summarize(self, content):
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            'Authorization': f"Bearer {self.DASHSCOPE_API_KEY}",
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'qwen-turbo',
            'input': {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': content
                    }
                ]
            },
            'parameters': {
                'result_format': 'text'
            }
        }
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f'call get dashscope qwen-plus llm (attempt {attempt + 1})')
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                # 添加5秒延迟以避免触发服务端限速
                time.sleep(5)
                return response.json().get('output',{}).get('text','')
            except requests.exceptions.RequestException as e:
                logger.warning(f'Dashscope API调用失败: {str(e)}')
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                logger.error('Dashscope API调用达到最大重试次数')
                return ''

    def llm_summarize(self, content, modelname):
        if modelname == "qwen":
            return self.dashscope_summarize(content)
        else:
            return self.qianfan_summarize(content)

    def dpp_sample(self, S, k):
        # S: similarity matrix
        # k: number of items to sample
        n = S.shape[0]
        # Initialize empty set Y
        Y = set()
        for _ in range(k):
            best_i = -1
            best_p = -1
            for i in range(n):
                if i not in Y:
                    # Compute determinant of submatrix
                    det_Yi = np.linalg.det(S[np.ix_(list(Y) + [i], list(Y) + [i])])
                    # Compute probability of adding i to Y
                    p_add = det_Yi / (1 + det_Yi)
                    if p_add > best_p:
                        best_p = p_add
                        best_i = i
            # Add best item to Y
            Y.add(best_i)
        return list(Y)

    def process_cluster(self, cluster_rows, features):
        # 提取 tfidf 特征值，构建 numpy 数组
        feature_values = np.array([[float(row[feature]) for feature in features] for row in cluster_rows])
        # 使用 sklearn 中的方法构建相似性矩阵
        S = cosine_similarity(feature_values)
        # 应用 DPP 采样
        sampled_indices = self.dpp_sample(S, int(self.k))
        sampled_rows = [cluster_rows[i] for i in sampled_indices]
        # 发送采样日志，由大模型生成摘要
        content_parts = [
            "你是 IT 运维和网络安全专家，请总结下面这段日志内容，输出尽量简短、非结构化、保留关键信息："
        ] + [row['raw_message'] for row in sampled_rows]
        content = "\n".join(content_parts)
        summary = self.llm_summarize(content, 'ernie')
        return summary

    def load_ssh_log(self, log_file):
        """从SSH日志文件加载数据"""
        table = Table()
        table.fields = ['raw_message']
        
        # 读取日志文件
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        # 将每行日志添加到表中
        for line in log_lines:
            table.add_row({'raw_message': line.strip()})
        
        logger.info(f"从{log_file}加载了{len(table.rows)}行日志数据")
        return table

    def extract_features(self, table):
        """从日志中提取TF-IDF特征"""
        # 获取所有原始消息
        raw_messages = [row['raw_message'] for row in table.rows]
        
        # 使用TF-IDF向量化器提取特征
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(raw_messages)
        
        # 获取特征名称
        feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        
        # 将特征添加到表中
        for i, row in enumerate(table.rows):
            for j, feature_name in enumerate(feature_names):
                row[feature_name] = str(tfidf_matrix[i, j])
        
        # 更新表的字段
        table.fields.extend(feature_names)
        
        logger.info(f"提取了{len(feature_names)}个TF-IDF特征")
        return table, feature_names

    def cluster_logs(self, table, feature_names, num_clusters=5):
        """使用简单的聚类方法对日志进行分组"""
        from sklearn.cluster import KMeans
        
        # 提取特征矩阵
        feature_matrix = np.array([[float(row[feature]) for feature in feature_names] for row in table.rows])
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_matrix)
        
        # 将聚类结果添加到表中
        for i, row in enumerate(table.rows):
            row['cluster'] = str(clusters[i])
        
        # 更新表的字段
        if 'cluster' not in table.fields:
            table.fields.append('cluster')
        
        logger.info(f"将日志分为{num_clusters}个聚类")
        return table

    def process_logs(self, log_file, num_clusters=5):
        """处理SSH日志文件并生成摘要"""
        # 加载日志数据
        table = self.load_ssh_log(log_file)
        
        # 提取特征
        table, features = self.extract_features(table)
        
        # 聚类
        table = self.cluster_logs(table, features, num_clusters)
        
        # 按聚类分组
        clusters = {}
        for row in table.rows:
            cluster_id = row['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(row)
        
        # 准备输出表
        result_table = Table()
        result_table.fields = ['cluster', 'summary']
        
        # 对每个聚类应用DPP采样和LLM总结
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_cluster = {
                executor.submit(self.process_cluster, cluster_rows, features): cluster_id
                for cluster_id, cluster_rows in clusters.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    summary = future.result()
                    result_table.add_row({'cluster': cluster_id, 'summary': summary})
                    logger.info(f"聚类{cluster_id}的摘要已生成")
                except Exception as exc:
                    logger.error(f'聚类{cluster_id}生成摘要时出错: {exc}')
        
        # 如果不需要按聚类输出，生成全局摘要
        if not self.by:
            total_table = Table()
            total_table.fields = ['log_summary']
            total_content_parts = [
                "你是 IT 运维和网络安全专家，下面是日志聚类后的关键信息摘要，请通盘考虑，输出中文总结和分析建议："
            ] + [row['summary'] for row in result_table.get_rows()]
            
            total_content = "\n\n## 聚类摘要\n\n".join(total_content_parts)
            total_summary = self.llm_summarize(total_content, 'qwen')
            
            if total_summary is None:
                logger.info("无法生成全局总结，请检查聚类总结内容。")
            
            total_table.add_row({'log_summary': total_summary})
            return total_table
        else:
            return result_table

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SSH日志摘要生成工具')
    parser.add_argument('--log_file', type=str, default='data/ssh.log', help='SSH日志文件路径')
    parser.add_argument('--k', type=int, default=5, help='每个聚类的采样数量')
    parser.add_argument('--clusters', type=int, default=10, help='聚类数量')
    parser.add_argument('--by', action='store_true', help='是否按聚类输出摘要')
    
    args = parser.parse_args()
    
    # 创建DPP处理器
    handler = DPPHandler(k=args.k, by_field=args.by)
    
    # 处理日志
    result_table = handler.process_logs(args.log_file, args.clusters)
    
    # 将摘要写入文件
    output_file = 'dpp_summary.txt'
    logger.info(f"正在将日志摘要写入到 {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if 'log_summary' in result_table.fields:
            f.write(result_table.get_rows()[0]['log_summary'])
        else:
            for row in result_table.get_rows():
                f.write(f"聚类 {row['cluster']}:\n{row['summary']}\n\n")
    
    logger.info(f"日志摘要已成功写入到 {output_file}")

