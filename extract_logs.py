#!/usr/bin/env python3

import json
import os
import sys
import traceback
import shutil

def extract_logs(json_file, output_dir):
    """从JSON文件中提取日志并保存为适合dpp.py和evaluate_summary.py的格式
    
    Args:
        json_file: 输入的JSON文件路径
        output_dir: 输出目录路径
    """
    # 删除并重新创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 读取JSON文件
    with open(json_file, 'r') as f:
        logs = json.load(f)
    
    print(f"共读取到 {len(logs)} 条日志记录")
    
    # 处理每个日志条目
    for i, log_entry in enumerate(logs):
        try:
            # 检查log_entry是否为字典类型并包含必要的字段
            if isinstance(log_entry, dict) and 'raw_log' in log_entry and 'levels' in log_entry and 'time' in log_entry:
                # 为每个日志条目创建一个子目录
                log_dir = os.path.join(output_dir, f"log_{i}_0")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                # 创建raw.log文件
                raw_log_path = os.path.join(log_dir, "raw.log")
                with open(raw_log_path, 'w', encoding='utf-8') as f:
                    # 将time、levels和raw_log按行拼接
                    for time, level, message in zip(log_entry['time'], log_entry['levels'], log_entry['raw_log']):
                        f.write(f"{time} {level} {message}\n")
                
                print(f"已创建 {raw_log_path}")
                
                # 创建summary.txt文件
                summary_path = os.path.join(log_dir, "summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    if 'summary' in log_entry:
                        f.write(log_entry['summary'])
                    else:
                        # 如果没有summary字段，使用第一条日志消息作为摘要
                        f.write(log_entry['raw_log'][0] if log_entry['raw_log'] else "无摘要")
                    
                print(f"已创建 {summary_path}")
            else:
                # 如果log_entry不是预期的格式，尝试处理嵌套结构
                if isinstance(log_entry, list):
                    for j, nested_entry in enumerate(log_entry):
                        try:
                            # 为每个嵌套日志条目创建一个子目录
                            log_dir = os.path.join(output_dir, f"log_{i}_{j}")
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                            
                            if isinstance(nested_entry, dict) and 'raw_log' in nested_entry and 'levels' in nested_entry and 'time' in nested_entry:
                                # 创建raw.log文件
                                raw_log_path = os.path.join(log_dir, "raw.log")
                                with open(raw_log_path, 'w', encoding='utf-8') as f:
                                    # 将time、levels和raw_log按行拼接
                                    for time, level, message in zip(nested_entry['time'], nested_entry['levels'], nested_entry['raw_log']):
                                        f.write(f"{time} {level} {message}\n")
                                
                                print(f"已创建 {raw_log_path}")
                                
                                # 创建summary.txt文件
                                summary_path = os.path.join(log_dir, "summary.txt")
                                with open(summary_path, 'w', encoding='utf-8') as f:
                                    if 'summary' in nested_entry:
                                        f.write(nested_entry['summary'])
                                    else:
                                        # 如果没有summary字段，使用第一条日志消息作为摘要
                                        f.write(nested_entry['raw_log'][0] if nested_entry['raw_log'] else "无摘要")
                                
                                print(f"已创建 {summary_path}")
                            else:
                                # 如果嵌套条目不是预期的格式，将其作为摘要处理
                                print(f"警告: log_{i}_{j} 不是预期的格式，将其作为摘要处理")
                                summary_path = os.path.join(log_dir, "summary.txt")
                                with open(summary_path, 'w', encoding='utf-8') as f:
                                    f.write(str(nested_entry))
                                print(f"已创建 {summary_path}")
                        except Exception as e:
                            print(f"处理嵌套日志 {i}_{j} 时出错: {e}")
                            traceback.print_exc()
                else:
                    # 如果log_entry既不是字典也不是列表，将其作为摘要处理
                    log_dir = os.path.join(output_dir, f"log_{i}_0")
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    
                    print(f"警告: log_{i}_0 不是预期的格式，将其作为摘要处理")
                    summary_path = os.path.join(log_dir, "summary.txt")
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(str(log_entry))
                    print(f"已创建 {summary_path}")
        except Exception as e:
            print(f"处理日志 {i} 时出错: {e}")
            traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("用法: python extract_logs.py <json_file> [output_dir]")
        print("示例: python extract_logs.py data/Apache/test.json extracted_logs")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_logs"
    
    extract_logs(json_file, output_dir)
    print(f"已将日志从 {json_file} 提取到 {output_dir} 目录")

if __name__ == "__main__":
    main()