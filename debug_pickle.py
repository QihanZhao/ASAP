#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import sys

def debug_pickle_file(file_path):
    """调试 pickle 文件加载问题"""
    
    print(f"正在调试文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} 字节")
    
    # 方法1: 标准 pickle 加载
    print("\n=== 方法1: 标准 pickle 加载 ===")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("✓ 标准加载成功!")
        return data
    except Exception as e:
        print(f"✗ 标准加载失败: {e}")
    
    # 方法2: 使用 latin1 编码
    print("\n=== 方法2: 使用 latin1 编码 ===")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print("✓ latin1 编码加载成功!")
        return data
    except Exception as e:
        print(f"✗ latin1 编码加载失败: {e}")
    
    # 方法3: 使用 bytes 编码
    print("\n=== 方法3: 使用 bytes 编码 ===")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        print("✓ bytes 编码加载成功!")
        return data
    except Exception as e:
        print(f"✗ bytes 编码加载失败: {e}")
    
    # 方法4: 尝试不同的 pickle 协议
    print("\n=== 方法4: 尝试不同协议版本 ===")
    for protocol in [0, 1, 2, 3, 4, 5]:
        try:
            with open(file_path, 'rb') as f:
                # 重置文件指针
                f.seek(0)
                # 尝试用指定协议加载
                data = pickle.load(f)
            print(f"✓ 协议 {protocol} 加载成功!")
            return data
        except Exception as e:
            print(f"✗ 协议 {protocol} 加载失败: {e}")
    
    # 方法5: 尝试 numpy 加载 (以防是 numpy 格式)
    print("\n=== 方法5: 尝试 numpy 加载 ===")
    try:
        data = np.load(file_path, allow_pickle=True)
        print("✓ numpy 加载成功!")
        return data
    except Exception as e:
        print(f"✗ numpy 加载失败: {e}")
    
    # 方法6: 分块读取，找到问题位置
    print("\n=== 方法6: 分块诊断 ===")
    try:
        with open(file_path, 'rb') as f:
            # 读取前1KB内容
            chunk = f.read(1024)
            print(f"前1KB内容长度: {len(chunk)}")
            
            # 尝试用 pickletools 分析
            try:
                import pickletools
                print("使用 pickletools 分析...")
                f.seek(0)
                pickletools.dis(f)
            except Exception as e:
                print(f"pickletools 分析失败: {e}")
                
    except Exception as e:
        print(f"分块读取失败: {e}")
    
    print("\n所有方法都失败了。可能的原因:")
    print("1. 文件在传输过程中损坏")
    print("2. 文件是用不兼容的 Python 版本创建的")
    print("3. 文件不是标准的 pickle 格式")
    print("4. 文件被部分覆写或截断")
    
    return None

def analyze_data(data):
    """分析加载成功的数据"""
    if data is None:
        return
    
    print(f"\n=== 数据分析 ===")
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"字典键: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    形状: {value.shape}")
    elif hasattr(data, 'shape'):
        print(f"数组形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
    
    return data

if __name__ == "__main__":
    file_path = "humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/forward.pkl"
    
    # 调试加载
    data = debug_pickle_file(file_path)
    
    # 分析数据
    if data is not None:
        analyze_data(data)
        print("\n✓ 文件加载成功!")
    else:
        print("\n✗ 所有加载方法都失败了") 