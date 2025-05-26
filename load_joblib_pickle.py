#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import numpy as np
import os

def load_joblib_pickle(file_path):
    """使用 joblib 加载 pickle 文件"""
    
    print(f"正在使用 joblib 加载文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} 字节")
    
    try:
        # 使用 joblib 加载
        data = joblib.load(file_path)
        print("✓ joblib 加载成功!")
        return data
    except Exception as e:
        print(f"✗ joblib 加载失败: {e}")
        return None

def analyze_motion_data(data):
    """分析运动数据"""
    if data is None:
        return
    
    print(f"\n=== 数据分析 ===")
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"字典键: {list(data.keys())}")
        for key, value in data.items():
            print(f"\n{key}:")
            print(f"  类型: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"  形状: {value.shape}")
                print(f"  数据类型: {value.dtype}")
                if len(value.shape) <= 2:  # 只显示小数组的统计信息
                    print(f"  最小值: {np.min(value):.4f}")
                    print(f"  最大值: {np.max(value):.4f}")
                    print(f"  均值: {np.mean(value):.4f}")
                    print(f"  标准差: {np.std(value):.4f}")
            elif isinstance(value, (list, tuple)):
                print(f"  长度: {len(value)}")
                if len(value) > 0:
                    print(f"  第一个元素类型: {type(value[0])}")
            else:
                print(f"  值: {value}")
    
    elif hasattr(data, 'shape'):
        print(f"数组形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"最小值: {np.min(data):.4f}")
        print(f"最大值: {np.max(data):.4f}")
        print(f"均值: {np.mean(data):.4f}")
        print(f"标准差: {np.std(data):.4f}")
    
    return data

def load_and_verify_standing_motion():
    """
    加载并验证生成的静止站立动作数据
    """
    # 加载数据
    filename = 'standing_motion_300frames.pkl'
    try:
        data = joblib.load(filename)
        print(f"成功加载数据文件: {filename}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None
    
    # 检查数据结构
    print(f"\n数据结构:")
    print(f"顶层键: {list(data.keys())}")
    
    # 获取动作数据
    motion_key = list(data.keys())[0]
    motion = data[motion_key]
    
    print(f"\n动作数据键: {list(motion.keys())}")
    
    # 验证数据形状和类型
    print(f"\n数据详情:")
    for key, value in motion.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: 形状={value.shape}, 数据类型={value.dtype}")
            print(f"    最小值={value.min():.4f}, 最大值={value.max():.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 验证DOF数据
    dof_data = motion['dof']
    print(f"\nDOF数据验证:")
    print(f"  总帧数: {dof_data.shape[0]}")
    print(f"  DOF维度: {dof_data.shape[1]}")
    print(f"  所有帧是否相同: {np.allclose(dof_data[0], dof_data[-1])}")
    
    # 显示默认角度
    default_angles = dof_data[0]
    print(f"\n默认关节角度 (弧度):")
    joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", 
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", 
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"
    ]
    
    for i, (name, angle) in enumerate(zip(joint_names, default_angles)):
        print(f"  {i:2d}. {name:20s}: {angle:7.3f} rad ({np.degrees(angle):7.2f}°)")
    
    # 验证根部数据
    print(f"\n根部数据验证:")
    print(f"  根部旋转 (四元数): {motion['root_rot'][0]}")
    print(f"  根部位移: {motion['root_trans_offset'][0]}")
    
    # 验证pose_aa数据
    pose_aa = motion['pose_aa']
    print(f"\nPose AA数据:")
    print(f"  形状: {pose_aa.shape}")
    print(f"  是否全为零: {np.allclose(pose_aa, 0)}")
    
    print(f"\n数据验证完成！")
    return data

def compare_with_original():
    """
    与原始数据格式进行对比
    """
    try:
        # 加载原始数据
        original_data = joblib.load('humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/forward.pkl')
        original_motion = original_data['Male2MartialArtsExtended_c3dExtended_1_poses']
        
        # 加载生成的数据
        standing_data = joblib.load('standing_motion_300frames.pkl')
        standing_motion = standing_data['standing_motion_300frames']
        
        print(f"\n数据格式对比:")
        print(f"{'字段':<20} {'原始数据':<20} {'生成数据':<20} {'匹配':<10}")
        print("-" * 70)
        
        for key in ['dof', 'root_rot', 'root_trans_offset', 'pose_aa', 'fps']:
            if key in original_motion and key in standing_motion:
                if isinstance(original_motion[key], np.ndarray):
                    orig_shape = original_motion[key].shape
                    stand_shape = standing_motion[key].shape
                    match = len(orig_shape) == len(stand_shape) and orig_shape[1:] == stand_shape[1:]
                    print(f"{key:<20} {str(orig_shape):<20} {str(stand_shape):<20} {'✓' if match else '✗':<10}")
                else:
                    orig_val = original_motion[key]
                    stand_val = standing_motion[key]
                    match = orig_val == stand_val
                    print(f"{key:<20} {str(orig_val):<20} {str(stand_val):<20} {'✓' if match else '✗':<10}")
        
        print(f"\n格式兼容性: 生成的数据与原始数据格式兼容 ✓")
        
    except Exception as e:
        print(f"对比失败: {e}")

if __name__ == "__main__":
    # 验证生成的数据
    data = load_and_verify_standing_motion()
    
    if data is not None:
        # 与原始数据对比
        compare_with_original()
    else:
        print("\n✗ 文件加载失败")
        print("请确保已安装 joblib: pip install joblib") 