import numpy as np
import joblib
from scipy.spatial.transform import Rotation as R

def create_standing_motion():
    """
    创建300帧的静止站立动作数据
    """
    # 默认角度 (23个DOF)
    default_angles = np.array([
        -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,  # 左腿 (6个DOF)
        -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,  # 右腿 (6个DOF)
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  # 腰部 (3个DOF) + 左臂 (3个DOF)
         0.0,  0.0,  0.0,  0.0,  0.0         # 右臂 (5个DOF)
    ], dtype=np.float32)
    
    # 参数设置
    num_frames = 300
    fps = 30
    
    # 创建DOF数据 - 所有帧都使用相同的默认角度
    dof_data = np.tile(default_angles, (num_frames, 1))
    
    # 创建根部旋转数据 - 保持直立 (单位四元数)
    # 四元数格式: [x, y, z, w]
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    root_rot_data = np.tile(identity_quat, (num_frames, 1))
    
    # 创建根部位移数据 - 保持在原点
    root_trans_offset = np.zeros((num_frames, 3), dtype=np.float32)
    root_trans_offset[:,2] = 0.79
    
    # 创建pose_aa数据 (轴角表示法)
    # 这里我们需要将关节角度转换为轴角表示
    # 对于静止站立，大部分关节的轴角都是零
    pose_aa_data = np.zeros((num_frames, 27, 3), dtype=np.float32)
    
    # 构建数据字典
    motion_data = {
        'dof': dof_data,
        'root_rot': root_rot_data,
        'root_trans_offset': root_trans_offset,
        'pose_aa': pose_aa_data,
        'fps': fps
    }
    
    # 包装成与原始数据相同的格式
    data = {
        'standing_motion_300frames': motion_data
    }
    
    return data

def main():
    # 创建静止站立动作数据
    standing_data = create_standing_motion()
    
    # 保存数据
    output_file = 'standing_motion_300frames.pkl'
    joblib.dump(standing_data, output_file)
    
    print(f"已创建300帧静止站立动作数据并保存到: {output_file}")
    
    # 验证数据
    motion = standing_data['standing_motion_300frames']
    print(f"\n数据验证:")
    print(f"帧数: {motion['dof'].shape[0]}")
    print(f"DOF维度: {motion['dof'].shape[1]}")
    print(f"根部旋转形状: {motion['root_rot'].shape}")
    print(f"根部位移形状: {motion['root_trans_offset'].shape}")
    print(f"Pose AA形状: {motion['pose_aa'].shape}")
    print(f"FPS: {motion['fps']}")
    
    print(f"\n第一帧的DOF值:")
    print(motion['dof'][0])
    
    print(f"\n第一帧的根部旋转 (四元数):")
    print(motion['root_rot'][0])

if __name__ == "__main__":
    main() 