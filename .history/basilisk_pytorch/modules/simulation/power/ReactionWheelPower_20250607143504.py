import torch
import torch.nn as nn
from torch import Tensor

class ReactionWheelPower(nn.Module):
    """
    反作用轮功率计算
    
    参数:
        base_power_need (float): 基础功率需求 (W)
        elec_to_mech_eff (float): 电到机械转换效率 (0-1]
        mech_to_elec_eff (float): 机械到电转换效率 (负值表示禁用能量回收)
    
    """
    def __init__(self, 
                 name = "ReactionWheelPower",
                 basePowerNeed=0.0, 
                 elecToMechEfficiency=1.0, 
                 mechToElecEfficiency=-1.0):
        
        super().__init__(name)
        
        if elecToMechEfficiency <= 0.0:
            raise ValueError("elec_to_mech_eff must be a strictly positive value.")
        
        # 注册为不可训练参数
        self.register_buffer("base_power_need", torch.tensor(basePowerNeed))
        self.register_buffer("elec_to_mech_eff", torch.tensor(elecToMechEfficiency))
        self.register_buffer("mech_to_elec_eff", torch.tensor(mechToElecEfficiency))
        
        # 初始化轮子状态
        self.register_buffer("omega", torch.tensor(0.0))
        self.register_buffer("u_current", torch.tensor(0.0))