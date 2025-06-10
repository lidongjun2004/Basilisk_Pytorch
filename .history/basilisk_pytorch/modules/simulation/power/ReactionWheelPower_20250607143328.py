import torch
import torch.nn as nn
from torch import Tensor

class ReactionWheelPower(nn.Module):
    """
    反作用轮功率计算模块
    
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
        
        self.base_power_need = torch.tensor(basePowerNeed, dtype=torch.float32)
        self.elec_to_mech_eff = torch.tensor(elecToMechEfficiency, dtype=torch.float32)
        self.mech_to_elec_eff = torch.tensor(mechToElecEfficiency, dtype=torch.float32)
        self.rw_power_need = torch.tensor(0.0)
        self.wheel_power = torch.tensor(0.0)

        if self.elec_to_mech_eff <= 0.0:
            raise ValueError("elecToMechEfficiency is {self.elec_to_mech_eff}, must a strictly positive value.")
        