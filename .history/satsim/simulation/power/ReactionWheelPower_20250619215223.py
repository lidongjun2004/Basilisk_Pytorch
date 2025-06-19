__all__ = [
    'ReactionWheelPower', 
    
]
import torch
import torch.nn as nn
from torch import Tensor
from enum import IntEnum, auto
from typing import TypedDict

class ReactionWheelPowerStateDict(TypedDict):
    base_power_need: Tensor


class ReactionWheelPower(nn.Module):
    """
    反作用轮功率计算
    
    参数:
        base_power_need (float): 基础功率需求 (W)
        elec_to_mech_eff (float): 电到机械转换效率 (0-1]
        mech_to_elec_eff (float): 机械到电转换效率 (负值表示禁用能量回收)
    
    """
    def __init__(self, 
                 basePowerNeed=0.0, 
                 elecToMechEfficiency=1.0, 
                 mechToElecEfficiency=-1.0):
        
        super().__init__()
        
        if elecToMechEfficiency <= 0.0:
            raise ValueError("elec_to_mech_eff must be a strictly positive value.")
        
        self.register_buffer("base_power_need", torch.tensor(basePowerNeed))
        self.register_buffer("elec_to_mech_eff", torch.tensor(elecToMechEfficiency))
        self.register_buffer("mech_to_elec_eff", torch.tensor(mechToElecEfficiency))
        
        self.register_buffer("omega", torch.tensor(0.0))
        self.register_buffer("u_current", torch.tensor(0.0))

    def forward(self, omega: Tensor, u_current: Tensor) -> Tensor:
        """
        计算反作用轮功率需求
        
        参数:
            omega (Tensor): 轮子角速度 (rad/s)
            u_current (Tensor): 当前控制力矩 (N·m)
            
        返回:
            Tensor: 净功率需求 (W)，负值表示消耗功率
        """

        self.omega = omega
        self.u_current = u_current
        
        wheel_power = omega * u_current
        # 创建条件掩码
        is_accel_mode = (self.mech_to_elec_eff < 0) | (wheel_power > 0)
        
        # 分别计算两种模式的功率
        accel_power = self.base_power_need + torch.abs(wheel_power) / self.elec_to_mech_eff
        regen_power = self.base_power_need + self.mech_to_elec_eff * wheel_power
        
        # 根据模式选择功率值
        total_power = torch.where(is_accel_mode, accel_power, regen_power)
        
        # 返回负值表示功率消耗
        return -total_power