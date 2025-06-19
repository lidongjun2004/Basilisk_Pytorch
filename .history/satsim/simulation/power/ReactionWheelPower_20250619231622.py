__all__ = [
    'ReactionWheelPower', 
    'ReactionWheelPowerStateDict', 
]
import torch
import torch.nn as nn
from torch import Tensor
from enum import IntEnum, auto
from typing import TypedDict
from satsim.architecture import Module

class ReactionWheelPowerStateDict(TypedDict):
    base_power_need: Tensor
    elec_to_mech_eff: Tensor
    mech_to_elec_eff: Tensor

class ReactionWheelPower(Module[ReactionWheelPowerStateDict]):
    """
    反作用轮功率计算
    
    参数:
        base_power_need (float): 基础功率需求 (W)
        elec_to_mech_eff (float): 电到机械转换效率 (0-1]
        mech_to_elec_eff (float): 机械到电转换效率 (负值表示禁用能量回收)
    
    """
    def __init__(
        self, 
        *args,
        base_power_need: float, 
        elec_to_mech_eff: float, 
        mech_to_elec_eff: float, 
        **kwargs, 
    ) -> None:
        super().__init__(*args, **kwargs)
        self._base_power_need = base_power_need
        self._elec_to_mech_eff = elec_to_mech_eff
        self._mech_to_elec_eff = mech_to_elec_eff

    def reset(self) -> ReactionWheelPowerStateDict:
        state_dict = super().reset()
        state_dict.update(
            base_power_need = torch.tensor(self._base_power_need, dtype=torch.float32),
            elec_to_mech_eff = torch.tensor(self._elec_to_mech_eff, dtype=torch.float32),
            mech_to_elec_eff = torch.tensor(self._mech_to_elec_eff, dtype=torch.float32),
        )
        return state_dict

    def forward(
        self, 
        state_dict: ReactionWheelPowerStateDict,
        *args, 
        omega: Tensor, 
        u_current: Tensor, 
        **kwargs, 
    ) -> tuple[ReactionWheelPowerStateDict, tuple[torch.Tensor]]:
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

        is_accel_mode = (self._mech_to_elec_eff < 0) | (wheel_power > 0)
        
        accel_power = self._base_power_need + torch.abs(wheel_power) / self._elec_to_mech_eff
        regen_power = self._base_power_need + self._mech_to_elec_eff * wheel_power
        
        total_power = torch.where(is_accel_mode, accel_power, regen_power)
        
        return state_dict, (-total_power, )