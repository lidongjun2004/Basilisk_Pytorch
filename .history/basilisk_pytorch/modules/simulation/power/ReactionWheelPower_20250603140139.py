import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
class ReactionWheelPower(SimulationModule):

    def __init__(self, 
                 base_power_need=0.0, 
                 elec_to_mech_eff=1.0, 
                 mech_to_elec_eff=-1.0):
        """
        反作用飞轮功率计算模块
        """
        super().__init__()
        
        # 注册为缓冲区，这些参数不会被优化但会随模型一起保存
        self.register_buffer('base_power_need', torch.tensor(base_power_need, dtype=torch.float32))
        self.register_buffer('elec_to_mech_eff', torch.tensor(elec_to_mech_eff, dtype=torch.float32))
        self.register_buffer('mech_to_elec_eff', torch.tensor(mech_to_elec_eff, dtype=torch.float32))