import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
class ReactionWheelPower(nn.Module):

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
        