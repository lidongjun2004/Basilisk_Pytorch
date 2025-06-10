import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
class ReactionWheelPower(SimulationModule):

    def __init__(self, 
                 name = "ReactionWheelPower",
                 basePowerNeed=0.0, 
                 elecToMechEfficiency=1.0, 
                 mechToElecEfficiency=-1.0):
        
        super().__init__(name)
        
        self.base_power_need = torch.tensor(basePowerNeed, dtype=torch.float32)
        self.elec_to_mech_eff = torch.tensor(v, dtype=torch.float32)
        self.mech_to_elec_eff = torch.tensor(mech_to_elec_eff, dtype=torch.float32)
        self.rw_power_need = torch.tensor()