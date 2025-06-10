import torch
from power.ReactioonWheelPower import ReactionWheelPower
# 创建功率计算模块
rw_power = ReactionWheelPower(
    base_power_need=5.0,       # 5W基础功率
    elec_to_mech_eff=0.8,      # 80%电到机械效率
    mech_to_elec_eff=0.6       # 60%机械到电效率
)

# 单个样本计算
omega = torch.tensor(100.0)    # rad/s
u_current = torch.tensor(0.1)  # N·m
power = rw_power(omega, u_current)
print(f"单个样本功率需求: {power.item():.2f} W")

# 批量计算
omegas = torch.tensor([50.0, 100.0, -80.0])
u_currents = torch.tensor([0.05, 0.1, 0.08])
powers = rw_power.vectorized_forward(omegas, u_currents)
print(f"批量功率需求: {powers.numpy()} W")