from typing import Any, Callable
import pytest
import torch

from satsim import Timer
from satsim.simulation import ReactionWheelPower

PowerOperator = Callable[
    [
        float,  # base_power_need
        float,  # elec_to_mech_eff
        float,  # mech_to_elec_eff
        torch.Tensor,  # omega
        torch.Tensor,  # u_current
    ],
    torch.Tensor,  # net_power
]

class TestReactionWheelPowerOperator:
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_acceleration_mode(self, device: str):
        """测试加速模式（正功率消耗）"""
        # 配置参数
        base_power = 10.0
        elec_mech_eff = 0.8
        mech_elec_eff = -1.0  # 禁用能量回收
        
        # 输入数据
        omega = torch.tensor([5.0, 10.0], device=device)
        u_current = torch.tensor([1.0, 2.0], device=device)
        
        # 计算功率
        rw = ReactionWheelPower(
            base_power_need=base_power,
            elec_to_mech_eff=elec_mech_eff,
            mech_to_elec_eff=mech_elec_eff
        )
        state = rw.reset()
        _, (net_power,) = rw(state, omega=omega, u_current=u_current)
        
        # 验证结果
        expected_power = torch.tensor([
            -(base_power + (5.0 * 1.0) / elec_mech_eff),
            -(base_power + (10.0 * 2.0) / elec_mech_eff)
        ], device=device)
        
        assert torch.allclose(net_power, expected_power, atol=1e-5)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_regeneration_mode(self, device: str):
        """测试再生制动模式（负功率消耗）"""
        # 配置参数
        base_power = 10.0
        elec_mech_eff = 0.8
        mech_elec_eff = 0.6  # 启用能量回收
        
        # 输入数据（负功率表示制动）
        omega = torch.tensor([-5.0, -10.0], device=device)
        u_current = torch.tensor([1.0, 2.0], device=device)
        
        # 计算功率
        rw = ReactionWheelPower(
            base_power_need=base_power,
            elec_to_mech_eff=elec_mech_eff,
            mech_to_elec_eff=mech_elec_eff
        )
        state = rw.reset()
        _, (net_power,) = rw(state, omega=omega, u_current=u_current)
        
        # 验证结果
        expected_power = torch.tensor([
            -(base_power + mech_elec_eff * (-5.0 * 1.0)),
            -(base_power + mech_elec_eff * (-10.0 * 2.0))
        ], device=device)
        
        assert torch.allclose(net_power, expected_power, atol=1e-5)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_mixed_modes(self, device: str):
        """测试混合模式（加速和制动同时存在）"""
        # 配置参数
        base_power = 10.0
        elec_mech_eff = 0.8
        mech_elec_eff = 0.6
        
        # 输入数据（混合模式）
        omega = torch.tensor([5.0, -5.0, 0.0], device=device)
        u_current = torch.tensor([1.0, 1.0, 1.0], device=device)
        
        # 计算功率
        rw = ReactionWheelPower(
            base_power_need=base_power,
            elec_to_mech_eff=elec_mech_eff,
            mech_to_elec_eff=mech_elec_eff
        )
        state = rw.reset()
        _, (net_power,) = rw(state, omega=omega, u_current=u_current)
        
        # 验证结果
        expected_power = torch.tensor([
            -(base_power + (5.0 * 1.0) / elec_mech_eff),  # 加速
            -(base_power + mech_elec_eff * (-5.0 * 1.0)),  # 制动
            -(base_power)  # 仅基础功率
        ], device=device)
        
        assert torch.allclose(net_power, expected_power, atol=1e-5)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_zero_conditions(self, device: str):
        """测试零输入条件"""
        # 配置参数
        base_power = 10.0
        elec_mech_eff = 0.8
        mech_elec_eff = 0.6
        
        # 输入数据
        omega = torch.tensor([0.0, 0.0], device=device)
        u_current = torch.tensor([0.0, 0.0], device=device)
        
        # 计算功率
        rw = ReactionWheelPower(
            base_power_need=base_power,
            elec_to_mech_eff=elec_mech_eff,
            mech_to_elec_eff=mech_elec_eff
        )
        state = rw.reset()
        _, (net_power,) = rw(state, omega=omega, u_current=u_current)
        
        # 验证结果
        expected_power = torch.tensor([-base_power, -base_power], device=device)
        assert torch.allclose(net_power, expected_power, atol=1e-5)

class TestReactionWheelPowerInitialization:
    
    def test_initialization(self) -> None:
        """测试模块初始化"""
        rw = ReactionWheelPower(
            base_power_need=15.0,
            elec_to_mech_eff=0.85,
            mech_to_elec_eff=0.7
        )
        
        assert rw._base_power_need == 15.0
        assert rw._elec_to_mech_eff == 0.85
        assert rw._mech_to_elec_eff == 0.7
    
    def test_reset(self) -> None:
        """测试reset方法"""
        rw = ReactionWheelPower(
            base_power_need=15.0,
            elec_to_mech_eff=0.85,
            mech_to_elec_eff=0.7
        )
        state = rw.reset()
        
        # 验证状态字典为空（无状态参数）
        assert state == {}

class TestReactionWheelPowerGradients:
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_gradient_propagation(self, device: str):
        """测试梯度传播"""
        rw = ReactionWheelPower(
            base_power_need=10.0,
            elec_to_mech_eff=0.8,
            mech_to_elec_eff=0.6
        )
        
        # 创建需要梯度的输入
        omega = torch.tensor([5.0], device=device, requires_grad=True)
        u_current = torch.tensor([1.0], device=device, requires_grad=True)
        
        # 前向传播
        state = rw.reset()
        _, (net_power,) = rw(state, omega=omega, u_current=u_current)
        
        # 反向传播
        net_power.backward()
        
        # 验证梯度
        assert omega.grad is not None
        assert u_current.grad is not None
        
        # 手动计算预期梯度
        wheel_power = omega * u_current
        is_accel = (rw._mech_to_elec_eff < 0) | (wheel_power > 0)
        
        # 梯度公式
        if is_accel:
            d_power_d_omega = -u_current / rw._elec_to_mech_eff
            d_power_d_u = -omega / rw._elec_to_mech_eff
        else:
            d_power_d_omega = -rw._mech_to_elec_eff * u_current
            d_power_d_u = -rw._mech_to_elec_eff * omega
        
        # 验证梯度值
        assert torch.allclose(omega.grad, d_power_d_omega)
        assert torch.allclose(u_current.grad, d_power_d_u)

def test_disabled_regeneration():
    """测试禁用能量回收模式"""
    # 配置禁用能量回收
    rw = ReactionWheelPower(
        base_power_need=10.0,
        elec_to_mech_eff=0.8,
        mech_to_elec_eff=-1.0  # 禁用回收
    )
    
    # 输入数据（制动模式）
    omega = torch.tensor([-5.0])
    u_current = torch.tensor([1.0])
    
    # 计算功率
    state = rw.reset()
    _, (net_power,) = rw(state, omega=omega, u_current=u_current)
    
    # 验证结果（应使用加速模式公式）
    expected_power = -(10.0 + abs(-5.0 * 1.0) / 0.8)
    assert torch.allclose(net_power, torch.tensor([expected_power]))

def test_efficiency_edge_cases():
    """测试效率边界情况"""
    # 100% 效率情况
    rw = ReactionWheelPower(
        base_power_need=10.0,
        elec_to_mech_eff=1.0,
        mech_to_elec_eff=1.0
    )
    
    # 加速模式
    state = rw.reset()
    _, (accel_power,) = rw(state, omega=torch.tensor([5.0]), u_current=torch.tensor([1.0]))
    assert torch.allclose(accel_power, torch.tensor([-(10.0 + 5.0)]))
    
    # 制动模式
    _, (regen_power,) = rw(state, omega=torch.tensor([-5.0]), u_current=torch.tensor([1.0]))
    assert torch.allclose(regen_power, torch.tensor([-(10.0 - 5.0)]))

def test_vectorization():
    """测试批量处理能力"""
    rw = ReactionWheelPower(
        base_power_need=10.0,
        elec_to_mech_eff=0.8,
        mech_to_elec_eff=0.6
    )
    
    # 批量输入数据
    omega = torch.tensor([5.0, -5.0, 0.0, 3.0])
    u_current = torch.tensor([1.0, 1.0, 1.0, 2.0])
    
    # 计算功率
    state = rw.reset()
    _, (net_power,) = rw(state, omega=omega, u_current=u_current)
    
    # 验证结果形状
    assert net_power.shape == (4,)
    
    # 验证计算结果
    expected_power = torch.tensor([
        -(10.0 + (5.0 * 1.0) / 0.8),  # 加速
        -(10.0 + 0.6 * (-5.0 * 1.0)),  # 制动
        -(10.0),  # 仅基础功率
        -(10.0 + (3.0 * 2.0) / 0.8)   # 加速
    ])
    
    assert torch.allclose(net_power, expected_power, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])