import torch
import pytest
from basilisk_pytorch.modules.simulation.power import ReactionWheelPower

def test_initialization():
    """Test module initialization with valid parameters"""
    rw_power = ReactionWheelPower(
        base_power_need=5.0,
        elec_to_mech_eff=0.8,
        mech_to_elec_eff=0.6
    )
    
    # Check parameter values
    assert torch.isclose(rw_power.base_power_need, torch.tensor(5.0))
    assert torch.isclose(rw_power.elec_to_mech_eff, torch.tensor(0.8))
    assert torch.isclose(rw_power.mech_to_elec_eff, torch.tensor(0.6))

def test_invalid_efficiency():
    """Test invalid efficiency parameter handling"""
    with pytest.raises(ValueError):
        # Invalid electrical to mechanical efficiency
        ReactionWheelPower(elec_to_mech_eff=0.0)
    
    with pytest.raises(ValueError):
        # Negative electrical to mechanical efficiency
        ReactionWheelPower(elec_to_mech_eff=-0.5)

def test_power_calculation_modes():
    """Test different power calculation modes"""
    # Create module with regenerative braking enabled
    rw_power = ReactionWheelPower(
        base_power_need=5.0,
        elec_to_mech_eff=0.8,
        mech_to_elec_eff=0.6
    )
    
    # Test acceleration mode (omega and u_current same sign)
    omega = torch.tensor([100.0, 200.0])  # rad/s
    u_current = torch.tensor([0.1, 0.2])  # N·m
    
    power = rw_power(omega, u_current)
    expected = 5.0 + (torch.abs(omega * u_current) / 0.8
    assert torch.allclose(power, -expected)
    
    # Test regenerative braking mode (omega and u_current opposite signs)
    omega = torch.tensor([100.0, -150.0])  # rad/s
    u_current = torch.tensor([-0.1, 0.15])  # N·m
    
    power = rw_power(omega, u_current)
    wheel_power = omega * u_current
    expected = 5.0 + wheel_power * 0.6
    assert torch.allclose(power, -expected)

def test_regenerative_braking_disabled():
    """Test behavior when regenerative braking is disabled"""
    rw_power = ReactionWheelPower(
        base_power_need=3.0,
        elec_to_mech_eff=0.7,
        mech_to_elec_eff=-1.0  # Disabled
    )
    
    # Both acceleration and deceleration should use the same calculation
    omega = torch.tensor([50.0, -80.0])  # rad/s
    u_current = torch.tensor([0.05, -0.08])  # N·m
    
    power = rw_power(omega, u_current)
    expected = 3.0 + torch.abs(omega * u_current) / 0.7
    assert torch.allclose(power, -expected)

def test_vectorized_calculation():
    """Test batch processing of multiple wheels"""
    rw_power = ReactionWheelPower(
        base_power_need=4.0,
        elec_to_mech_eff=0.75,
        mech_to_elec_eff=0.5
    )
    
    # Create input for 4 wheels
    omega = torch.tensor([100.0, -150.0, 200.0, -50.0])  # rad/s
    u_current = torch.tensor([0.1, 0.15, -0.2, 0.05])  # N·m
    
    power = rw_power(omega, u_current)
    
    # Calculate expected values individually
    wheel_power = omega * u_current
    expected = torch.zeros_like(omega)
    
    # Wheel 0: accelerating (same sign)
    expected[0] = 4.0 + torch.abs(wheel_power[0]) / 0.75
    
    # Wheel 1: decelerating (opposite signs) with regenerative braking
    expected[1] = 4.0 + wheel_power[1] * 0.5
    
    # Wheel 2: decelerating (opposite signs) with regenerative braking
    expected[2] = 4.0 + wheel_power[2] * 0.5
    
    # Wheel 3: accelerating (same sign)
    expected[3] = 4.0 + torch.abs(wheel_power[3]) / 0.75
    
    assert torch.allclose(power, -expected)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])