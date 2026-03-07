"""
Unit tests for the Body class in the neutral_physics_engine package.

This module verifies that a Body object correctly initializes with valid
parameters and properly raises ValueErrors when given invalid parameters
(e.g., non-positive mass/radius, or incorrectly shaped vectors).
"""

from neutral_physics_engine.body import Body
import numpy as np
import pytest


def test_valid_body():
    """
    Test the successful creation of a Body instance.

    Verifies that valid inputs are correctly assigned to properties,
    and that array-like inputs for position and velocity are cast
    to numpy arrays.
    """
    # Initialize a Body with valid positive mass/radius and 3D vectors
    body = Body(mass=1.0, pos=[5, 30, -25.56], vel=[34e12, -5.598, 5e-10], radius=0.5)

    # Check scalar attributes
    assert body.mass == 1.0
    assert body.radius == 0.5

    # Ensure position and velocity lists were successfully converted to numpy arrays
    assert isinstance(body.pos, np.ndarray)
    assert isinstance(body.vel, np.ndarray)

    # Verify the values within the arrays match the input data
    np.testing.assert_array_equal(body.pos, np.array([5, 30, -25.56]))
    np.testing.assert_array_equal(body.vel, np.array([34e12, -5.598, 5e-10]))


def test_invalid_mass():
    """
    Test that a ValueError is raised when initializing a Body with invalid mass.
    Mass must be strictly greater than zero.
    """
    # Test zero mass
    with pytest.raises(ValueError) as e:
        Body(mass=0, pos=[0, 0, 0], vel=[0, 0, 0], radius=1)
    assert str(e.value) == "Body mass must be positive and non-zero"

    # Test negative mass
    with pytest.raises(ValueError) as e:
        Body(mass=-1, pos=[0, 0, 0], vel=[0, 0, 0], radius=1)
    assert str(e.value) == "Body mass must be positive and non-zero"


def test_invalid_radius():
    """
    Test that a ValueError is raised when initializing a Body with invalid radius.
    Radius must be strictly greater than zero.
    """
    # Test zero radius
    with pytest.raises(ValueError) as e:
        Body(mass=1, pos=[0, 0, 0], vel=[0, 0, 0], radius=0)
    assert str(e.value) == "Body radius must be positive"

    # Test negative radius
    with pytest.raises(ValueError) as e:
        Body(mass=1, pos=[0, 0, 0], vel=[0, 0, 0], radius=-1)
    assert str(e.value) == "Body radius must be positive"


def test_invalid_position_shape():
    """
    Test that a ValueError is raised when the position vector is not 3-dimensional.
    """
    # Provide a 2D position vector instead of the expected 3D vector
    with pytest.raises(ValueError) as e:
        Body(mass=1, pos=[0, 0], vel=[0, 0, 0], radius=1)
    assert str(e.value) == "Position must be a 3D vector of shape (3,)"


def test_invalid_velocity_shape():
    """
    Test that a ValueError is raised when the velocity vector is not 3-dimensional.
    """
    # Provide a 2D velocity vector instead of the expected 3D vector
    with pytest.raises(ValueError) as e:
        Body(mass=1, pos=[0, 0, 0], vel=[0, 0], radius=1)
    assert str(e.value) == "Velocity must be a 3D vector of shape (3,)"
