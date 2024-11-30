import torch
from representations_across_sizes.utils import dirichlet_energy

def test_dirichlet_energy():
    # Test with zero matrix should give 0 energy
    zero_points = torch.zeros((5, 3))
    assert dirichlet_energy(zero_points) == 0

    # Test with constant matrix should give 0 energy
    constant_points = torch.full((4, 2), fill_value=3.14)
    assert dirichlet_energy(constant_points) == 0

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Test symmetry - energy should be the same if we permute the points
    points1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    points2 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    )  # same points, different order
    assert torch.allclose(dirichlet_energy(points1), dirichlet_energy(points2))

    # Test scaling - energy should scale quadratically with distance
    scaled_points = 2 * points  # double all distances
    assert torch.allclose(
        dirichlet_energy(scaled_points), 4 * dirichlet_energy(points)
    )  # energy should be 4x larger

    # Test translation invariance - energy should not change if we translate all points
    translation = torch.tensor([2.0, -3.0])
    translated_points = points + translation
    assert torch.allclose(dirichlet_energy(points), dirichlet_energy(translated_points))

    # Test with higher dimensions
    high_dim_points = torch.randn(5, 10)  # 5 points in 10D space
    assert (
        dirichlet_energy(high_dim_points) >= 0
    )  # energy should always be non-negative

    # Test with single point - should return 0
    single_point = torch.randn(1, 3)
    assert dirichlet_energy(single_point) == 0
