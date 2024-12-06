import torch
import numpy as np
import pytest
from representations_across_sizes.gride import calculate_gride_id

@pytest.fixture
def random_data():
    """Fixture for random test data"""
    torch.manual_seed(42)
    return torch.randn(2, 100, 3)  # [batch_size, n_points, dim]

@pytest.fixture
def uniform_data():
    """Fixture for uniform test data"""
    torch.manual_seed(42)
    n_points = 1000
    dim = 5
    return torch.rand(2, n_points, dim)  # [batch_size, n_points, dim]

@pytest.mark.parametrize("n_points,dim,tolerance", [
    (1000, 5, 0.75),
    (2000, 3, 0.75),
    (500, 2, 0.75),
])
def test_uniform_data(n_points, dim, tolerance):
    """Test that uniform data in d dimensions gives approximately d intrinsic dimensions"""
    torch.manual_seed(42)
    batch_size = 2
    data = torch.rand(batch_size, n_points, dim)
    
    estimated_ids = calculate_gride_id(data)
    
    assert estimated_ids.shape == (batch_size,)
    for id_estimate in estimated_ids:
        assert abs(id_estimate - dim) < tolerance

@pytest.mark.parametrize("n_points,true_id,ambient_dim,tolerance", [
    (1000, 2, 5, 0.5),
    (2000, 3, 6, 0.5),
])
def test_linear_manifold(n_points, true_id, ambient_dim, tolerance):
    """Test that data lying on a linear subspace has correct ID"""
    torch.manual_seed(42)
    batch_size = 2
    
    # Create random data in lower dimension
    data_low_dim = torch.rand(batch_size, n_points, true_id)
    
    # Create random linear map to higher dimension
    projection = torch.randn(true_id, ambient_dim)
    data_high_dim = data_low_dim @ projection
    
    estimated_ids = calculate_gride_id(data_high_dim)
    
    assert estimated_ids.shape == (batch_size,)
    for id_estimate in estimated_ids:
        assert abs(id_estimate - true_id) < tolerance

def test_swiss_roll():
    """Test on Swiss roll manifold which is intrinsically 2D"""
    torch.manual_seed(42)
    batch_size = 2
    n_points = 1000
    
    # Create two Swiss rolls
    data_list = []
    for _ in range(batch_size):
        t = torch.linspace(0, 4*np.pi, n_points)
        height = torch.rand(n_points)
        
        x = t * torch.cos(t)
        y = height
        z = t * torch.sin(t)
        
        data = torch.stack([x, y, z], dim=1)
        data_list.append(data)
    
    data = torch.stack(data_list)  # [batch_size, n_points, 3]
    
    estimated_ids = calculate_gride_id(data)
    
    assert estimated_ids.shape == (batch_size,)
    for id_estimate in estimated_ids:
        # Should be close to 2 (the intrinsic dimension of a Swiss roll)
        assert abs(id_estimate - 2) < 0.5

@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_scale_invariance(random_data, scale):
    """Test numerical stability with different scales"""
    scaled_data = scale * random_data
    id_estimates = calculate_gride_id(scaled_data)
    
    assert id_estimates.shape == (random_data.shape[0],)
    for id_estimate in id_estimates:
        assert not np.isnan(id_estimate)
        assert not np.isinf(id_estimate)
        assert id_estimate > 0

@pytest.mark.parametrize("n1,n2", [(1,2), (2,4), (3,6)])
def test_different_neighbors(random_data, n1, n2):
    """Test different n1,n2 combinations"""
    id_estimates = calculate_gride_id(random_data, n1=n1, n2=n2)
    
    assert id_estimates.shape == (random_data.shape[0],)
    for id_estimate in id_estimates:
        assert not np.isnan(id_estimate)
        assert not np.isinf(id_estimate)
        assert id_estimate > 0

def test_small_sample():
    """Test behavior with small sample sizes"""
    torch.manual_seed(42)
    data = torch.randn(2, 10, 5)  # [batch_size, n_points, dim]
    id_estimates = calculate_gride_id(data)
    
    assert id_estimates.shape == (2,)
    for id_estimate in id_estimates:
        assert not np.isnan(id_estimate)
        assert id_estimate > 0

def test_identical_points():
    """Test handling of identical points"""
    torch.manual_seed(42)
    data = torch.randn(2, 10, 3)  # [batch_size, n_points, dim]
    # Add duplicates to each batch
    data = torch.cat([data, data[:,:2].clone()], dim=1)  # Add 2 duplicates
    
    id_estimates = calculate_gride_id(data)
    assert id_estimates.shape == (2,)
    for id_estimate in id_estimates:
        assert not np.isnan(id_estimate)

def test_reproducibility(random_data):
    """Test that results are reproducible with same random seed"""
    torch.manual_seed(42)
    id1 = calculate_gride_id(random_data)
    
    torch.manual_seed(42)
    id2 = calculate_gride_id(random_data)
    
    assert torch.allclose(id1, id2)