import pandas as pd
import pytest
from src.analysis.signal_maps import _categorize_signal

@pytest.fixture
def sample_row():
    return pd.Series({'axis_x': 5.0, 'axis_y': 10.0})

def test_categorize_signal():
    """
    Tests the four quadrants of the signal map categorization logic.
    This is a perfect example of a unit test [cite: 805-807].
    """
    median_x = 4.0
    median_y = 8.0
    
    # 1. Test Strong Signal (High X, High Y)
    row_strong = pd.Series({'axis_x': 5.0, 'axis_y': 10.0})
    assert _categorize_signal(row_strong, median_x, median_y) == "Strong Signal"
    
    # 2. Test Weak Signal (Low X, High Y)
    row_weak = pd.Series({'axis_x': 3.0, 'axis_y': 10.0})
    assert _categorize_signal(row_weak, median_x, median_y) == "Weak Signal"
    
    # 3. Test Latent Signal (Low X, Low Y)
    row_latent = pd.Series({'axis_x': 3.0, 'axis_y': 5.0})
    assert _categorize_signal(row_latent, median_x, median_y) == "Latent Signal"
    
    # 4. Test Well-known (High X, Low Y)
    row_well_known = pd.Series({'axis_x': 5.0, 'axis_y': 5.0})
    assert _categorize_signal(row_well_known, median_x, median_y) == "Well-known but not strong"
    
    # 5. Test "on the line" (edge case)
    row_on_line = pd.Series({'axis_x': 4.0, 'axis_y': 8.0})
    # Our logic uses >=, so this should be "Strong Signal"
    assert _categorize_signal(row_on_line, median_x, median_y) == "Strong Signal"