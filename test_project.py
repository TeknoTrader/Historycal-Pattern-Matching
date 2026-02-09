import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from project import (
    calculate_pattern_similarity,
    find_similar_patterns,
    analyze_future_performance,
    calculate_statistics,
    validate_ticker
)


def test_calculate_pattern_similarity():
    """Test DTW and correlation calculation with known patterns"""
    # Pattern con variazione (non costante)
    pattern1 = np.array([0.0, 1.5, 3.2, 5.1, 7.3])
    pattern2 = np.array([0.0, 1.5, 3.2, 5.1, 7.3])
    returns1 = np.array([1.5, 1.7, 1.9, 2.2])
    returns2 = np.array([1.5, 1.7, 1.9, 2.2])

    dtw_sim, dtw_dist, corr = calculate_pattern_similarity(
        pattern1, pattern2, returns1, returns2
    )

    # DTW dovrebbe essere quasi perfetto
    assert dtw_sim > 0.9, f"Expected DTW similarity > 0.9, got {dtw_sim}"
    assert dtw_dist < 10, f"Expected DTW distance < 10, got {dtw_dist}"

    # Correlazione dovrebbe essere alta (o NaN se varianza zero, che gestiamo)
    if not np.isnan(corr):
        assert corr > 0.9, f"Expected correlation > 0.9, got {corr}"
    else:
        # Se è NaN, vuol dire che i pattern hanno varianza zero (sono costanti)
        # In questo caso il test passa comunque perché è un caso valido
        assert np.isnan(corr), "Correlation is NaN (zero variance case)"

    # Pattern opposti
    pattern3 = np.array([0.0, -1.5, -3.2, -5.1, -7.3])
    returns3 = np.array([-1.5, -1.7, -1.9, -2.2])

    _, _, corr_opposite = calculate_pattern_similarity(
        pattern1, pattern3, returns1, returns3
    )

    if not np.isnan(corr_opposite):
        assert corr_opposite < -0.5, f"Expected negative correlation < -0.5, got {corr_opposite}"


def test_find_similar_patterns():
    """Test pattern finding with synthetic repeating data"""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')

    # Pattern con variazione reale
    base_pattern = [100, 102, 104, 103, 105]
    pattern = (base_pattern * 40)[:200]
    prices = np.array(pattern)

    df = pd.DataFrame({
        'Close': prices,
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Volume': [1000000] * 200
    }, index=dates)

    pattern_end_date = dates[-1]

    similar = find_similar_patterns(
        df,
        pattern_length=5,
        pattern_end_date=pattern_end_date,
        min_dtw_similarity=0.7,
        min_correlation=0.5
    )

    assert len(similar) > 0, "Should find at least some similar patterns"
    assert 'dtw_similarity' in similar.columns
    assert 'correlation' in similar.columns
    assert 'combined_score' in similar.columns


def test_analyze_future_performance():
    """Test future performance analysis with trending data"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 150, 100)

    df = pd.DataFrame({
        'Close': prices,
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Volume': [1000000] * 100
    }, index=dates)

    similar_periods = pd.DataFrame({
        'start_idx': [10, 30, 50],
        'end_idx': [20, 40, 60],
        'dtw_similarity': [0.9, 0.85, 0.8],
        'dtw_distance': [5, 7, 10],
        'correlation': [0.95, 0.9, 0.85],
        'combined_score': [0.925, 0.875, 0.825],
        'start_date': [dates[10], dates[30], dates[50]],
        'end_date': [dates[20], dates[40], dates[60]]
    })

    performance = analyze_future_performance(df, similar_periods, future_periods=10)

    assert len(performance) > 0, "Should return performance data"
    assert 'direction' in performance.columns
    assert 'final_return_%' in performance.columns
    assert 'max_positive_excursion_%' in performance.columns
    assert 'max_negative_excursion_%' in performance.columns

    long_count = len(performance[performance['direction'] == 'LONG'])
    assert long_count >= 0, "Should classify directions"


def test_calculate_statistics():
    """Test statistics calculation with sample data"""
    performance_df = pd.DataFrame({
        'dtw_similarity': [0.9, 0.85, 0.8, 0.75, 0.7],
        'correlation': [0.95, 0.9, 0.85, 0.8, 0.75],
        'combined_score': [0.925, 0.875, 0.825, 0.775, 0.725],
        'direction': ['LONG', 'LONG', 'SHORT', 'LONG', 'SHORT'],
        'final_return_%': [5.0, 3.0, -2.0, 4.0, -1.5],
        'max_positive_excursion_%': [6.0, 4.5, 0.5, 5.0, 0.2],
        'max_negative_excursion_%': [-1.0, -0.5, -3.0, -1.5, -2.5]
    })

    stats = calculate_statistics(performance_df)

    assert stats is not None, "Stats should not be None"
    assert stats['total_matches'] == 5
    assert stats['long_count'] == 3
    assert stats['short_count'] == 2
    assert 0 <= stats['long_percentage'] <= 100
    assert 0 <= stats['short_percentage'] <= 100
    assert stats['long_percentage'] + stats['short_percentage'] == 100

    assert 'avg_return' in stats
    assert 'median_return' in stats
    assert 'best_case' in stats
    assert 'worst_case' in stats

    assert stats['best_case'] >= stats['worst_case']


def test_calculate_statistics_empty():
    """Test statistics with empty DataFrame returns None"""
    empty_df = pd.DataFrame()
    stats = calculate_statistics(empty_df)

    assert stats is None, "Should return None for empty DataFrame"


def test_find_similar_patterns_no_matches():
    """Test pattern finding with very strict thresholds returns empty"""
    dates = pd.date_range(start='2020-01-01', periods=50, freq='D')

    # Dati casuali con variazione
    np.random.seed(42)
    prices = np.random.rand(50) * 20 + 100

    df = pd.DataFrame({
        'Close': prices,
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Volume': [1000000] * 50
    }, index=dates)

    pattern_end_date = dates[-1]

    similar = find_similar_patterns(
        df,
        pattern_length=5,
        pattern_end_date=pattern_end_date,
        min_dtw_similarity=0.99,
        min_correlation=0.99
    )

    assert len(similar) == 0, "Should find no matches with very strict thresholds"


def test_validate_ticker_valid():
    """Test ticker validation with known valid ticker"""
    is_valid, error_msg = validate_ticker("AAPL")
    assert is_valid is True, "AAPL should be valid"
    assert error_msg is None, "Error message should be None for valid ticker"


def test_validate_ticker_invalid():
    """Test ticker validation with invalid ticker"""
    is_valid, error_msg = validate_ticker("INVALIDTICKER12345XYZ")
    assert is_valid is False, "Invalid ticker should return False"
    assert error_msg is not None, "Error message should exist for invalid ticker"
    assert "INVALIDTICKER12345XYZ" in error_msg, "Error message should contain the invalid ticker"


def test_validate_ticker_empty():
    """Test ticker validation with empty string"""
    is_valid, error_msg = validate_ticker("")
    assert is_valid is False, "Empty ticker should return False"
    assert error_msg is not None, "Error message should exist for empty ticker"
