"""
Data Processing Module
Contains core data processing and validation logic.
"""

import pandas as pd
from typing import List, Dict, Any


def validate_input(data: pd.DataFrame) -> bool:
    """
    Validate input data structure and content.
    
    Args:
        data: Input DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    if data is None or data.empty:
        return False
    
    required_columns = ['id', 'name', 'value']
    for col in required_columns:
        if col not in data.columns:
            print(f"Missing required column: {col}")
            return False
    
    return True


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input data with transformations and calculations.
    
    Args:
        data: Input DataFrame to process
        
    Returns:
        Processed DataFrame with additional calculated columns
    """
    # Create a copy to avoid modifying original
    processed = data.copy()
    
    # Add calculated columns
    processed['value_squared'] = processed['value'] ** 2
    processed['value_normalized'] = processed['value'] / processed['value'].max()
    
    # Add category based on value
    processed['category'] = processed['value'].apply(_categorize_value)
    
    return processed


def _categorize_value(value: float) -> str:
    """
    Internal helper to categorize values.
    
    Args:
        value: Numeric value to categorize
        
    Returns:
        Category string
    """
    if value < 10:
        return "low"
    elif value < 50:
        return "medium"
    else:
        return "high"


def aggregate_results(data: pd.DataFrame, group_by: str = 'category') -> Dict[str, Any]:
    """
    Aggregate processed data by specified column.
    
    Args:
        data: Processed DataFrame
        group_by: Column name to group by
        
    Returns:
        Dictionary with aggregated statistics
    """
    aggregates = data.groupby(group_by)['value'].agg(['mean', 'sum', 'count'])
    return aggregates.to_dict()
