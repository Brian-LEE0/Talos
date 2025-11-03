"""
File Handling Module
Manages file I/O operations for CSV files.
"""

import pandas as pd
import os
from typing import Optional
from config import INPUT_DIR, OUTPUT_DIR


def read_csv_file(filename: str, input_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Read a CSV file from the input directory.
    
    Args:
        filename: Name of the CSV file
        input_dir: Optional custom input directory (defaults to config.INPUT_DIR)
        
    Returns:
        DataFrame with the file contents
    """
    if input_dir is None:
        input_dir = INPUT_DIR
    
    filepath = os.path.join(input_dir, filename)
    
    try:
        data = pd.read_csv(filepath)
        print(f"Successfully read {len(data)} rows from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()


def write_results(data: pd.DataFrame, filename: str, output_dir: Optional[str] = None) -> bool:
    """
    Write processed results to a CSV file.
    
    Args:
        data: DataFrame to write
        filename: Output filename
        output_dir: Optional custom output directory (defaults to config.OUTPUT_DIR)
        
    Returns:
        True if successful, False otherwise
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        data.to_csv(filepath, index=False)
        print(f"Successfully wrote {len(data)} rows to {filename}")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False


def list_input_files() -> list:
    """
    List all CSV files in the input directory.
    
    Returns:
        List of CSV filenames
    """
    try:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []
