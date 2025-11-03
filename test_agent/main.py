"""
Main Application Entry Point
This is the main entry point for the data processing application.
"""

from utils.data_processor import process_data, validate_input
from utils.file_handler import read_csv_file, write_results
from config import DATABASE_URL, MAX_RETRIES, TIMEOUT


def main():
    """Main function to orchestrate data processing."""
    print(f"Connecting to database: {DATABASE_URL}")
    print(f"Configuration: MAX_RETRIES={MAX_RETRIES}, TIMEOUT={TIMEOUT}")
    
    # Read input data
    data = read_csv_file("input_data.csv")
    
    # Validate input
    if not validate_input(data):
        print("Input validation failed!")
        return False
    
    # Process the data
    results = process_data(data)
    
    # Write results
    write_results(results, "output_results.csv")
    
    print("Processing completed successfully!")
    return True


if __name__ == "__main__":
    main()
