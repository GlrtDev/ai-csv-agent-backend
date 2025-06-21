import pandas as pd
import logging
import time
from typing import List, Dict
import io

logger = logging.getLogger(__name__)

def process_csv_data(file_path: str, background_task: bool = False) -> List[Dict]:
    """
    Processes the CSV file, converts it to a list of dictionaries, and simulates processing time.

    Args:
        file_path (str): The path to the CSV file.
        background_task (bool): Indicates if the task is running in the background.

    Returns:
        List[Dict]: A list of dictionaries representing the CSV data.  Returns an empty list on error.
    """
    try:
        if background_task:
            logger.info(f"Background task: Processing CSV file: {file_path}")
        else:
            logger.info(f"Processing CSV file: {file_path}")

        # Simulate file processing delay (remove or adjust as needed)
        time.sleep(2)

        # Read the CSV file using pandas
        df = pd.read_csv(file_path)

        # Convert the DataFrame to a list of dictionaries
        data = df.to_dict(orient='records')
        logger.info(f"Successfully processed CSV data from {file_path}")
        return data
    except Exception as e:
        error_message = f"Error processing CSV file: {e}"
        logger.error(error_message)
        if background_task:
            logger.error(f"Background task failed: {error_message}")
        return []

def cleanup_file(file_path: str):
    """
    Cleans up (deletes) a file.  Used as a background task.
    """
    import os
    try:
        logger.info(f"Background task: Cleaning up file: {file_path}")
        os.remove(file_path)
        logger.info(f"Background task: Successfully deleted file: {file_path}")
    except Exception as e:
        logger.error(f"Background task: Error deleting file {file_path}: {e}")


def sanitize_for_csv_injection(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes a Pandas DataFrame to prevent CSV formula injection.
    Prepends a tab character to string cells that start with '=', '+', '-', or '@'.
    """
    def sanitize_cell(cell):
        if isinstance(cell, str) and cell.startswith(('=', '+', '-', '@')):
            return f'\t{cell}'
        return cell

    return data.applymap(sanitize_cell)

def process_csv_with_pandas(file_content: bytes) -> pd.DataFrame:
    """
    Processes the CSV data using Pandas. Add your specific logic here.
    """
    df = pd.read_csv(io.BytesIO(file_content), dtype=str)  # Read all as strings initially
    # Add your data processing logic here, e.g., type conversions, cleaning, etc.
    # Example: Convert 'column_with_numbers' to numeric if expected
    # if 'column_with_numbers' in df.columns:
    #     df['column_with_numbers'] = pd.to_numeric(df['column_with_numbers'], errors='coerce')
    return df
