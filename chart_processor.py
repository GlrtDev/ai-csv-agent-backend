import pandas as pd
import json

def create_chartjs_data(df: pd.DataFrame, column_names: list, chart_type: str) -> dict:
    """
    Extracts specific columns from a pandas DataFrame and transforms them
    into Chart.js data format.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        column_names (list): A list of strings representing the column names to extract.
                             For 'pie' and 'doughnut' charts, this list should
                             contain exactly two column names: one for labels and one for values.
                             For other chart types, the first column is treated as labels
                             and subsequent columns as values.
        chart_type (str): The type of chart (e.g., "pie", "bar", "line", "doughnut").

    Returns:
        dict: A dictionary formatted for Chart.js.
    """
    chart_data = {
        "chartType": chart_type,
        "data": [],
        "labelsKey": "",
        "valuesKey": "Value",  # Default valueKey, can be adjusted for multiple value columns
        "options": {
            "scales": {
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Value"}},
                "x": {"title": {"display": True, "text": "Category"}}
            }
        }
    }

    convert_to_numeric(df)
    for col in column_names:
        df = remove_outliers_iqr(df, col)

    if not column_names:
        raise ValueError("column_names cannot be empty.")

    # Ensure all specified columns exist in the DataFrame
    missing_columns = [col for col in column_names if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {', '.join(missing_columns)}")

    if chart_type.lower() in ["pie", "doughnut"]:
        if len(column_names) != 2:
            raise ValueError(f"For '{chart_type}' charts, 'column_names' must contain exactly two columns: [labels_column, values_column].")
        
        labels_col = column_names[0]
        values_col = column_names[1]

        chart_data["labelsKey"] = labels_col
        chart_data["valuesKey"] = values_col

        # Create data array for pie/doughnut charts
        for index, row in df[[labels_col, values_col]].iterrows():
            chart_data["data"].append({
                f"{labels_col}": int(row[labels_col]) if pd.api.types.is_integer_dtype(df[labels_col]) else float(row[labels_col]),
                f"{values_col}": int(row[values_col]) if pd.api.types.is_integer_dtype(df[values_col]) else float(row[values_col])
            })
        
        # Pie/Doughnut charts don't typically have x and y scales in the same way bar/line charts do
        chart_data["options"].pop("scales", None)

    else: # For bar, line, and other charts
        labels_col = column_names[0]
        chart_data["labelsKey"] = labels_col
        
        # Populate labels for the 'data' structure
        chart_data["data"] = [{f"{labels_col}": int(val)} for val in df[labels_col].tolist()]

        # For multiple value columns, we need to restructure the 'data' and 'valuesKey'
        # The provided Chart.js format seems to imply a single 'value' field per data point.
        # For multi-series charts, Chart.js typically uses a 'datasets' array.
        # This implementation will focus on the provided single 'data' array structure
        # for simplicity, assuming the first column is labels and subsequent are values,
        # aggregating them into a single 'value' if multiple numeric columns are provided.
        # If you need multi-series, the 'data' and 'options' structure would need to be more complex.

        if len(column_names) > 1:
            value_columns = column_names[1:]
            
            # If there's only one value column, just add it directly
            if len(value_columns) == 1:
                chart_data["valuesKey"] = value_columns[0]
                for i, val in enumerate(df[value_columns[0]].tolist()):
                    chart_data["data"][i][f"{value_columns[0]}"] = int(val) if pd.api.types.is_integer_dtype(df[value_columns[0]]) else float(val)
            else:
                # If multiple value columns, this structure needs a more complex 'data' array
                # or a 'datasets' array. For the given output format, we'll sum them up
                # or you'll need to clarify how multiple value columns should be represented.
                # For now, let's assume we're adding them as separate properties within the 'name' object
                # or picking the first one. Let's adapt to the provided format:
                # {{"name": "string", "value": int}} -- this implies a single value.
                # So for multiple columns, we will need to decide which one is the "value".
                # For this example, let's assume the first *numeric* column after the label column is the 'value'.
                
                found_value_col = False
                for col in value_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        chart_data["valuesKey"] = col
                        for i, val in enumerate(df[col].tolist()):
                            chart_data["data"][i][f"{col}"] = int(val) if pd.api.types.is_integer_dtype(df[col]) else float(val)
                        found_value_col = True
                        break
                if not found_value_col:
                     raise ValueError("No numeric column found to represent 'value' in the selected columns.")

    chart_data["options"]["scales"]["x"]["title"]["text"] = labels_col
    
    chart_data["data"] = sorted(chart_data["data"], key= lambda item: list(item.values())[0])
    df_grouped = pd.DataFrame(chart_data["data"])
    numerical_cols = df_grouped.select_dtypes(include=['number']).columns.tolist()

    aggregate_by_col = column_names[0]

    if aggregate_by_col in numerical_cols:
        numerical_cols.remove(aggregate_by_col)

    #aggregation_functions = ['mean', 'max', 'min']
    aggregation_functions = ['mean']

    agg_dict = {col: aggregation_functions for col in numerical_cols}
    grouped_df = df_grouped.groupby(aggregate_by_col).agg(agg_dict)
    grouped_df.columns = [multiindex[0] for multiindex in grouped_df.columns.values]
    chart_data["data"] = grouped_df.reset_index().to_dict(orient='records')
    return chart_data


def convert_to_numeric(df: pd.DataFrame):
    for col in df.columns:
        try:
            # Attempt to convert to numeric, coercing errors to NaN
            s = pd.to_numeric(df[col], errors='coerce')

            # Check if there are any NaN values after conversion
            if s.isnull().any():
                print(f"Column '{col}' contains non-numeric values and cannot be fully converted to int without loss.")
                # Option 1: Keep the original column or handle the NaNs differently
                # df[col] = s # If you want to keep the NaN values
            else:
                # Check if all values are effectively integers (e.g., '8.0' can be int)
                # This ensures we don't convert floats like 9.1 to 9
                if all(x == int(x) for x in s if pd.notnull(x)): # Only check non-nulls
                    df[col] = s.astype(int)
                    print(f"Column '{col}' successfully converted to int.")
                else:
                    print(f"Column '{col}' contains float values that would lose precision if converted to int. Keeping as is.")

        except Exception as e:
            print(f"An unexpected error occurred while processing column '{col}': {e}")

def remove_outliers_iqr(dataframe, column_name):
    """
    Removes outliers from a specified column in a DataFrame using the IQR method.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to remove outliers from.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed from the specified column.
    """
    print(f"Removing outliers from column: {column_name}...")
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the rows where the column value is outside the bounds
    df_cleaned = dataframe[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)]
    print(f"Outliers removed from column: {column_name}. Rows remaining: {len(df_cleaned)}")
    return df_cleaned