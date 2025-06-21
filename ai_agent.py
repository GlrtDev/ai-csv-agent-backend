import pandas as pd
from typing import Dict, Any, Optional
import json
import time
from chart_processor import create_chartjs_data
from dataModels.chartModels import ChartDataOutput, ChartOptions, ChartOptionsScales, ChartOptionsScalesAxis, ChartOptionsScalesAxisTitle
from custom_llama_cpp import CustomLlamaCLI # Assuming this is your custom LLM wrapper

YEARLY_COMPUTE_COST_IN_CENTS = 1500
HOURS_IN_YEAR = 365 * 24
HOURLY_COMPUTE_COST = YEARLY_COMPUTE_COST_IN_CENTS / HOURS_IN_YEAR
SECOND_COMPUTE_COST = HOURLY_COMPUTE_COST / 3600

class DataPlottingAgent:
    def __init__(self):
        self.llm = CustomLlamaCLI(
            n_predict=64,      # Increased prediction length for code generation
            threads=24,           # Number of threads for llama.cpp
            ctx_size=256,       # Increased context window size for more complex prompts/data
            temperature=0.5,     # Lower temperature for more deterministic code generation
            conversation=False
        )

    def _format_chart_data(self, df: pd.DataFrame, chart_type: str, labels_key: str, values_key: str) -> ChartDataOutput:
        """
        Helper to format data into the ChartDataOutput Pydantic model.
        This function assumes the LLM correctly identifies labels_key and values_key.
        """
        data_list = df.to_dict(orient="records")

        # Basic options inference
        chart_options = ChartOptions(
            scales=ChartOptionsScales(
                y=ChartOptionsScalesAxis(
                    beginAtZero=True,
                    title=ChartOptionsScalesAxisTitle(display=True, text=values_key)
                ),
                x=ChartOptionsScalesAxis(
                    title=ChartOptionsScalesAxisTitle(display=True, text=labels_key)
                )
            )
        )

        return ChartDataOutput(
            chartType=chart_type,
            data=data_list,
            labelsKey=labels_key,
            valuesKey=values_key,
            options=chart_options
        )

    def process_data_and_plot(self, df: pd.DataFrame, user_prompt: str) -> Dict[str, Any]:
        """
        Processes the DataFrame and user prompt by generating and executing
        pandas code, then formatting the result for charting and summary.
        """

        try:
            prompt_instructions = f"""You are an expert data analyst.
            Data with column names: {df.head(1)}
            User prompt: {user_prompt}.
            Question: What chart type (bar, line or pie) would you use for Data and what column namesUser prompt?
            Answear: """

            start_time = time.time()
            llm_response_raw = self.llm.invoke(prompt_instructions)
            end_time = time.time()
            duration = end_time - start_time
            llm_cost = SECOND_COMPUTE_COST * duration
            found_chart_types = { "bar" : -1, "line" : -1, "pie" : -1 }
            found_column_names = {col_name: -1 for col_name in df.columns.values}
            DataPlottingAgent.find_substrings_in_response(llm_response_raw, found_chart_types)
            DataPlottingAgent.find_substrings_in_response(llm_response_raw, found_column_names)
            filtered_chart_types = {k: v for k, v in found_chart_types.items() if v != -1}
            filtered_column_names = {k: v for k, v in found_column_names.items() if v != -1}
            sorted_chart_types = dict(sorted(filtered_chart_types.items(), key=lambda item: item[1]))
            sorted_column_names = dict(sorted(filtered_column_names.items(), key=lambda item: item[1]))
            choosen_chart_type = next(iter(sorted_chart_types))

            print(f"LLM Cost: ${llm_cost / 100} dollars")
            print(f"Raw LLM Response: {llm_response_raw}") # For debugging

            chart_data = create_chartjs_data(df, list(sorted_column_names.keys()), choosen_chart_type)
        
            return {
                "chart_data": chart_data,
                "summary": DataPlottingAgent.shorten_response(llm_response_raw),
                "error": None
            }

        except Exception as e:
            print(f"An unexpected error occurred during data processing: {e}")
            return {
                "chart_data": None,
                "summary": f"An unexpected error occurred: {e}",
                "error": "UNEXPECTED_ERROR"
            }

    @staticmethod
    def find_substrings_in_response(response_str: str, stringsToFind: dict) -> int:
        for key, _ in stringsToFind.items():
            stringsToFind[key] = response_str.find(key)

    @staticmethod
    def shorten_response(s: str):
        last_comma_index = s.rfind(',')
        if last_comma_index != -1:
            return s[:last_comma_index]
        else:
            return s