from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChartOptionsScalesAxisTitle(BaseModel):
    display: bool = True
    text: str

class ChartOptionsScalesAxis(BaseModel):
    beginAtZero: Optional[bool] = None
    title: Optional[ChartOptionsScalesAxisTitle] = None

class ChartOptionsScales(BaseModel):
    y: Optional[ChartOptionsScalesAxis] = None
    x: Optional[ChartOptionsScalesAxis] = None

class ChartOptions(BaseModel):
    scales: Optional[ChartOptionsScales] = None

class ChartDataOutput(BaseModel):
    chartType: str
    data: List[Dict[str, Any]]
    labelsKey: str
    valuesKey: str
    options: ChartOptions

class AgentResponse(BaseModel):
    chart_data: Optional[ChartDataOutput] = None
    summary: str
    error: Optional[str] = None