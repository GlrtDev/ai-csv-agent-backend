from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from pydantic import BaseModel
import logging
import uvicorn
from csv_processor import process_csv_with_pandas, sanitize_for_csv_injection
from jose import jwt
import secrets
from datetime import datetime, timedelta
import pandas as pd
from ai_agent import DataPlottingAgent
from dataModels.chartModels import AgentResponse, ChartDataOutput

app = FastAPI()

origins = [
    "http://localhost:5173", # dev
    "http://localhost:3000", # production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


security = HTTPBearer()
SECRET_KEY = "your-super-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
token_store: Dict[str, Dict] = {} # need to change that to secure cache

agent_processor = DataPlottingAgent()


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_data_by_token(token: str):
    if token in token_store:
        return token_store[token]["data"]
    return None

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI in Docker!"}


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)) -> Dict:
    """
    Endpoint to upload a CSV file, process it securely with Pandas, and return a token.
    """
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        # Efficiently check file size without reading the entire content into memory
        contents = await file.read()
        file_size = len(contents)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds the limit of 5 MB.")

        df = process_csv_with_pandas(contents)

        # Sanitize the DataFrame to prevent CSV injection
        sanitized_df = sanitize_for_csv_injection(df)

        # Generate a unique token
        access_token = create_access_token({"sub": secrets.token_urlsafe(16)})
        token_store[access_token] = {"data": sanitized_df, "timestamp": datetime.utcnow()}

        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException as e:
        raise e
    except Exception as e:
        error_message = f"An error occurred during CSV processing with Pandas: {e}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)

class ProcessDataRequest(BaseModel):
    prompt: str

@app.post("/send-prompt")
async def get_processed_data(
    request: ProcessDataRequest = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Endpoint to retrieve processed data using a valid token and prompt.
    Returns data as JSON or a string representation based on the 'format' parameter.
    """

    stored_data = await get_data_by_token(credentials.credentials)
    if stored_data is None or stored_data.empty:
        raise HTTPException(status_code=404, detail="Token not found or expired")

    result = agent_processor.process_data_and_plot(stored_data, request.prompt)

    return AgentResponse(
            chart_data=result["chart_data"],
            summary=result["summary"],
            error=result["error"]
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)