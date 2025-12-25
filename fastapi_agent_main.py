from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import json
import duckdb
import os
import uuid
from pathlib import Path
from openai import OpenAI
import tempfile
import shutil
from helper import get_openai_api_key

# Initialize FastAPI app
app = FastAPI(
    title="CSV Agent API",
    description="An AI agent that can analyze CSV datasets, generate SQL queries, and create visualizations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=get_openai_api_key())
MODEL = "gpt-4o-mini"

# Directory to store uploaded files
UPLOAD_DIR = Path("uploaded_datasets")
UPLOAD_DIR.mkdir(exist_ok=True)

# Store active sessions (in production, use Redis or a proper database)
active_sessions = {}


# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    visualization_code: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    filename: str
    columns: List[str]
    row_count: int


class VisualizationConfig(BaseModel):
    chart_type: str
    x_axis: str
    y_axis: str
    title: str


# Prompts
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
Use matplotlib and pandas. The data is already loaded as a pandas DataFrame called 'df'.
config: {config}
"""

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the user's uploaded dataset.
You have access to tools for database lookup, data analysis, and visualization generation.
"""


# Tool implementations
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt, 
        columns=columns, 
        table_name=table_name
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    
    return response.choices[0].message.content


def lookup_given_dataset(prompt: str, session_id: str) -> str:
    """Implementation of dataset lookup using SQL"""
    try:
        session = active_sessions.get(session_id)
        if not session:
            return "Error: No active session found. Please upload a dataset first."
        
        dataset_path = session["dataset_path"]
        table_name = session["table_name"]
        
        # Read the CSV file
        df = pd.read_csv(dataset_path)
        duckdb.sql(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

        # Generate SQL query
        sql_query = generate_sql_query(prompt, df.columns.tolist(), table_name)
        sql_query = sql_query.strip().replace("```sql", "").replace("```", "")
        
        # Execute the SQL query
        result = duckdb.sql(sql_query).df()
        
        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"


def analyze_given_dataset(prompt: str, data: str) -> str:
    """Implementation of AI-powered data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    
    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"


def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """Generate chart visualization configuration"""
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=data,
        visualization_goal=visualization_goal
    )

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        response_format=VisualizationConfig,
    )
    
    try:
        content = response.choices[0].message.parsed
        
        return {
            "chart_type": content.chart_type,
            "x_axis": content.x_axis,
            "y_axis": content.y_axis,
            "title": content.title,
            "data": data
        }
    except Exception:
        return {
            "chart_type": "line", 
            "x_axis": "index",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data
        }


def create_chart(config: dict) -> str:
    """Create chart code based on configuration"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    
    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").strip()
    
    return code


def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal"""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    return code


# Tool definitions for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_given_dataset",
            "description": "Look up data from the uploaded dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string", 
                        "description": "The unchanged prompt that the user provided."
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_given_dataset", 
            "description": "Analyze given data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string", 
                        "description": "The lookup_given_dataset tool's output."
                    },
                    "prompt": {
                        "type": "string", 
                        "description": "The unchanged prompt that the user provided."
                    }
                },
                "required": ["data", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python matplotlib code to create data visualizations",
            "parameters": {
                "type": "object", 
                "properties": {
                    "data": {
                        "type": "string", 
                        "description": "The lookup_given_dataset tool's output."
                    },
                    "visualization_goal": {
                        "type": "string", 
                        "description": "The goal of the visualization."
                    }
                },
                "required": ["data", "visualization_goal"]
            }
        }
    }
]


def handle_tool_calls(tool_calls, messages, session_id):
    """Execute tools and append results to messages"""
    tool_implementations = {
        "lookup_given_dataset": lambda **kwargs: lookup_given_dataset(session_id=session_id, **kwargs),
        "analyze_given_dataset": analyze_given_dataset, 
        "generate_visualization": generate_visualization
    }
    
    for tool_call in tool_calls:   
        function = tool_implementations[tool_call.function.name]
        function_args = json.loads(tool_call.function.arguments)
        result = function(**function_args)
        messages.append({
            "role": "tool", 
            "content": result, 
            "tool_call_id": tool_call.id
        })
        
    return messages


def run_agent(messages, session_id):
    """Main agent loop"""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
        
    # Add system prompt if needed
    if not any(msg.get("role") == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    visualization_code = None
    
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        
        messages.append(response.choices[0].message)
        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            # Check if visualization was generated
            for tool_call in tool_calls:
                if tool_call.function.name == "generate_visualization":
                    function_args = json.loads(tool_call.function.arguments)
                    visualization_code = generate_visualization(**function_args)
            
            messages = handle_tool_calls(tool_calls, messages, session_id)
        else:
            return response.choices[0].message.content, visualization_code


# API Endpoints
@app.post("/upload", response_model=SessionInfo)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset and create a new session"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read dataset info
        df = pd.read_csv(file_path)
        
        # Store session info
        active_sessions[session_id] = {
            "dataset_path": str(file_path),
            "table_name": f"dataset_{session_id.replace('-', '_')}",
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "messages": []
        }
        
        return SessionInfo(
            session_id=session_id,
            filename=file.filename,
            columns=df.columns.tolist(),
            row_count=len(df)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the agent"""
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = active_sessions[request.session_id]
        
        # Add user message to conversation history
        session["messages"].append({"role": "user", "content": request.message})
        
        # Run agent
        response_text, viz_code = run_agent(
            session["messages"].copy(), 
            request.session_id
        )
        
        # Update conversation history
        session["messages"].append({"role": "assistant", "content": response_text})
        
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            visualization_code=viz_code
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        filename=session["filename"],
        columns=session["columns"],
        row_count=session["row_count"]
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated data"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = active_sessions[session_id]
        
        # Delete uploaded file
        file_path = Path(session["dataset_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Remove session
        del active_sessions[session_id]
        
        return {"message": "Session deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CSV Agent API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload a CSV file",
            "chat": "POST /chat - Chat with the agent",
            "session": "GET /session/{session_id} - Get session info",
            "delete": "DELETE /session/{session_id} - Delete session"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)