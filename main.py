from openai import AsyncOpenAI #type: ignore
from agents import Agent, OpenAIChatCompletionsModel, Runner ,function_tool ,ModelSettings #type: ignore
from dotenv import load_dotenv #type: ignore
import os
from pydantic import BaseModel #type: ignore
from fastapi import FastAPI, Request
load_dotenv()
app = FastAPI(title="QuickSeat Agent", description="QuickSeat Agent is a chatbot that can help you with your questions about QuickSeat.")

class Message(BaseModel):
    message: str


gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

agent = Agent(
    name="Assistant",
    instructions="You are an QuickSeat Assistant Which is a GCUF bus Service Your task is to handle User Quries .",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    model_settings=ModelSettings(
        temperature=2,
        max_tokens=1000,

    )    
)
history=[]




@app.post("/chat")
async def agent_endpoint(message: Message):
    query = message.message
    history.append({"role": "user", "content": query})
    if not query:
        return {"error": "Query is required"}
    
    try:
        result = await Runner.run(agent, history)
        history.append({"role": "assistant", "content": result.final_output})
        return {"response": result.final_output}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Welcome to the QuickSeat Agent API. Use the /chat endpoint to interact with the agent."}
