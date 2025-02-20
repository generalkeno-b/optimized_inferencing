from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model when the application starts.
llm = Llama(
    model_path="models/vicuna-7b-v1.3.Q4_K_M.gguf",
    # n_gpu_layers=-1,  # Uncomment to use GPU acceleration
    # seed=1337,        # Uncomment to set a specific seed
    # n_ctx=2048,       # Uncomment to increase the context window
)

# Define a request schema using Pydantic.
class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 32  # default value

# Option 1: Serve the frontend at the root endpoint.
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Option 2: Alternatively, you can keep this endpoint as a simple JSON response.
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Llama model API!"}

@app.post("/generate")
def generate_text(request: QueryRequest):
    try:
        output = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            stop=["Q:", "\n"],
            echo=False
        )
        # Extract only the generated text
        generated_text = output['choices'][0]['text']
        return {"output": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
