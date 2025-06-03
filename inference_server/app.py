# inference server using fastapi (serve on DGX)

import time
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import numpy as np
from transformers import AutoModel, AutoTokenizer

# FastAPI Initialization
app = FastAPI()

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Pretrained BERT Model and Tokenizer
bert_model_path = "/data/students/jeryl/TE/bert-task-regressor"
bert_model = AutoModel.from_pretrained(bert_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

# Load the KPCA and scaler
loaded_model_data = joblib.load("/data/students/jeryl/TE/best_kpca_model_v2.pkl")
loaded_scaler = loaded_model_data["scaler"]
loaded_kpca = loaded_model_data["kpca"]

# Define Bi-LSTM Model
class BiLSTMRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMRegressor, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)  # Bidirectional * 2

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# Load the BiLSTM Model
checkpoint_path = "/data/students/jeryl/TE/best_bilstm_model_with_params.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model_params = checkpoint['model_params']
bilstm_model = BiLSTMRegressor(
    input_dim=model_params['input_dim'],
    hidden_dim=model_params['hidden_dim'],
    num_layers=model_params['num_layers'],
    dropout=model_params['dropout']
).to(device)
bilstm_model.load_state_dict(checkpoint['model_state_dict'])
bilstm_model.eval()



class TaskRequest(BaseModel):
    task_name: str

def process_embeddings(task_name):
    inputs = tokenizer(task_name, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

@app.post("/predict/")
async def predict_task_duration(request: TaskRequest):
    if not request.task_name:
        raise HTTPException(status_code=400, detail="Task name cannot be empty")

    try:
        start_time = time.perf_counter()  # Start timer
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        embedding = process_embeddings(request.task_name).cpu().numpy().reshape(1, -1)
        embedding_scaled = loaded_scaler.transform(embedding)
        embedding_kpca = loaded_kpca.transform(embedding_scaled)
        embedding_kpca_tensor = torch.tensor(embedding_kpca, dtype=torch.float32).to(device).unsqueeze(0)

        with torch.no_grad():
            predicted_duration = bilstm_model(embedding_kpca_tensor).cpu().numpy()

        end_time = time.perf_counter()  # End timer
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        memory_used = mem_after - mem_before  # Calculate memory used

        return {
            "estimated_duration": float(round(predicted_duration[0][0], 2)),
            "inference_time": round(inference_time, 2),
            "memory_used": round(memory_used, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Run on DGX
