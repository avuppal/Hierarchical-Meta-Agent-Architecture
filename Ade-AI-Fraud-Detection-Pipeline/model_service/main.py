from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI(title="ML Model Serving")

# Placeholder for loading the optimized ONNX model
try:
    sess = ort.InferenceSession("model.onnx") 
except Exception as e:
    print(f"Model loading failed: {e}")
    sess = None

@app.get("/health")
def health_check():
    return {"status": "OK", "model_loaded": sess is not None}

@app.post("/score")
def score_transaction(data: dict):
    if not sess:
        return {"error": "Model not loaded"}, 503
    
    # Placeholder: In a real system, 'data' would be preprocessed into model inputs
    dummy_input = np.random.rand(1, 10).astype(np.float32)
    
    # Simulate inference
    scores = sess.run(None, {sess.get_inputs()[0].name: dummy_input})
    
    return {"transaction_id": data.get("tx_id", "N/A"), "anomaly_score": float(scores[0][0][0])}