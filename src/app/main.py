from fastapi import FastAPI, HTTPException
import pickle
import os
#from schema import News  # adatta se il path è diverso
from src.app.schema import News
MODELS_DIR = "models"

app = FastAPI(title="Fake News Detector")

# GLOBAL STATE
model = None
vectorizer = None
current_model_name = None


# LOAD FUNCTION
def load_model(model_name: str):
    global model, vectorizer, current_model_name

    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found")

    # carica modello
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # vectorizer fisso (puoi anche renderlo dinamico)
    with open(os.path.join(MODELS_DIR, "vectorizer_tfidf5000.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    current_model_name = model_name



load_model("logreg.pkl")


# MAIN PAGE
@app.get("/")
async def root():
    return {
        "message": "Fake News Detector API",
        "current_model": current_model_name
    }


# SWITCH MODEL
@app.get("/set_model")
async def set_model(model_name: str):


    try:
        load_model(model_name)
        return {
            "status": "ok",
            "loaded_model": model_name
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")


# LIST MODELS
@app.get("/models")
async def list_models():
    files = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") and not f.startswith("vectorizer")
    ]

    return {
        "available_models": files,
        "current_model": current_model_name
    }


# PREDICT REQUEST
@app.post("/predict", response_model=News)
async def predict(request: News):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = vectorizer.transform([request.text])
    pred = model.predict(X)[0]

    return News(
        text=request.text,
        label=int(pred)
    )