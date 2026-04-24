from fastapi import FastAPI, HTTPException
import pickle
import os
#from schema import News  # adatta se il path è diverso
from src.app.schema import News
    
import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import pandas as pd
from datetime import datetime


MODELS_DIR = "models"

app = FastAPI(title="Fake News Detector")

# GLOBAL STATE
model = None
vectorizer = None
current_model_name = None


FEEDBACK_PATH = os.path.join("data", "external", "feedback.csv")


def save_feedback(text: str, prediction: int, feedback: str):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "prediction": prediction,
        "model": current_model_name,
    }

    df = pd.DataFrame([row])

    if os.path.exists(FEEDBACK_PATH):
        df.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(FEEDBACK_PATH, index=False)
        
        
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
    

# API FUNCTIONS UI

def predict_ui(text):
    if not text.strip():
        return "⚠️ Inserisci un testo valido", "", None

    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])

    label = "REAL" if pred == 0 else "FAKE"
    emoji = "🟢" if pred == 0 else "🔴"

    return f"{emoji} {label}", f"Model: {current_model_name}", pred


def feedback_ui(text, pred, feedback):
    if pred is None:
        return "⚠️ No results founded"

    save_feedback(text, int(pred), feedback)
    return "✅ Feedback sent"


def switch_model_ui(model_name):
    try:
        load_model(model_name)
        return f"✅ {model_name} actual"
    except Exception as e:
        return f"❌ {str(e)}"


def get_models():
    return [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") and not f.startswith("vectorizer")
    ]


# UI

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Fake News Detector"
) as demo:

    # HEADER
    gr.Markdown("# Fake News Detector")
    gr.Markdown("---")
    with gr.Row():

        # LEFT PANEL (INPUT)
        with gr.Column(scale=2):

            gr.Markdown("### 📝 Input")

            text_input = gr.Textbox(
                placeholder="Insert a text news...",
                lines=10,
                show_label=False
            )

            predict_btn = gr.Button("Verifiy news", variant="primary")

        # RIGHT PANEL (RESULT)
        with gr.Column(scale=1):

            gr.Markdown("### 📊 Results")

            result_label = gr.Textbox(label="Prediction", interactive=False)
            model_info = gr.Textbox(label="Model", interactive=False)

    # FEEDBACK SECTION
    gr.Markdown("---")
    with gr.Row():

        feedback = gr.Radio(
            ["👍 Correct", "👎 Error"],
            label="Feedback"
        )

        save_btn = gr.Button(
            "💾 Salva",
            variant="secondary",
            scale=1
        )
        feedback_status = gr.Textbox(show_label=False)

    # MODEL CONTROL
    gr.Markdown("---")
    with gr.Accordion("⚙️ Models Options", open=False):

        model_dropdown = gr.Dropdown(
            choices=get_models(),
            value=current_model_name,
            label="Select model"
        )

        switch_btn = gr.Button("Load model")
        switch_status = gr.Textbox(show_label=False)

   
    hidden_pred = gr.State()

    # EVENTS
    predict_btn.click(
        fn=predict_ui,
        inputs=text_input,
        outputs=[result_label, model_info, hidden_pred]
    )

    save_btn.click(
        fn=feedback_ui,
        inputs=[text_input, hidden_pred, feedback],
        outputs=feedback_status
    )

    switch_btn.click(
        fn=switch_model_ui,
        inputs=model_dropdown,
        outputs=switch_status
    )


app = mount_gradio_app(app, demo, path="/ui")
