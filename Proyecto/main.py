import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.xgboost
import numpy as np
import gradio as gr
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import threading
import os
import pickle

# Cargar el modelo registrado con MLflow
app = FastAPI()

models_path = os.path.join("models", "best_model.pkl")
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
# Definir el esquema para los datos de entrada
class TransactionData(BaseModel):
    wallet_age: float
    incoming_tx_count: int
    outgoing_tx_count: int
    net_incoming_tx_count: int
    risky_tx_count: int
    outgoing_tx_sum_eth: float
    incoming_tx_sum_eth: float
    min_eth_ever: float
    total_balance_eth: float
    risk_factor: float
    total_collateral_eth: float
    total_available_borrows_eth: float
    avg_weighted_risk_factor: float
    risk_factor_above_threshold_daily_count: int
    avg_risk_factor: float
    max_risk_factor: float
    borrow_amount_sum_eth: float
    borrow_amount_avg_eth: float
    deposit_count: int
    deposit_amount_sum_eth: float
    withdraw_amount_sum_eth: float
    liquidation_amount_sum_eth: float

# Ruta de prueba para verificar la API
@app.get("/")
async def home():
    return {"message": "API de predicción de transacciones riesgosas"}

# Endpoint para hacer predicciones en FastAPI
@app.post("/predict")
async def predict(data: TransactionData):
    features = np.array([[
        data.wallet_age, data.incoming_tx_count, data.outgoing_tx_count,
        data.net_incoming_tx_count, data.risky_tx_count, data.outgoing_tx_sum_eth,
        data.incoming_tx_sum_eth, data.min_eth_ever, data.total_balance_eth,
        data.risk_factor, data.total_collateral_eth, data.total_available_borrows_eth,
        data.avg_weighted_risk_factor, data.risk_factor_above_threshold_daily_count,
        data.avg_risk_factor, data.max_risk_factor, data.borrow_amount_sum_eth,
        data.borrow_amount_avg_eth, data.deposit_count, data.deposit_amount_sum_eth,
        data.withdraw_amount_sum_eth, data.liquidation_amount_sum_eth
    ]])

    prediction = model.predict(features)[0]
    return {"target": int(prediction)}

# Crear la interfaz de Gradio
def gradio_predict(wallet_age, incoming_tx_count, outgoing_tx_count, net_incoming_tx_count, risky_tx_count, outgoing_tx_sum_eth,
                   incoming_tx_sum_eth, min_eth_ever, total_balance_eth, risk_factor, total_collateral_eth, 
                   total_available_borrows_eth, avg_weighted_risk_factor, risk_factor_above_threshold_daily_count, 
                   avg_risk_factor, max_risk_factor, borrow_amount_sum_eth, borrow_amount_avg_eth, deposit_count, 
                   deposit_amount_sum_eth, withdraw_amount_sum_eth, liquidation_amount_sum_eth):
    features = np.array([[
        wallet_age, incoming_tx_count, outgoing_tx_count, net_incoming_tx_count, risky_tx_count, outgoing_tx_sum_eth,
        incoming_tx_sum_eth, min_eth_ever, total_balance_eth, risk_factor, total_collateral_eth, total_available_borrows_eth,
        avg_weighted_risk_factor, risk_factor_above_threshold_daily_count, avg_risk_factor, max_risk_factor, borrow_amount_sum_eth,
        borrow_amount_avg_eth, deposit_count, deposit_amount_sum_eth, withdraw_amount_sum_eth, liquidation_amount_sum_eth
    ]])

    prediction = model.predict(features)[0]
    return int(prediction)

# Crear la interfaz de Gradio
gr_interface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Slider(minimum=0, maximum=100, label="Wallet Age"),
        gr.Slider(minimum=0, maximum=100, label="Incoming TX Count"),
        gr.Slider(minimum=0, maximum=100, label="Outgoing TX Count"),
        gr.Slider(minimum=0, maximum=100, label="Net Incoming TX Count"),
        gr.Slider(minimum=0, maximum=100, label="Risky TX Count"),
        gr.Slider(minimum=0, maximum=100, label="Outgoing TX Sum (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Incoming TX Sum (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Min ETH Ever"),
        gr.Slider(minimum=0, maximum=100, label="Total Balance (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Risk Factor"),
        gr.Slider(minimum=0, maximum=100, label="Total Collateral (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Total Available Borrows (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Avg Weighted Risk Factor"),
        gr.Slider(minimum=0, maximum=100, label="Risk Factor Above Threshold Daily Count"),
        gr.Slider(minimum=0, maximum=100, label="Avg Risk Factor"),
        gr.Slider(minimum=0, maximum=100, label="Max Risk Factor"),
        gr.Slider(minimum=0, maximum=100, label="Borrow Amount Sum (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Borrow Amount Avg (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Deposit Count"),
        gr.Slider(minimum=0, maximum=100, label="Deposit Amount Sum (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Withdraw Amount Sum (ETH)"),
        gr.Slider(minimum=0, maximum=100, label="Liquidation Amount Sum (ETH)")
    ],
    outputs="text"
)

# Crear un endpoint para servir la interfaz de Gradio en /gradio
@app.get("/gradio", response_class=HTMLResponse)
async def gradio_ui(request: Request):
    gradio_html = gr_interface.launch(share=False, inline=True)
    return gradio_html

# Iniciar el servidor cuando se ejecute el script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)