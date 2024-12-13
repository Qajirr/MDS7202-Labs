# gradio_interface.py

import gradio as gr
import requests

# URL de la API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# Función para hacer la solicitud a la API
def predict_transaction(wallet_age, incoming_tx_count, outgoing_tx_count, net_incoming_tx_count, risky_tx_count, 
                        outgoing_tx_sum_eth, incoming_tx_sum_eth, min_eth_ever, total_balance_eth, 
                        risk_factor, total_collateral_eth, total_available_borrows_eth, avg_weighted_risk_factor, 
                        risk_factor_above_threshold_daily_count, avg_risk_factor, max_risk_factor, borrow_amount_sum_eth, 
                        borrow_amount_avg_eth, deposit_count, deposit_amount_sum_eth, withdraw_amount_sum_eth, liquidation_amount_sum_eth):
    
    data = {
        "wallet_age": wallet_age,
        "incoming_tx_count": incoming_tx_count,
        "outgoing_tx_count": outgoing_tx_count,
        "net_incoming_tx_count": net_incoming_tx_count,
        "risky_tx_count": risky_tx_count,
        "outgoing_tx_sum_eth": outgoing_tx_sum_eth,
        "incoming_tx_sum_eth": incoming_tx_sum_eth,
        "min_eth_ever": min_eth_ever,
        "total_balance_eth": total_balance_eth,
        "risk_factor": risk_factor,
        "total_collateral_eth": total_collateral_eth,
        "total_available_borrows_eth": total_available_borrows_eth,
        "avg_weighted_risk_factor": avg_weighted_risk_factor,
        "risk_factor_above_threshold_daily_count": risk_factor_above_threshold_daily_count,
        "avg_risk_factor": avg_risk_factor,
        "max_risk_factor": max_risk_factor,
        "borrow_amount_sum_eth": borrow_amount_sum_eth,
        "borrow_amount_avg_eth": borrow_amount_avg_eth,
        "deposit_count": deposit_count,
        "deposit_amount_sum_eth": deposit_amount_sum_eth,
        "withdraw_amount_sum_eth": withdraw_amount_sum_eth,
        "liquidation_amount_sum_eth": liquidation_amount_sum_eth
    }
    
    response = requests.post(API_URL, json=data)
    return response.json()

# Crear interfaz de Gradio
iface = gr.Interface(
    fn=predict_transaction,
    inputs=[
        gr.Slider(0, 100, step=1, label="Wallet Age"),
        gr.Slider(0, 1000, step=1, label="Incoming TX Count"),
        gr.Slider(0, 1000, step=1, label="Outgoing TX Count"),
        gr.Slider(0, 1000, step=1, label="Net Incoming TX Count"),
        gr.Slider(0, 1000, step=1, label="Risky TX Count"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Outgoing TX Sum (ETH)"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Incoming TX Sum (ETH)"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Min ETH Ever"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Total Balance (ETH)"),
        gr.Slider(0.0, 10.0, step=0.1, label="Risk Factor"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Total Collateral (ETH)"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Total Available Borrows (ETH)"),
        gr.Slider(0.0, 10.0, step=0.1, label="Avg Weighted Risk Factor"),
        gr.Slider(0, 100, step=1, label="Risk Factor Above Threshold Daily Count"),
        gr.Slider(0.0, 10.0, step=0.1, label="Avg Risk Factor"),
        gr.Slider(0.0, 10.0, step=0.1, label="Max Risk Factor"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Borrow Amount Sum (ETH)"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Borrow Amount Avg (ETH)"),
        gr.Slider(0, 1000, step=1, label="Deposit Count"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Deposit Amount Sum (ETH)"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Withdraw Amount Sum (ETH)"),
        gr.Slider(0.0, 1000.0, step=0.1, label="Liquidation Amount Sum (ETH)")
    ],
    outputs="json",
    title="Predicción de Transacciones Riesgosas",
    description="Interfaz para predecir si una transacción es riesgosa según múltiples características"
)

# Iniciar la interfaz
iface.launch()
