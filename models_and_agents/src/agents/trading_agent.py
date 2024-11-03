import torch
import json
import openai
from typing import Dict, List
from ..models.lstm_model import LSTMModel

def run_agent(timestep: int, coin_list: List[str], global_model: LSTMModel, 
             test_sequences: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Generate trading predictions for each coin."""
    model_predictions = {}
    global_model.eval()
    
    with torch.no_grad():
        for coin in coin_list:
            if coin not in test_sequences or timestep >= len(test_sequences[coin]):
                continue
                
            sequence = test_sequences[coin][timestep].unsqueeze(0)
            output = global_model(sequence)
            probs = torch.softmax(output, dim=1)[0]
            
            model_predictions[coin] = {
                'price_change_pct': (probs[2] - probs[0]) * 100,  # Buy - Sell probability
                'confidence_score': torch.max(probs).item()
            }
    
    return model_predictions

def construct_prompt(timestep: int, predictions: Dict[str, Dict[str, float]]) -> tuple:
    """Construct prompt for GPT model."""
    system_prompt = f"""
    Assistant, you are an advanced AI trading agent. Your objective is to maximize profits by making optimal trades based on real-time market conditions and a set of model predictions from a collaborative LSTM model.
    At each timestep, you will evaluate market conditions, interpret predictive signals from the model, and decide whether to buy, sell, or hold based on the following:

    Instructions:
    - Analyze the predicted price change percentages and confidence scores.
    - Use these probabilities to assess the likelihood of price movement in each asset.
    - Make decisions that maximize potential profits while minimizing risk.

    Function Call Instructions:
    - To execute a trade, use the following functions:
      - buy(coin: string, amount: number) - Buys the specified coin at the current price.
      - sell(coin: string, amount: number) - Sells the specified coin at the current price.
    - Respond in the following format:
      {{
          "action": "buy",
          "coin": "<coin_name>",
          "amount": <number>
      }}
    - If no trade is deemed necessary, respond with {{"action": "hold"}}.

    Market Data and Model Predictions:
    """

    for coin, pred in predictions.items():
        system_prompt += f"{coin.capitalize()}: Predicted price change: {pred['price_change_pct']:.2f}%, Confidence: {pred['confidence_score']*100:.1f}%\n"

    user_prompt = """
    Based on the provided predictions, select an appropriate action (buy, sell, or hold) for each asset, and provide a brief reasoning for your decision.
    Your decision-making should be based on maximizing the portfolio value by predicting market trends effectively.
    """

    return system_prompt, user_prompt

def get_trading_functions():
    """Define the trading functions for GPT model."""
    return [
        {
            "name": "buy",
            "description": "Buy the specified coin at the current price",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "The coin to buy"},
                    "amount": {"type": "number", "description": "The amount in USD to spend"}
                },
                "required": ["coin", "amount"]
            }
        },
        {
            "name": "sell",
            "description": "Sell the specified coin at the current price",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "The coin to sell"},
                    "amount": {"type": "number", "description": "The amount in USD worth to sell"}
                },
                "required": ["coin", "amount"]
            }
        }
    ]

def call_openai_api(system_prompt: str, user_prompt: str, functions: List[Dict]):
    """Call OpenAI API for trading decisions."""
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        functions=functions,
        function_call="auto"
    )
    return response

def process_api_response(api_response, portfolio: Dict[str, float], 
                        current_prices: Dict[str, float]) -> Dict[str, float]:
    """Process the API response and execute trades."""
    if api_response.choices[0].function_call:
        function_name = api_response.choices[0].function_call.name
        function_args = json.loads(api_response.choices[0].function_call.arguments)
        
        coin = function_args["coin"]
        usd_amount = function_args["amount"]
        
        if function_name == "buy" and portfolio.get("USD", 0) >= usd_amount:
            coin_price = current_prices[coin]
            coin_amount = usd_amount / coin_price
            portfolio["USD"] = portfolio.get("USD", 0) - usd_amount
            portfolio[coin] = portfolio.get(coin, 0) + coin_amount
            print(f"Bought ${usd_amount} worth of {coin} ({coin_amount} coins)")
            
        elif function_name == "sell":
            coin_price = current_prices[coin]
            if coin in portfolio:
                coin_amount = min(usd_amount / coin_price, portfolio[coin])
                portfolio[coin] -= coin_amount
                portfolio["USD"] = portfolio.get("USD", 0) + (coin_amount * coin_price)
                print(f"Sold {coin_amount} {coin} for ${coin_amount * coin_price}")
                if portfolio[coin] == 0:
                    del portfolio[coin]
    
    return portfolio

def simulate_trading(timestep: int, coin_list: List[str], global_model: LSTMModel, 
                    portfolio: Dict[str, float], test_sequences: Dict[str, torch.Tensor],
                    current_prices: Dict[str, float]) -> Dict[str, float]:
    """Simulate trading with the AI agent."""
    predictions = run_agent(timestep, coin_list, global_model, test_sequences)
    system_prompt, user_prompt = construct_prompt(timestep, predictions)
    functions = get_trading_functions()

    api_response = call_openai_api(system_prompt, user_prompt, functions)
    return process_api_response(api_response, portfolio, current_prices)