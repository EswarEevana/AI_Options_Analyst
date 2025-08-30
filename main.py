import requests
import json
import os
import textwrap
import numpy as np
import uuid
import re
import pandas as pd

#  Installation Note 
# This script requires additional libraries. Please install them using pip:
# pip install requests sentence-transformers numpy rich chromadb yfinance pandas

# Attempt to import heavy libraries and provide a helpful error message if they are missing.
try:
    from sentence_transformers import SentenceTransformer
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.status import Status
    import chromadb
    import yfinance as yf
except ImportError:
    print("Error: Required libraries are not installed.")
    print("Please run: pip install sentence-transformers numpy rich chromadb yfinance pandas")
    exit()


#  Configuration 
API_KEY = os.getenv("GEMINI_API_KEY", "") 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
HEADERS = {"Content-Type": "application/json"}
pd.set_option('display.max_colwidth', None) # Prevent yfinance data from being truncated

#  UI Component 
console = Console()

#  1. Knowledge Base Part A: Options Strategies 
# This remains the same, providing the strategic knowledge.
options_strategies_data = {
    "covered_call": {
        "name": "Covered Call",
        "outlook": "Moderately bullish to neutral",
        "description": "Sell an out-of-the-money (OTM) call option against shares of the underlying stock you already own.",
        "risk_reward": "Limited profit (premium + stock appreciation up to strike), limited loss (stock depreciation offset by premium).",
        "tags": ["bullish", "neutral", "income", "equity"],
        "historical_examples": [
            {
                "scenario": "RELIANCE after strong quarterly results, with high implied volatility.",
                "entry_conditions": "Owned a lot of RELIANCE shares (e.g., 250 shares) at ₹2850. Stock was trading sideways but fundamentally strong.",
                "trade_setup": "Sold 1 lot of RELIANCE ₹3000 Call (30 DTE) for ₹40 premium.",
                "outcome": "RELIANCE moved to ₹2950 at expiration. The call expired worthless. Kept the premium of ₹10,000, boosting returns on the stock holding."
            }
        ]
    },
    "protective_put": {
        "name": "Protective Put",
        "outlook": "Bullish, with downside protection",
        "description": "Buy a put option for shares of stock you already own. This acts like insurance, setting a floor on your potential losses.",
        "risk_reward": "Unlimited profit potential (stock appreciation), limited loss (cost of put option + stock depreciation down to strike).",
        "tags": ["bullish", "hedging", "insurance", "equity"],
        "historical_examples": [
            {
                "scenario": "Hedging a large HDFCBANK position through a volatile pre-budget period, despite positive long-term fundamentals.",
                "entry_conditions": "Owned 550 shares of HDFCBANK at ₹1500. Balance sheet looked strong, but market sentiment was nervous.",
                "trade_setup": "Bought 1 lot of HDFCBANK ₹1450 Put (60 DTE) for ₹25.",
                "outcome": "HDFCBANK dropped to ₹1420 due to market panic. The put's value increased, offsetting the temporary stock loss."
            }
        ]
    },
     "bull_put_spread": {
        "name": "Bull Put Spread",
        "outlook": "Moderately bullish",
        "description": "A vertical credit spread strategy. Sell a higher-strike put option and simultaneously buy a lower-strike put option with the same expiration.",
        "risk_reward": "Limited profit (net credit received), limited loss (difference between strikes minus the net credit).",
        "tags": ["bullish", "credit", "spread", "limited_risk"],
        "historical_examples": [
            {
                "scenario": "Trading a slow, upward-trending, fundamentally sound stock like ITC.",
                "entry_conditions": "ITC trading at ₹430, showing strong support at the ₹420 level and consistent dividend history.",
                "trade_setup": "Sold 1 lot of ITC ₹420 Put and Bought 1 lot of ITC ₹410 Put (45 DTE) for a net credit of ₹3.",
                "outcome": "ITC closed at ₹435 at expiration. Both puts expired worthless. Kept the full credit for a high-probability win based on technical support and company stability."
            }
        ]
    },
    "bear_call_spread": {
        "name": "Bear Call Spread",
        "outlook": "Moderately bearish",
        "description": "A vertical credit spread strategy. Sell a lower-strike call option and simultaneously buy a higher-strike call option with the same expiration.",
        "risk_reward": "Limited profit (net credit received), limited loss (difference between strikes minus the net credit).",
        "tags": ["bearish", "credit", "spread", "limited_risk"],
         "historical_examples": [
            {
                "scenario": "A stock like ZOMATO has had a massive run-up and is showing signs of exhaustion at a technical resistance level, with questionable profitability.",
                "entry_conditions": "ZOMATO trading at ₹180, failing to break above the ₹185 resistance. P/E ratio is extremely high.",
                "trade_setup": "Sold 1 lot of ZOMATO ₹190 Call and Bought 1 lot of ZOMATO ₹200 Call (30 DTE) for a net credit of ₹2.50.",
                "outcome": "ZOMATO pulled back to ₹175 at expiration. Both calls expired worthless. Kept the credit."
            }
        ]
    },
}

#  2. Knowledge Base Part B: Live Financial Data Fetcher 
# A list of Nifty 50 tickers + the Nifty Index itself.
NIFTY_50_TICKERS = {
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "HINDUNILVR", "ITC", "KOTAKBANK", "SBIN",
    "BHARTIARTL", "BAJFINANCE", "LT", "AXISBANK", "ASIANPAINT", "HCLTECH", "MARUTI", "SUNPHARMA",
    "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "BAJAJFINSV", "TATASTEEL", "POWERGRID",
    "INDUSINDBK", "M&M", "NTPC", "TECHM", "TATAMOTORS", "HDFCLIFE", "JSWSTEEL", "ADANIPORTS",
    "GRASIM", "BAJAJ-AUTO", "HINDALCO", "ONGC", "DIVISLAB", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "UPL", "BPCL", "SHREECEM", "TATACONSUM", "HEROMOTOCO", "BRITANNIA", "SBILIFE",
    "ADANIENT", "NIFTY", "NIFTY50", "NIFTY 50"
}
NIFTY_INDEX_TICKER = "^NSEI"

def get_live_financial_data(ticker_symbol: str) -> str:
    """
    Fetches live financial and options data for a given stock ticker using yfinance.
    """
    try:
        # Format ticker for yfinance (append .NS for stocks, use special for index)
        if ticker_symbol.upper() in ["NIFTY", "NIFTY50", "NIFTY 50"]:
             yf_ticker_str = NIFTY_INDEX_TICKER
        else:
             yf_ticker_str = f"{ticker_symbol.upper()}.NS"

        ticker = yf.Ticker(yf_ticker_str)
        info = ticker.info

        #  Build Financial Context String 
        current_price = info.get('regularMarketPrice', 'N/A')
        high = info.get('fiftyTwoWeekHigh', 'N/A')
        low = info.get('fiftyTwoWeekLow', 'N/A')
        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 'N/A'
        
        financial_context = (
            f"\n\n LIVE FINANCIAL SNAPSHOT FOR {ticker_symbol.upper()} \n"
            f"**Fundamental Summary:** Revenue Growth: {info.get('revenueGrowth', 'N/A') * 100:.2f}% | Return on Equity: {roe:.2f}% | Total Debt: {info.get('totalDebt', 'N/A')}\n"
            f"**Technical Summary:** Current Price: ₹{current_price} | 52-Week High: ₹{high} | 52-Week Low: ₹{low}\n"
        )
        
        #  Options Chain Snapshot
        expirations = ticker.options
        if expirations:
            opt = ticker.option_chain(expirations[0])
            # Find the at-the-money (ATM) strike
            atm_strike = min(opt.calls['strike'], key=lambda x: abs(x - current_price))
            
            atm_call = opt.calls[opt.calls['strike'] == atm_strike]
            atm_put = opt.puts[opt.puts['strike'] == atm_strike]
            
            call_premium = atm_call['lastPrice'].iloc[0] if not atm_call.empty else 'N/A'
            put_premium = atm_put['lastPrice'].iloc[0] if not atm_put.empty else 'N/A'
            
            financial_context += f"**Options Snapshot (Expiry: {expirations[0]}):** ATM Strike: ₹{atm_strike} | ATM Call Premium: ₹{call_premium} | ATM Put Premium: ₹{put_premium}\n"
        
        financial_context += "-\n"
        return financial_context

    except Exception as e:
        # console.print(f"[bold red]Could not fetch data for {ticker_symbol}: {e}[/bold red]")
        return f"\n Could not retrieve live financial data for {ticker_symbol}. It might be an invalid ticker. \n"


# 3. RAG System Components (with ChromaDB)
embedding_model = None
chroma_collection = None

def initialize_rag_system():
    global embedding_model, chroma_collection
    client = chromadb.Client()
    chroma_collection = client.get_or_create_collection("options_strategies_v3")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    documents_to_add = []
    ids_to_add = []
    
    for key, strategy in options_strategies_data.items():
        existing_doc = chroma_collection.get(ids=[key])
        if existing_doc['ids']:
            continue
            
        historical_data_str = "\n\nHistorical Scenarios:\n" + "".join(
            [f"- Scenario: {ex['scenario']}\n  Entry: {ex['entry_conditions']}\n  Setup: {ex['trade_setup']}\n  Outcome: {ex['outcome']}\n" 
             for ex in strategy.get("historical_examples", [])]
        )

        doc_content = (
            f"Strategy Name: {strategy['name']}\n"
            f"Market Outlook: {strategy['outlook']}\n"
            f"Description: {strategy['description']}\n"
            f"Risk/Reward: {strategy['risk_reward']}\n"
            f"Tags: {', '.join(strategy['tags'])}"
            f"{historical_data_str}"
        )
        documents_to_add.append(doc_content)
        ids_to_add.append(key)

    if documents_to_add:
        console.print(f"Found {len(documents_to_add)} new strategies to add to the knowledge base...")
        embeddings = embedding_model.encode(documents_to_add).tolist()
        chroma_collection.add(embeddings=embeddings, documents=documents_to_add, ids=ids_to_add)
        console.print("[bold green]Knowledge base updated.[/bold green]")

    console.print("[bold green]RAG system with ChromaDB is ready.[/bold green]")


#  4. Large Language Model (LLM) Integration 
def generate_llm_response(prompt: str) -> str:
    if not API_KEY:
        return "Error: API_KEY is not set. Please set the GEMINI_API_KEY environment variable."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        candidate = result.get("candidates", [{}])[0]
        return candidate.get("content", {}).get("parts", [{}])[0].get("text", "I couldn't generate a response.")
    except requests.exceptions.RequestException as e:
        return f"Sorry, the API request failed: {e}"
    except (KeyError, IndexError):
        return f"Sorry, I received an unexpected response from the API."

#  5. The RAG Pipeline Function (Upgraded for Live Data) 
def rag_pipeline(query: str) -> str:
    #  Part 1: Retrieve Options Strategies 
    query_embedding = embedding_model.encode([query]).tolist()
    results = chroma_collection.query(query_embeddings=query_embedding, n_results=2)
    retrieved_docs = results['documents'][0]
    strategy_context = "\n\n\n\n".join(retrieved_docs)

    #  Part 2: Retrieve LIVE Financial Data if a stock is mentioned 
    financial_context = ""
    found_ticker = None
    for ticker in NIFTY_50_TICKERS:
        if re.search(r'\b' + ticker + r'\b', query, re.IGNORECASE):
            found_ticker = ticker
            break
    
    if found_ticker:
        with console.status(f"[bold yellow]Fetching live data for {found_ticker}...[/]", spinner="earth"):
            financial_context = get_live_financial_data(found_ticker)

    #  Part 3: Augment Prompt and Generate 
    prompt = (
        "You are an expert AI techno-fundamental analyst for the INDIAN stock market. Your goal is to provide a detailed, actionable trading idea by synthesizing both strategic options knowledge and LIVE financial data.\n\n"
        "Analyze the user's query and the two contexts provided (Options Strategies and Live Financial Snapshot).\n"
        "1.  **If the query is a financial question:**\n"
        "    a. First, analyze the **Live Financial Snapshot**. Is the company fundamentally strong (e.g., good ROE)? Is it trading near a high or low? Use this LIVE data to form a market view (bullish, bearish, neutral).\n"
        "    b. Then, select the most appropriate strategy from the **Options Strategies Context** that aligns with that market view.\n"
        "    c. Construct a plausible, illustrative trade setup using the LIVE numbers from the snapshot. Explain your reasoning clearly, linking the strategy choice to the live financial data.\n"
        "    d. **Always state that this is a simulated example based on live data, not financial advice.**\n"
        "2.  **If the query is NOT a financial question** (e.g., 'thank you', 'ok'): IGNORE the contexts and give a brief, friendly, conversational response.\n\n"
        f" USER QUERY \n{query}\n\n"
        f" CONTEXT 1: OPTIONS STRATEGIES \n{strategy_context}\n"
        f" CONTEXT 2: LIVE FINANCIAL SNAPSHOT \n{financial_context if financial_context else 'No specific Nifty 50 stock mentioned.'}\n\n"
        " RESPONSE "
    )
    
    response = generate_llm_response(prompt)
    return response

#  6. User Interface 
def display_suggestions():
    suggestions_text = (
        "[bold]Here are a few things you can ask:[/bold]\n"
        "1. I own RELIANCE and want to generate income. What should I do?\n"
        "2. How can I hedge my HDFCBANK stock? Give me a full trade setup.\n"
        "3. What's a good strategy for INFY right now?\n"
        "4. Analyze the NIFTY index and suggest a strategy.\n\n"
        "[italic]You can type a number or your own question about any Nifty 50 stock.[/italic]"
    )
    console.print(Panel(suggestions_text, title="Suggestions", border_style="blue", title_align="left"))

def main():
    console.print(Panel.fit("[bold] AI Live Market Analyst for Indian Options [/bold]", style="bold blue"))
    with console.status("[bold green]Initializing RAG system...[/]", spinner="dots"):
        initialize_rag_system()
    
    console.print("\nHello! I'm your AI analyst for Indian options trading.", style="green")
    console.print("I can now fetch LIVE data for any Nifty 50 stock or the Nifty index.", style="yellow")
    console.print("Type 'quit' or 'exit' to end the chat.", style="yellow")
    
    display_suggestions()

    suggestion_map = {
        "1": "I own RELIANCE and want to generate income. What should I do?",
        "2": "How can I hedge my HDFCBANK stock? Give me a full trade setup.",
        "3": "What's a good strategy for INFY right now?",
        "4": "Analyze the NIFTY index and suggest a strategy."
    }

    while True:
        user_input = console.input("\n[bold cyan]You: [/bold cyan]").strip()
        if user_input.lower() in ["quit", "exit"]:
            console.print("\n[green]Assistant:[/green] Goodbye! Always trade responsibly.", style="bold")
            break
        if user_input in suggestion_map:
            user_input = suggestion_map[user_input]
            console.print(Panel(user_input, title="You (selected)", title_align="left", border_style="cyan"))
        
        # We don't need a separate "thinking" status as the data fetching status serves this purpose.
        final_response = rag_pipeline(user_input)
        
        response_markdown = Markdown(final_response)
        console.print(Panel(response_markdown, title="Analyst's Recommendation", title_align="left", border_style="green"))

if __name__ == "__main__":
    main()

