"""
🌙 Moon Dev's RBI AI v2.0 (Research-Backtest-Implement-Execute)
Built with love by Moon Dev 🚀

NEW IN v2.0: EXECUTION LOOP! 
- Automatically executes backtests
- Captures errors and stats
- Loops back to debug agent on failures
- Continues until success!

Required Setup:
1. Same folder structure as v1
2. Conda environment 'tflow' with backtesting packages
3. Everything else is automated!
"""

# Standard library imports
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# Third-party imports
import openai

# Standard library from imports
from datetime import datetime
from itertools import cycle
from pathlib import Path

# Third-party from imports
from anthropic import Anthropic
from dotenv import load_dotenv
from termcolor import cprint

# Load environment variables FIRST
load_dotenv()
print("✅ Environment variables loaded")

# Add config values directly to avoid import issues
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 4000

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models.model_priority import ModelPriority, model_priority_queue
    print("✅ Successfully imported model_priority")
except ImportError as e:
    print(f"⚠️ Could not import model_priority: {e}")
    sys.exit(1)

# Model Priority System - Automatic fallback: GPT-5 → Claude → Gemini
# HIGH priority: Best models for complex tasks (research, backtest, debug)
# Uses model_priority_queue with automatic fallback!

# Execution Configuration
CONDA_ENV = "tflow"  # Your conda environment
MAX_DEBUG_ITERATIONS = 10  # Max times to try debugging
EXECUTION_TIMEOUT = 300  # 5 minutes

# DeepSeek Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Get today's date for organizing outputs
TODAY_DATE = datetime.now().strftime("%m_%d_%Y")

# Update data directory paths - V2 uses separate folder structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/rbi_v2"  # NEW: Separate V2 folder
TODAY_DIR = DATA_DIR / TODAY_DATE
RESEARCH_DIR = TODAY_DIR / "research"
BACKTEST_DIR = TODAY_DIR / "backtests"
PACKAGE_DIR = TODAY_DIR / "backtests_package"
FINAL_BACKTEST_DIR = TODAY_DIR / "backtests_final"
CHARTS_DIR = TODAY_DIR / "charts"
EXECUTION_DIR = TODAY_DIR / "execution_results"  # NEW!
PROCESSED_IDEAS_LOG = DATA_DIR / "processed_ideas.log"

# IDEAS file is now in the V2 folder 
IDEAS_FILE = DATA_DIR / "ideas.txt"

# Create main directories if they don't exist
for dir in [DATA_DIR, TODAY_DIR, RESEARCH_DIR, BACKTEST_DIR, PACKAGE_DIR, 
            FINAL_BACKTEST_DIR, CHARTS_DIR, EXECUTION_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# All prompts (same as v1)
RESEARCH_PROMPT = """
You are Moon Dev's Research AI 🌙

IMPORTANT NAMING RULES:
1. Create a UNIQUE TWO-WORD NAME for this specific strategy
2. The name must be DIFFERENT from any generic names like "TrendFollower" or "MomentumStrategy"
3. First word should describe the main approach (e.g., Adaptive, Neural, Quantum, Fractal, Dynamic)
4. Second word should describe the specific technique (e.g., Reversal, Breakout, Oscillator, Divergence)
5. Make the name SPECIFIC to this strategy's unique aspects

Examples of good names:
- "AdaptiveBreakout" for a strategy that adjusts breakout levels
- "FractalMomentum" for a strategy using fractal analysis with momentum
- "QuantumReversal" for a complex mean reversion strategy
- "NeuralDivergence" for a strategy focusing on divergence patterns

BAD names to avoid:
- "TrendFollower" (too generic)
- "SimpleMoving" (too basic)
- "PriceAction" (too vague)

Output format must start with:
STRATEGY_NAME: [Your unique two-word name]

Then analyze the trading strategy content and create detailed instructions.
Focus on:
1. Key strategy components
2. Entry/exit rules
3. Risk management
4. Required indicators

Your complete output must follow this format:
STRATEGY_NAME: [Your unique two-word name]

STRATEGY_DETAILS:
[Your detailed analysis]

Remember: The name must be UNIQUE and SPECIFIC to this strategy's approach!
"""

BACKTEST_PROMPT = """
You are Moon Dev's Backtest AI 🌙 ONLY SEND BACK CODE, NO OTHER TEXT.
Create a backtesting.py implementation for the strategy.
USE BACKTESTING.PY
Include:
1. All necessary imports
2. Strategy class with indicators
3. Entry/exit logic
4. Risk management
5. your size should be 1,000,000
6. If you need indicators use TA lib or pandas TA.

IMPORTANT DATA HANDLING:
1. Clean column names by removing spaces: data.columns = data.columns.str.strip().str.lower()
2. Drop any unnamed columns: data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
3. Ensure proper column mapping to match backtesting requirements:
   - Required columns: 'Open', 'High', 'Low', 'Close', 'Volume'
   - Use proper case (capital first letter)

FOR THE PYTHON BACKTESTING LIBRARY USE BACKTESTING.PY AND SEND BACK ONLY THE CODE, NO OTHER TEXT.

INDICATOR CALCULATION RULES:
1. ALWAYS use self.I() wrapper for ANY indicator calculations
2. Use talib functions instead of pandas operations:
   - Instead of: self.data.Close.rolling(20).mean()
   - Use: self.I(talib.SMA, self.data.Close, timeperiod=20)
3. For swing high/lows use talib.MAX/MIN:
   - Instead of: self.data.High.rolling(window=20).max()
   - Use: self.I(talib.MAX, self.data.High, timeperiod=20)

BACKTEST EXECUTION ORDER:
1. Run initial backtest with default parameters first
2. Print full stats using print(stats) and print(stats._strategy)
3. no optimization code needed, just print the final stats, make sure full stats are printed, not just part or some. stats = bt.run() print(stats) is an example of the last line of code. no need for plotting ever.

do not creeate charts to plot this, just print stats. no charts needed.

CRITICAL POSITION SIZING RULE:
When calculating position sizes in backtesting.py, the size parameter must be either:
1. A fraction between 0 and 1 (for percentage of equity)
2. A whole number (integer) of units

The common error occurs when calculating position_size = risk_amount / risk, which results in floating-point numbers. Always use:
position_size = int(round(position_size))

Example fix:
❌ self.buy(size=3546.0993)  # Will fail
✅ self.buy(size=int(round(3546.0993)))  # Will work

RISK MANAGEMENT:
1. Always calculate position sizes based on risk percentage
2. Use proper stop loss and take profit calculations
4. Print entry/exit signals with Moon Dev themed messages

If you need indicators use TA lib or pandas TA. 

Use this data path: /Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv
the above data head looks like below
datetime, open, high, low, close, volume,
2023-01-01 00:00:00, 16531.83, 16532.69, 16509.11, 16510.82, 231.05338022,
2023-01-01 00:15:00, 16509.78, 16534.66, 16509.11, 16533.43, 308.12276951,

Always add plenty of Moon Dev themed debug prints with emojis to make debugging easier! 🌙 ✨ 🚀

FOR THE PYTHON BACKTESTING LIBRARY USE BACKTESTING.PY AND SEND BACK ONLY THE CODE, NO OTHER TEXT.
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

DEBUG_PROMPT = """
You are Moon Dev's Debug AI 🌙
Fix technical issues in the backtest code WITHOUT changing the strategy logic.

CRITICAL ERROR TO FIX:
{error_message}

CRITICAL DATA LOADING REQUIREMENTS:
The CSV file has these exact columns after processing:
- datetime, open, high, low, close, volume (all lowercase after .str.lower())
- After capitalization: Datetime, Open, High, Low, Close, Volume

CRITICAL BACKTESTING REQUIREMENTS:
1. Data Loading Rules:
   - Use data.columns.str.strip().str.lower() to clean columns
   - Drop unnamed columns: data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
   - Rename columns properly: data.rename(columns={{'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}})
   - Set datetime as index: data = data.set_index(pd.to_datetime(data['datetime']))

2. Position Sizing Rules:
   - Must be either a fraction (0 < size < 1) for percentage of equity
   - OR a positive whole number (round integer) for units
   - NEVER use floating point numbers for unit-based sizing

3. Indicator Issues:
   - Cannot use .shift() on backtesting indicators
   - Use array indexing like indicator[-2] for previous values
   - All indicators must be wrapped in self.I()

4. Position Object Issues:
   - Position object does NOT have .entry_price attribute
   - Use self.trades[-1].entry_price if you need entry price from last trade
   - Available position attributes: .size, .pl, .pl_pct
   - For partial closes: use self.position.close() without parameters (closes entire position)
   - For stop losses: use sl= parameter in buy/sell calls, not in position.close()

5. No Trades Issue (Signals but no execution):
   - If strategy prints "ENTRY SIGNAL" but shows 0 trades, the self.buy() call is not executing
   - Common causes: invalid size parameter, insufficient cash, missing self.buy() call
   - Ensure self.buy() is actually called in the entry condition block
   - Check size parameter: must be fraction (0-1) or positive integer
   - Verify cash/equity is sufficient for the trade size

Focus on:
1. KeyError issues with column names
2. Syntax errors and import statements
3. Indicator calculation methods
4. Data loading and preprocessing
5. Position object attribute errors (.entry_price, .close() parameters)

DO NOT change strategy logic, entry/exit conditions, or risk management rules.

Return the complete fixed code with Moon Dev themed debug prints! 🌙 ✨
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

PACKAGE_PROMPT = """
You are Moon Dev's Package AI 🌙
Your job is to ensure the backtest code NEVER uses ANY backtesting.lib imports or functions.

❌ STRICTLY FORBIDDEN:
1. from backtesting.lib import *
2. import backtesting.lib
3. from backtesting.lib import crossover
4. ANY use of backtesting.lib

✅ REQUIRED REPLACEMENTS:
1. For crossover detection:
   Instead of: backtesting.lib.crossover(a, b)
   Use: (a[-2] < b[-2] and a[-1] > b[-1])  # for bullish crossover
        (a[-2] > b[-2] and a[-1] < b[-1])  # for bearish crossover

2. For indicators:
   - Use talib for all standard indicators (SMA, RSI, MACD, etc.)
   - Use pandas-ta for specialized indicators
   - ALWAYS wrap in self.I()

3. For signal generation:
   - Use numpy/pandas boolean conditions
   - Use rolling window comparisons with array indexing
   - Use mathematical comparisons (>, <, ==)

Example conversions:
❌ from backtesting.lib import crossover
❌ if crossover(fast_ma, slow_ma):
✅ if fast_ma[-2] < slow_ma[-2] and fast_ma[-1] > slow_ma[-1]:

❌ self.sma = self.I(backtesting.lib.SMA, self.data.Close, 20)
✅ self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20)

IMPORTANT: Scan the ENTIRE code for any backtesting.lib usage and replace ALL instances!
Return the complete fixed code with proper Moon Dev themed debug prints! 🌙 ✨
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

def execute_backtest(file_path: str, strategy_name: str) -> dict:
    """
    Execute a backtest file in conda environment and capture output
    This is the NEW MAGIC! 🚀
    """
    cprint(f"\n🚀 Executing backtest: {strategy_name}", "cyan")
    cprint(f"📂 File: {file_path}", "cyan")
    cprint(f"🐍 Using conda env: {CONDA_ENV}", "cyan")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Build the command
    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "python", str(file_path)
    ]
    
    start_time = datetime.now()
    
    # Run the backtest
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=EXECUTION_TIMEOUT
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    output = {
        "success": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save execution results
    result_file = EXECUTION_DIR / f"{strategy_name}_{datetime.now().strftime('%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    # Print results
    if output['success']:
        cprint(f"✅ Backtest executed successfully in {execution_time:.2f}s!", "green")
        if output['stdout']:
            cprint("\n📊 BACKTEST RESULTS:", "green")
            print(output['stdout'])
    else:
        cprint(f"❌ Backtest failed with return code: {output['return_code']}", "red")
        if output['stderr']:
            cprint("\n🐛 ERRORS:", "red")
            print(output['stderr'])
    
    return output

def parse_execution_error(execution_result: dict) -> str:
    """Extract meaningful error message for debug agent"""
    if execution_result.get('stderr'):
        stderr = execution_result['stderr'].strip()
        
        # Return the full stderr for better debugging context
        # This includes the full Python traceback, not just the conda error
        return stderr
    return execution_result.get('error', 'Unknown error')

def get_idea_hash(idea: str) -> str:
    """Generate a unique hash for an idea to track processing status"""
    # Create a hash of the idea to use as a unique identifier
    return hashlib.md5(idea.encode('utf-8')).hexdigest()

def is_idea_processed(idea: str) -> bool:
    """Check if an idea has already been processed"""
    if not PROCESSED_IDEAS_LOG.exists():
        return False
        
    idea_hash = get_idea_hash(idea)
    
    with open(PROCESSED_IDEAS_LOG, 'r', encoding='utf-8') as f:
        processed_hashes = [line.strip().split(',')[0] for line in f if line.strip()]

    return idea_hash in processed_hashes

def log_processed_idea(idea: str, strategy_name: str = "Unknown") -> None:
    """Log an idea as processed with timestamp and strategy name"""
    idea_hash = get_idea_hash(idea)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the log file if it doesn't exist
    if not PROCESSED_IDEAS_LOG.exists():
        PROCESSED_IDEAS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_IDEAS_LOG, 'w', encoding='utf-8') as f:
            f.write("# Moon Dev's RBI AI - Processed Ideas Log 🌙\n")
            f.write("# Format: hash,timestamp,strategy_name,idea_snippet\n")

    # Add the entry
    idea_snippet = idea[:50].replace(',', ';') + ('...' if len(idea) > 50 else '')
    with open(PROCESSED_IDEAS_LOG, 'a', encoding='utf-8') as f:
        f.write(f"{idea_hash},{timestamp},{strategy_name},{idea_snippet}\n")
    
    cprint(f"📝 Logged processed idea: {strategy_name}", "green")

# Include all the original functions from v1
def init_deepseek_client():
    """Initialize DeepSeek client with proper error handling"""
    try:
        deepseek_key = os.getenv("DEEPSEEK_KEY")
        if not deepseek_key:
            cprint("⚠️ DEEPSEEK_KEY not found - DeepSeek models will not be available", "yellow")
            return None
            
        client = openai.OpenAI(
            api_key=deepseek_key,
            base_url=DEEPSEEK_BASE_URL
        )
        return client
    except Exception as e:
        print(f"❌ Error initializing DeepSeek client: {str(e)}")
        return None

def has_nan_results(execution_result: dict) -> bool:
    """Check if backtest results contain NaN values indicating no trades"""
    if not execution_result.get('success'):
        return False
        
    stdout = execution_result.get('stdout', '')
    
    # Look for indicators of no trades/NaN results
    nan_indicators = [
        '# Trades                                    0',
        'Win Rate [%]                              NaN',
        'Exposure Time [%]                         0.0',
        'Return [%]                                0.0'
    ]
    
    # Check if multiple NaN indicators are present
    nan_count = sum(1 for indicator in nan_indicators if indicator in stdout)
    return nan_count >= 2  # If 2+ indicators, likely no trades taken

def analyze_no_trades_issue(execution_result: dict) -> str:
    """Analyze why strategy shows signals but no trades"""
    stdout = execution_result.get('stdout', '')
    
    # Check if entry signals are being printed but no trades executed
    if 'ENTRY SIGNAL' in stdout and '# Trades                                    0' in stdout:
        return "Strategy is generating entry signals but self.buy() calls are not executing. This usually means: 1) Position sizing issues (size parameter invalid), 2) Insufficient cash/equity, 3) Logic preventing buy execution, or 4) Missing actual self.buy() call in the code. The strategy prints signals but never calls self.buy()."
    
    elif '# Trades                                    0' in stdout:
        return "Strategy executed but took 0 trades, resulting in NaN values. The entry conditions are likely too restrictive or there are logic errors preventing trade execution."
    
    return "Strategy executed but took 0 trades, resulting in NaN values. Please adjust the strategy logic to actually generate trading signals and take trades."

def chat_with_model(system_prompt, user_content, priority=ModelPriority.HIGH):
    """Chat with AI using model_priority with automatic fallback"""
    response, provider, model = model_priority_queue.get_model(
        priority=priority,
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=AI_TEMPERATURE,
        max_tokens=AI_MAX_TOKENS
    )

    if not response:
        raise ValueError("❌ All AI models failed!")

    cprint(f"✅ Used model: {provider}:{model}", "green")
    return response.content

def clean_model_output(output, content_type="text"):
    """Clean model output by removing thinking tags and extracting code from markdown"""
    cleaned_output = output
    
    # Remove thinking tags if present
    if "<think>" in output and "</think>" in output:
        clean_content = output.split("</think>")[-1].strip()
        if not clean_content:
            import re
            clean_content = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        if clean_content:
            cleaned_output = clean_content
    
    # Extract code from markdown if needed
    if content_type == "code" and "```" in cleaned_output:
        try:
            import re
            code_blocks = re.findall(r'```python\n(.*?)\n```', cleaned_output, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', cleaned_output, re.DOTALL)
            if code_blocks:
                cleaned_output = "\n\n".join(code_blocks)
        except Exception as e:
            cprint(f"❌ Error extracting code: {str(e)}", "red")
    
    return cleaned_output

def animate_progress(agent_name, stop_event):
    """Fun animation while AI is thinking"""
    spinners = ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘']
    messages = [
        "brewing coffee ☕️",
        "studying charts 📊",
        "checking signals 📡",
        "doing math 🔢",
        "reading docs 📚",
        "analyzing data 🔍",
        "making magic ✨",
        "trading secrets 🤫",
        "Moon Dev approved 🌙",
        "to the moon! 🚀"
    ]

    spinner = cycle(spinners)
    message = cycle(messages)
    
    while not stop_event.is_set():
        sys.stdout.write(f'\r{next(spinner)} {agent_name} is {next(message)}...')
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write('\r' + ' ' * 50 + '\r')
    sys.stdout.flush()

def run_with_animation(func, agent_name, *args, **kwargs):
    """Run a function with a fun loading animation"""
    stop_animation = threading.Event()
    animation_thread = threading.Thread(target=animate_progress, args=(agent_name, stop_animation))
    
    try:
        animation_thread.start()
        result = func(*args, **kwargs)
        return result
    finally:
        stop_animation.set()
        animation_thread.join()

# Include all the other functions from v1 (research, backtest, package, etc.)
def research_strategy(content):
    """Research AI: Analyzes and creates trading strategy"""
    cprint("\n🔍 Starting Research AI...", "cyan")
    
    output = run_with_animation(
        chat_with_model,
        "Research AI",
        RESEARCH_PROMPT,
        content,
        ModelPriority.HIGH
    )
    
    if output:
        output = clean_model_output(output, "text")
        
        # Extract strategy name
        strategy_name = "UnknownStrategy"
        if "STRATEGY_NAME:" in output:
            try:
                name_section = output.split("STRATEGY_NAME:")[1].strip()
                if "\n\n" in name_section:
                    strategy_name = name_section.split("\n\n")[0].strip()
                else:
                    strategy_name = name_section.split("\n")[0].strip()
                    
                strategy_name = re.sub(r'[^\w\s-]', '', strategy_name)
                strategy_name = re.sub(r'[\s]+', '', strategy_name)
                
                cprint(f"✅ Strategy name: {strategy_name}", "green")
            except Exception as e:
                cprint(f"⚠️ Error extracting strategy name: {str(e)}", "yellow")
        
        # Save research output
        filepath = RESEARCH_DIR / f"{strategy_name}_strategy.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"📝 Research saved to {filepath}", "green")
        return output, strategy_name
    return None, None

def create_backtest(strategy, strategy_name="UnknownStrategy"):
    """Backtest AI: Creates backtest implementation"""
    cprint("\n📊 Starting Backtest AI...", "cyan")
    
    output = run_with_animation(
        chat_with_model,
        "Backtest AI",
        BACKTEST_PROMPT,
        f"Create a backtest for this strategy:\n\n{strategy}",
        ModelPriority.HIGH
    )
    
    if output:
        output = clean_model_output(output, "code")
        
        filepath = BACKTEST_DIR / f"{strategy_name}_BT.py"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"🔥 Backtest saved to {filepath}", "green")
        return output
    return None

def package_check(backtest_code, strategy_name="UnknownStrategy"):
    """Package AI: Ensures correct indicator packages are used"""
    cprint("\n📦 Starting Package AI...", "cyan")
    
    output = run_with_animation(
        chat_with_model,
        "Package AI",
        PACKAGE_PROMPT,
        f"Check and fix indicator packages in this code:\n\n{backtest_code}",
        ModelPriority.HIGH
    )
    
    if output:
        output = clean_model_output(output, "code")
        
        filepath = PACKAGE_DIR / f"{strategy_name}_PKG.py"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"📦 Package-fixed code saved to {filepath}", "green")
        return output
    return None

def debug_backtest(backtest_code, error_message, strategy_name="UnknownStrategy", iteration=1):
    """Debug AI: Fixes technical issues in backtest code"""
    cprint(f"\n🔧 Starting Debug AI (iteration {iteration})...", "cyan")
    cprint(f"🐛 Error to fix: {error_message}", "yellow")
    
    # Create debug prompt with specific error
    debug_prompt_with_error = DEBUG_PROMPT.format(error_message=error_message)
    
    output = run_with_animation(
        chat_with_model,
        "Debug AI",
        debug_prompt_with_error,
        f"Fix this backtest code:\n\n{backtest_code}",
        ModelPriority.HIGH
    )
    
    if output:
        output = clean_model_output(output, "code")
        
        filepath = FINAL_BACKTEST_DIR / f"{strategy_name}_BTFinal_v{iteration}.py"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"🔧 Debugged code saved to {filepath}", "green")
        return output
    return None

def process_trading_idea_with_execution(idea: str) -> None:
    """
    THE NEW PROCESS WITH EXECUTION LOOP! 🚀
    Research -> Backtest -> Package -> Execute -> Debug (loop) -> Success!
    """
    print("\n🚀 Moon Dev's RBI AI v2.0 Processing New Idea!")
    print("🌟 Now with EXECUTION LOOP!")
    print(f"📝 Processing idea: {idea[:100]}...")
    
    # Phase 1: Research
    print("\n🧪 Phase 1: Research")
    # For this example, using the idea directly
    strategy, strategy_name = research_strategy(idea)
    
    if not strategy:
        raise ValueError("Research phase failed - no strategy generated")
        
    print(f"🏷️ Strategy Name: {strategy_name}")
    
    # Log the idea as processed once we have a strategy name
    log_processed_idea(idea, strategy_name)
    
    # Phase 2: Backtest
    print("\n📈 Phase 2: Backtest")
    backtest = create_backtest(strategy, strategy_name)
    
    if not backtest:
        raise ValueError("Backtest phase failed - no code generated")
    
    # Phase 3: Package Check
    print("\n📦 Phase 3: Package Check")
    package_checked = package_check(backtest, strategy_name)
    
    if not package_checked:
        raise ValueError("Package check failed - no fixed code generated")
    
    # Save the package-checked version
    package_file = PACKAGE_DIR / f"{strategy_name}_PKG.py"
    
    # Phase 4: EXECUTION LOOP! 🔄
    print("\n🔄 Phase 4: Execution Loop")
    
    debug_iteration = 0
    current_code = package_checked
    current_file = package_file
    error_history = []  # Track previous errors to detect loops
    
    while debug_iteration < MAX_DEBUG_ITERATIONS:
        # Execute the current code
        print(f"\n🚀 Execution attempt {debug_iteration + 1}/{MAX_DEBUG_ITERATIONS}")
        execution_result = execute_backtest(current_file, strategy_name)
        
        if execution_result['success']:
            # Check if results have NaN values (no trades taken)
            if has_nan_results(execution_result):
                print("\n⚠️ BACKTEST EXECUTED BUT NO TRADES TAKEN (NaN results)")
                print("🔧 Sending to Debug AI to fix strategy logic...")
                
                # Analyze the specific no-trades issue
                error_message = analyze_no_trades_issue(execution_result)
                
                debug_iteration += 1
                
                if debug_iteration < MAX_DEBUG_ITERATIONS:
                    debugged_code = debug_backtest(
                        current_code, 
                        error_message, 
                        strategy_name, 
                        debug_iteration
                    )
                    
                    if not debugged_code:
                        raise ValueError("Debug AI failed to generate fixed code")
                        
                    current_code = debugged_code
                    current_file = FINAL_BACKTEST_DIR / f"{strategy_name}_BTFinal_v{debug_iteration}.py"
                    print("🔄 Retrying with debugged code...")
                    continue
                else:
                    print(f"\n❌ Max debug iterations ({MAX_DEBUG_ITERATIONS}) reached - strategy still not taking trades")
                    print("🔄 Moving to next idea...")
                    return  # Move to next idea instead of crashing
            else:
                # SUCCESS! 🎉
                print("\n🎉 BACKTEST EXECUTED SUCCESSFULLY WITH TRADES!")
                print("📊 Strategy is ready to trade!")
                
                # Save final working version
                final_file = FINAL_BACKTEST_DIR / f"{strategy_name}_BTFinal_WORKING.py"
                with open(final_file, 'w', encoding='utf-8') as f:
                    f.write(current_code)
                
                print(f"✅ Final working backtest saved to: {final_file}")
                break
            
        else:
            # Extract error and debug
            error_message = parse_execution_error(execution_result)
            print(f"\n🐛 Execution failed with error: {error_message}")
            
            # Check for repeated errors (infinite loop detection)
            error_signature = error_message.split('\n')[-1] if '\n' in error_message else error_message
            if error_signature in error_history:
                print(f"\n🔄 DETECTED REPEATED ERROR: {error_signature}")
                print("🛑 Breaking loop to prevent infinite debugging")
                raise ValueError(f"Repeated error detected after {debug_iteration + 1} attempts: {error_signature}")
            
            error_history.append(error_signature)
            debug_iteration += 1
            
            if debug_iteration < MAX_DEBUG_ITERATIONS:
                # Debug the code
                print(f"\n🔧 Sending to Debug AI (attempt {debug_iteration})...")
                debugged_code = debug_backtest(
                    current_code, 
                    error_message, 
                    strategy_name, 
                    debug_iteration
                )
                
                if not debugged_code:
                    raise ValueError("Debug AI failed to generate fixed code")
                    
                current_code = debugged_code
                current_file = FINAL_BACKTEST_DIR / f"{strategy_name}_BTFinal_v{debug_iteration}.py"
                print("🔄 Retrying with debugged code...")
            else:
                print(f"\n❌ Max debug iterations ({MAX_DEBUG_ITERATIONS}) reached - could not fix code")
                print("🔄 Moving to next idea...")
                return  # Move to next idea instead of crashing
    
    print("\n✨ Processing complete!")

def main():
    """Main function - process ideas from file"""
    cprint(f"\n🌟 Moon Dev's RBI AI v2.0 Starting Up!", "green")
    cprint(f"📅 Today's Date: {TODAY_DATE}", "magenta")
    cprint(f"🔄 EXECUTION LOOP ENABLED!", "yellow")
    cprint(f"🐍 Using conda env: {CONDA_ENV}", "cyan")
    cprint(f"🔧 Max debug iterations: {MAX_DEBUG_ITERATIONS}", "cyan")
    
    cprint(f"\n📂 RBI v2.0 Data Directory: {DATA_DIR}", "magenta")
    cprint(f"📝 Reading ideas from: {IDEAS_FILE}", "magenta")
    
    # Use the ideas file from original RBI directory
    ideas_file = IDEAS_FILE
    
    if not ideas_file.exists():
        cprint("❌ ideas.txt not found! Creating template...", "red")
        ideas_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ideas_file, 'w', encoding='utf-8') as f:
            f.write("# Add your trading ideas here (one per line)\n")
            f.write("# Can be YouTube URLs, PDF links, or text descriptions\n")
            f.write("# Lines starting with # are ignored\n\n")
            f.write("Create a simple RSI strategy that buys when RSI < 30 and sells when RSI > 70\n")
            f.write("Momentum strategy using 20/50 SMA crossover with volume confirmation\n")
        cprint(f"📝 Created template ideas.txt at: {ideas_file}", "yellow")
        cprint("💡 Add your trading ideas and run again!", "yellow")
        return
        
    with open(ideas_file, 'r', encoding='utf-8') as f:
        ideas = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
    total_ideas = len(ideas)
    cprint(f"\n🎯 Found {total_ideas} trading ideas to process", "cyan")
    
    # Count how many ideas have already been processed
    already_processed = sum(1 for idea in ideas if is_idea_processed(idea))
    new_ideas = total_ideas - already_processed
    
    cprint(f"🔍 Status: {already_processed} already processed, {new_ideas} new ideas", "cyan")
    
    for i, idea in enumerate(ideas, 1):
        # Check if this idea has already been processed
        if is_idea_processed(idea):
            cprint(f"\n{'='*50}", "red")
            cprint(f"⏭️  SKIPPING idea {i}/{total_ideas} - ALREADY PROCESSED", "red", attrs=['reverse'])
            idea_snippet = idea[:100] + ('...' if len(idea) > 100 else '')
            cprint(f"📝 Idea: {idea_snippet}", "red")
            cprint(f"{'='*50}\n", "red")
            continue
        
        cprint(f"\n{'='*50}", "yellow")
        cprint(f"🌙 Processing idea {i}/{total_ideas}", "cyan")
        cprint(f"📝 Idea: {idea[:100]}{'...' if len(idea) > 100 else ''}", "yellow")
        cprint(f"{'='*50}\n", "yellow")
        
        process_trading_idea_with_execution(idea)
        
        cprint(f"\n{'='*50}", "green")
        cprint(f"✅ Completed idea {i}/{total_ideas}", "green")
        cprint(f"{'='*50}\n", "green")
        
        # Break between ideas
        if i < total_ideas:
            cprint("😴 Taking a break before next idea...", "yellow")
            time.sleep(5)

if __name__ == "__main__":
    main()