

import os
import sys 
from pathlib import Path 


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
BASE_DIR = Path(__file__).resolve().parents[1]

ml_history_fp = BASE_DIR / 'Data' / 'betting_results' / 'moneyline_results.csv'
parlay_history_fp = BASE_DIR / 'Data' / 'betting_results' / 'parlay_results.csv'