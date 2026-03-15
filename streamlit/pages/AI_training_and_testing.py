import streamlit as st

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from BettingStrategy.ModelStrategy.LogisticRegression.get_train_test import TrainTestBuilder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from utils import show_image

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

st.markdown('<h2>Motivation</h2>', unsafe_allow_html=True)

st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            In order to seperate winners from loosers in future UFC events, data needs to be properly managed to ensure accurate and robust results. 
            The first order of business is to establish restraints which define the ways in which data goes into some 'AI', and then how the resulting output can be interpreted.
            The process of predicting future outcomes is inherently a time series problem in which some data from previous events has been recorded, and within that data
            is expected to be signals or relevant information which an AI can look at to form its decisions for future events. Therefore, all data used to predict an outcome must be set prior to the match, otherwise
            the AI will be cheating and future events for which there is no data cannot be modeled.  
            </p> 
""", unsafe_allow_html=True)

st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            Once all the relevant predictors of fight outcomes have been created, the AI, or rather the function f(X)=y, can be built. 
            The creation of all AI models follows simple patterns which seperate the entire process into segments with clear motivations. AI must be trained and tested in order to establish the reliability 
            and limitations imposed by the underlying problem. In this instance, train test data followed an 85/15 percent split, where the AI was optimized on 85% of the data and the remaining 15% was used 
            as an out of sample test. The important detail to note is that once the AI is optimized, it cannot be changed and the test data will never be used to optimize the model. The results shown in the betting simulation
            page are all derived from test sample data ensuring the integrity and validity of results. Essentially, test results boil down to an estimation of unleashing AI on data it has never seen; AI draws upon
            its pre existing knowledge to infer outcomes and arrive at optimized predictions. 
            </p> 
""", unsafe_allow_html=True)

st.markdown('<h2>Results Analysis</h2>', unsafe_allow_html=True)

st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            There are many choices for the type of AI model to be used, and I elected for a simple Logistic Regression with lasso regularization. The details of the model will not be explained here, but the 
            curious reader might want to see <a href="https://en.wikipedia.org/wiki/Logistic_regression" target="_blank">Wikipedia </a>. The important thing to note is the AI will make a binary prediction; either
            1 for fighter A wins, or 0 for fighter A loses. Its trivial to derive the results for fighter B, and because the result is a floating point between 0 and 1, that output can be interpreted as a probability.
            </p> 
""", unsafe_allow_html=True)

st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            The results displayed below are standard for AI evaluation, and the details of each are obmitted due the already vast information online explaining each. However, one notable aspect of the model training 
            was probability calibration, which in short has the objective of making probability scores reliable. If there are 10 predictions that report 80% accuracy then a calibrated predictor will correctly 
            predict the outcome 80% of the time (8 out of 10 times). The better the calibration, the better the betting algorithmns performed. See the first plot to the left for evalution of calibration.
            <a href="https://en.wikipedia.org/wiki/Calibration_(statistics)" target="_blank">Calibration_wiki </a>. <a href="https://en.wikipedia.org/wiki/Confusion_matrix" target="_blank">confusion_matrix_wiki </a>.
            <a href="https://en.wikipedia.org/wiki/F-score" target="_blank">F1_score_wiki</a>
            </p> 
""", unsafe_allow_html=True)


st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            As displayed below, there are several subsets of train test results marked by the odds type(open, close1, close2). I elected to train a seperate model per odds type due to the high influence
            each type's derived probability had on the outcome of the model. Interesting to note, the open model had the highest test accuracy despite not having the signal of market movement which is 
            encoded in the close1 and 2 odds. This can be for a number of reasons, but my best guess is that sports bettors are not always optimal and the line movement can be more noise than signal. 
            Also, this is GREAT for the opening model because it validates its edge; the model can predict the optimal winner before the market does and thus place bets with the highest amount of equity before line movement wipes it out. 
            </p> 
""", unsafe_allow_html=True)


train_open_fp = BASE_DIR / "Data" / "plot_pngs" / "train_open_model_metrics.png" 
test_open_fp = BASE_DIR / "Data" / "plot_pngs" / "test_open_model_metrics.png" 

show_image(train_open_fp, title='Train Open Results')
show_image(test_open_fp, title='Test Open Results')


train_close1_fp = BASE_DIR / "Data" / "plot_pngs" / "train_close1_model_metrics.png" 
test_close1_fp = BASE_DIR / "Data" / "plot_pngs" / "test_close1_model_metrics.png" 

show_image(train_close1_fp, title='Train Close1 Results')
show_image(test_close1_fp, title='Test Close1 Results')

train_close2_fp = BASE_DIR / "Data" / "plot_pngs" / "train_close2_model_metrics.png" 
test_close2_fp = BASE_DIR / "Data" / "plot_pngs" / "test_close2_model_metrics.png" 

show_image(train_close2_fp, title='Train Close2 Results')
show_image(test_close2_fp, title='Test Close2 Results')