"""Display saved Kelly searches and explain the implemented betting controls."""
import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import display_paginated_df, show_image
from ufc_betting.config import settings


def parameter_table(path):
    """Expand nested saved controls into readable rows, including opening controls."""
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"parameter", "Best parameters"} <= set(frame.columns):
        raise ValueError("Expected parameter and Best parameters columns")
    rows = []

    def append_value(name, value):
        if isinstance(value, dict):
            for key, item in value.items():
                append_value(f"{name}.{key}", item)
        else:
            if value is None or value == "":
                value = "Not set"
            elif isinstance(value, (int, float)):
                value = f"{value:.4f}"
            rows.append({"Parameter": name, "Value": str(value)})

    for name, value in frame[["parameter", "Best parameters"]].itertuples(index=False, name=None):
        if isinstance(value, str) and value:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        append_value(name, value)
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


st.title("Betting Strategy")
st.write("Saved Kelly search parameters and diagnostics for opening, close1 and close2 odds.")
results_dir = Path(st.text_input(
    "Saved search results folder",
    value=str(settings.data_dir / "kelly_params" / "search_results"),
)).expanduser()

for tab, market in zip(st.tabs(["Opening", "Close1", "Close2"]), ("open", "close1", "close2")):
    with tab:
        table_path = results_dir / f"{market}_best_parameters.csv"
        if table_path.is_file():
            try:
                table = parameter_table(table_path)
                if table.empty:
                    st.info("The saved parameter table is empty.")
                else:
                    display_paginated_df(table, title=f"{market}: best parameters",
                                         key_prefix=f"strategy_{market}")
            except (OSError, ValueError, pd.errors.ParserError) as error:
                st.error(f"Could not read {table_path.name}: {error}")
        else:
            st.info(f"No saved parameter table: {table_path.name}")
        for suffix, title in (
            ("optimization_history", "Optimization history"),
            ("parameter_importance", "Parameter importance"),
            ("fitted_decay", "Fitted decay"),
        ):
            image_path = results_dir / f"{market}_{suffix}.png"
            if image_path.is_file():
                try:
                    show_image(image_path, title=f"{market}: {title}")
                except (OSError, ValueError) as error:
                    st.error(f"Could not display {image_path.name}: {error}")
            elif suffix != "fitted_decay":
                st.info(f"No saved plot: {image_path.name}")

st.header("Search objective")
st.write("Optuna uses its Tree-structured Parzen Estimator (TPE) sampler to propose betting parameters and maximize the score across fixed sampled validation paths. "
         "The event_log_growth objective rewards average log bankroll growth per event while penalizing severe drawdowns and sustained time below prior bankroll peaks.")
st.latex(r"\theta^*=\arg\max_{\theta}\ J(\theta),\qquad J(\theta)=\frac{1}{K}\sum_{k=1}^{K}g_k(\theta)-\lambda_D\frac{1}{m}\sum_{j=1}^{m}D_{(j)}(\theta)-\lambda_U\frac{1}{K}\sum_{k=1}^{K}U_k(\theta)")
st.latex(r"g_k=\frac{1}{T_k}\sum_{t=1}^{T_k}\left[\ln\!\left(\max\!\left(\frac{B_{k,t}}{B_{k,0}},\varepsilon\right)\right)-\ln\!\left(\max\!\left(\frac{B_{k,t-1}}{B_{k,0}},\varepsilon\right)\right)\right]")
st.write("K is the number of paths, Tₖ counts all scheduled events, and Bₖ,ₜ is the bankroll after event t. "
         "The wealth floor ε keeps the log score finite without changing the simulated bankroll; no-bet and post-ruin events remain in the average.")
st.latex(r"H_{k,t}=\max_{0\leq s\leq t}B_{k,s},\qquad d_{k,t}=\frac{H_{k,t}-B_{k,t}}{H_{k,t}},\qquad D_k=\max_t d_{k,t},\qquad U_k=\sqrt{\frac{1}{T_k}\sum_{t=1}^{T_k}d_{k,t}^{\,2}}")
st.latex(r"D_{(1)}\geq\cdots\geq D_{(K)},\qquad m=\max(1,\lceil\alpha K\rceil)")
st.write("λ_D is drawdown_penalty, α is drawdown_tail_fraction, and λ_U is ulcer_penalty: the penalties use the worst m path maximum drawdowns and the mean path Ulcer Index. "
         "Drawdowns use actual balances as fractions, with neither the log wealth floor nor clipping applied.")

st.header("Betting parameter equations")
st.write("Here p is the model win probability, q is the fair market probability, o is decimal odds, "
         "and f is a fraction of bankroll; [x]₊ means max(x, 0).")
st.subheader("Starting Kelly fraction")
st.latex(r"f_K(p,o)=\left[\frac{po-1}{o-1}\right]_+,\qquad p_T=p_1p_2,\quad o_T=o_1o_2")
st.write("Moneylines use fair decimal odds in Kelly sizing, while the parlay uses the product of offered odds and leg probabilities. "
         "The two parlay legs are selected by their unadjusted offered-price EV.")

st.subheader("Drawdown sizing definitions")
st.latex(r"r_w=\ln(1+f(o-1)),\quad r_l=\ln(1-f),\quad \mu=pr_w+(1-p)r_l")
st.latex(r"\sigma(f;p,o)=\sqrt{p(r_w-\mu)^2+(1-p)(r_l-\mu)^2}")
st.latex(r"A(n)=\sqrt{2\ln n}-\frac{\ln\ln n+\ln(4\pi)}{2\sqrt{2\ln n}}")
st.write("The implementation approximates maximum log drawdown by σA(n). "
         "The drawdown parameters constrain this approximation, rather than guaranteeing a realized drawdown limit.")

PARAMETERS = [
    ("mdd — moneyline drawdown budget",
     r"f_{ML}=\max_{0\leq f\leq f_K(p,o_{fair})}\{f:\ \sigma(f;p,o_{fair})A(N)\leq-\ln(1-\mathrm{mdd})\}",
     "Sets the drawdown budget used to reduce the full-Kelly moneyline stake. A larger budget permits a larger stake, up to full Kelly."),
    ("N — moneyline horizon",
     r"\widehat D_{ML}(f)=\sigma(f;p,o_{fair})A(N),\qquad N>1",
     "Controls the assumed number of bets in the moneyline drawdown approximation. Increasing N tightens sizing for the same drawdown budget."),
    ("mdd_parlay — parlay drawdown budget",
     r"f_T=\max_{0\leq f\leq f_K(p_T,o_T)}\{f:\ \sigma(f;p_T,o_T)A(N_{parlay})\leq-\ln(1-\mathrm{mdd}_{parlay})\}",
     "Sets the drawdown budget for the whole parlay ticket. It is applied once to the ticket, not separately to each leg."),
    ("N_parlay — parlay horizon",
     r"\widehat D_T(f)=\sigma(f;p_T,o_T)A(N_{parlay}),\qquad N_{parlay}>1",
     "Controls the assumed number of parlay bets in the drawdown approximation. It can differ from the moneyline horizon."),
    ("z — probability uncertainty adjustment",
     r"e_i=[p_{risk,i}-q_i]_+,\quad c_i=[p_{risk,i}-z\,SE_i-q_i]_+,\quad m_i=\operatorname{clip}\!\left(\frac{c_i}{e_i},0,1\right)",
     "Subtracts z standard errors before calculating the conservative edge and scales each stake by conservative edge divided by original edge. The multiplier is zero when its denominator is zero; disabling z uses a multiplier of one."),
    ("z — parlay ticket multiplier",
     r"m_T=\operatorname{clip}\!\left(\frac{c_1+c_2}{e_1+e_2},0,1\right)",
     "The parlay uses the ratio of summed leg edges, with zero used for a zero denominator. One leg losing its conservative edge does not automatically eliminate the ticket."),
    ("moneyline_min_american — minimum offered moneyline",
     r"a_i\geq a_{min},\qquad a_{min}=\mathrm{moneyline\_min\_american}",
     "Excludes moneyline bets whose offered American odds fall below this bound. For example, −225 excludes −300 favorites."),
    ("moneyline_max_american — maximum offered moneyline",
     r"a_i\leq a_{max},\qquad a_{max}=\mathrm{moneyline\_max\_american}",
     "Excludes moneyline bets whose offered American odds exceed this bound. An unset bound is unrestricted."),
    ("k_down — moneyline decay after an odds decrease",
     r"\Delta_i=\ln(o_{current,i}/o_{open,i}),\quad w_i=e^{-k_{down}[-\Delta_i]_+-k_{up}[\Delta_i]_+}",
     "Controls how quickly opening-policy weight decreases when decimal odds fall. Larger k_down shifts the stake toward the closing policy faster."),
    ("k_up — moneyline decay after an odds increase",
     r"\Delta_i>0\ \Longrightarrow\ w_i=e^{-k_{up}\Delta_i}",
     "Controls the opening-policy weight when decimal odds rise. Larger k_up shifts the stake toward the closing policy faster."),
    ("k_parlay_down — parlay downward-movement decay",
     r"D=\sum_{i\in T}[-\Delta_i]_+,\quad U=\sum_{i\in T}[\Delta_i]_+,\quad w_T=e^{-k_{parlay,down}D-k_{parlay,up}U}",
     "Controls decay from the sum of downward movements across the parlay legs. Upward and downward movements do not cancel."),
    ("k_parlay_up — parlay upward-movement decay",
     r"D=0\ \Longrightarrow\ w_T=e^{-k_{parlay,up}U}",
     "Controls decay from the sum of upward movements across the parlay legs. Larger values reduce opening-policy weight faster."),
    ("opening_params — saved opening-policy controls",
     r"f_{blend}=w\,f_{open\ policy}+(1-w)\,f_{closing\ policy}",
     "Both policies are evaluated using current predictions and prices before blending their stakes. The closing policy's z adjustment is applied after blending."),
]
for title, equation, description in PARAMETERS:
    st.subheader(title)
    st.latex(equation)
    st.write(description)

st.subheader("Final joint event exposure cap")
st.latex(r"\widetilde f_j=m_jf_{blend,j},\quad C=\frac{1}{\max(1,\sum_j\widetilde f_j)},\quad \mathrm{stake}_j=B\,C\widetilde f_j")
st.write("The shared cap limits combined moneyline and parlay exposure to the available bankroll B. "
         "Each parlay ticket contributes once to the total exposure.")
