import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import subprocess

import sys
import os 

from bets_utils import generate_bets


# Email setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_FROM = "jcmarkufc@gmail.com"  # Gmail sender

EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]
EMAIL_TO = ['jcmarkowicz@outlook.com'] # 'jimmymarkowicz28@gmail.com','jasonszat@gmail.com']

columns_to_email = ['fighter_red', 'fighter_blue','pred_name_open',
                    'open_red','open_blue','pred_winner_open',
                    'fstar_open','stake_open'
                    ]  

parlay_columns = [
    'choice_fighter_name_open',
    'parlay_fstar_open',
    'parlay_odds_open',
    'stake_open'
]

def email_bets(df_, date):

    df = df_.copy()

    df['math_red'] = df['math_red'].astype('category')
    df['math_blue'] = df['math_blue'].astype('category')
    df['elo_pred'] = df['elo_pred'].astype('category')

    df_bets, df_parlay = generate_bets(df, select_odds=0)
    

    msg = MIMEMultipart()
    msg["Subject"] = f"Betting Report {date}"
    msg["From"] = EMAIL_FROM
    # Join the list into a comma-separated string for the header
    msg["To"] = ", ".join(EMAIL_TO)


    # ---- Body with HTML link ----
    html_body = """
    <p>See attached newly announced fights.</p>
    <p>Click here for all upcoming picks: <a href="https://sportsbetting-cn2kwvhykyrxdw2gxmuifl.streamlit.app/upcoming_picks">Dashboard</a></p>
    """
    msg.attach(MIMEText(html_body, "html"))

    # ---- Straight Bets CSV ----
    bets_csv = df_bets[columns_to_email].to_csv(index=False)

    part = MIMEBase("application", "octet-stream")
    part.set_payload(bets_csv.encode())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        "attachment; filename=straight_bets.csv",
    )
    msg.attach(part)



    parlay_csv = df_parlay[parlay_columns].to_csv(index=False)

    part2 = MIMEBase("application", "octet-stream")
    part2.set_payload(parlay_csv.encode())
    encoders.encode_base64(part2)
    part2.add_header(
        "Content-Disposition",
        "attachment; filename=parlay_bets.csv",
    )
    msg.attach(part2)

    # ---- Send email ----
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)

        # Pass the list of recipients to send_message
        server.send_message(msg, from_addr=EMAIL_FROM, to_addrs=EMAIL_TO)
