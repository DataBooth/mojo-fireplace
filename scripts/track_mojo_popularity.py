# /// script
# dependencies = [
#   "requests",
#   "pytrends",
#   "pandas",
#   "plotly",
# ]
# requires-python = ">=3.9"
# ///

import requests
import json
import time
import csv
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pytrends.request import TrendReq

# ---------- Configuration ----------
GITHUB_REPOS = [
    "mojicians/awesome-mojo",
    "modularml/mojo",
    "modular/modular",
]

SO_TAGS = ["mojolang", "mojo"]

TRENDS_KEYWORD = "Mojo programming language"

CSV_FILE = "mojo_popularity_tracking.csv"
HTML_PLOT_FILE = "mojo_popularity_trends.html"

# ---------- Helper Functions ----------
def get_github_stats(owner_repo):
    url = f"https://api.github.com/repos/{owner_repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return {
            "stars": data["stargazers_count"],
            "forks": data["forks_count"],
            "watchers": data["subscribers_count"]
        }
    else:
        print(f"Error fetching {owner_repo}: {response.status_code}")
        return {"stars": 0, "forks": 0, "watchers": 0}

def get_so_question_count(tag):
    url = f"https://api.stackexchange.com/2.3/tags/{tag}/info"
    params = {"site": "stackoverflow"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data["items"][0]["count"] if data["items"] else 0
    else:
        print(f"Error fetching SO tag {tag}: {response.status_code}")
        return 0

# ---------- Main Tracking Function ----------
def track_mojo_popularity():
    date = datetime.now().strftime("%Y-%m-%d")
    
    # GitHub
    github_data = {}
    total_stars = 0
    for repo in GITHUB_REPOS:
        stats = get_github_stats(repo)
        github_data[repo] = stats
        total_stars += stats["stars"]
        time.sleep(1.1)
    
    # Stack Overflow
    so_counts = {tag: get_so_question_count(tag) for tag in SO_TAGS}
    total_so_questions = sum(so_counts.values())
    time.sleep(1)
    
    # Google Trends
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([TRENDS_KEYWORD], timeframe='today 3-m')
        recent_df = pytrends.interest_over_time()
        current_trends_score = recent_df[TRENDS_KEYWORD].mean() if not recent_df.empty else 0
    except Exception as e:
        print("Google Trends failed:", e)
        current_trends_score = 0
    
    # Row
    row = {
        "date": date,
        "total_github_stars": total_stars,
        "so_questions": total_so_questions,
        "google_trends_score": round(current_trends_score, 2),
        "details_github": json.dumps(github_data),
        "details_so": json.dumps(so_counts)
    }
    
    # Append
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"[{date}] Tracked → GitHub stars: {total_stars} | SO questions: {total_so_questions} | Trends: {current_trends_score:.2f}")

# ---------- Plot ----------
def plot_trends():
    if not os.path.isfile(CSV_FILE):
        print("No data yet.")
        return
    
    df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    df = df.sort_values("date")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "GitHub Stars (Aggregate)",
            "Stack Overflow Questions (cumulative)",
            "Google Trends Interest (0–100, 3-month avg)"
        ),
        vertical_spacing=0.08
    )
    
    fig.add_trace(go.Scatter(x=df["date"], y=df["total_github_stars"], mode="lines+markers", line=dict(color="#636EFA")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["so_questions"], mode="lines+markers", line=dict(color="#EF553B")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["google_trends_score"], mode="lines+markers", line=dict(color="#00CC96")), row=3, col=1)
    
    fig.update_layout(height=900, title_text="Mojo Programming Language Popularity Over Time", showlegend=False, template="plotly_dark")
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Stars", row=1, col=1)
    fig.update_yaxes(title_text="Questions", row=2, col=1)
    fig.update_yaxes(title_text="Interest Score", row=3, col=1)
    
    fig.write_html(HTML_PLOT_FILE, include_plotlyjs="cdn")
    print(f"Plot saved to {HTML_PLOT_FILE}")

# ---------- Run ----------
if __name__ == "__main__":
    track_mojo_popularity()
    plot_trends()