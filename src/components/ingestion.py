# src/components/ingestion.py
import os
import json
import yfinance as yf
import pandas as pd
import feedparser
from datetime import date, timedelta
import subprocess
import sys
import argparse
import re
from pathlib import Path

# =========================================================
# Helper utilities
# =========================================================
def safe_dvc_command(cmd_list, use_dvc):
    if not use_dvc:
        print(f"⚠️ Skipping DVC command: {' '.join(cmd_list)}")
        return
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ DVC command failed: {e}")

# =========================================================
# Config defaults (these may be overridden by envs or args)
# =========================================================
TICKER_DEFAULT = "^NSEI"
QUERY_DEFAULT = "Nifty"

# =========================================================
# Fetch stock
# =========================================================
def fetch_stock_data(stock_csv_path: Path, ticker: str, start_date: str, end_date: str) -> Path:
    print(f"📥 Downloading stock data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        print("⚠️ No stock data returned. CSV will still be created as empty.")
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    stock_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stock_csv_path)
    print(f"✅ Stock data saved to {stock_csv_path}")
    return stock_csv_path

# =========================================================
# Fetch news
# =========================================================
def fetch_news_data(news_csv_path: Path, query: str) -> Path:
    print(f"📰 Fetching news articles from Economic Times RSS for '{query}'")
    RSS_URL = "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/1977021501.cms"
    feed = feedparser.parse(RSS_URL)

    articles = []
    for entry in feed.entries[:50]:
        articles.append({
            "title": getattr(entry, "title", ""),
            "link": getattr(entry, "link", ""),
            "published": getattr(entry, "published", ""),
            "summary": getattr(entry, "summary", "")
        })

    df = pd.DataFrame(articles)
    news_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(news_csv_path, index=False)
    print(f"✅ Saved {len(df)} news articles to {news_csv_path}")
    return news_csv_path

# =========================================================
# DVC tracking
# =========================================================
def track_with_dvc(file_path: Path, use_dvc: bool):
    try:
        safe_dvc_command([sys.executable, "-m", "dvc", "add", str(file_path)], use_dvc)
        dvc_file = f"{file_path}.dvc"
        if use_dvc:
            subprocess.run(["git", "add", dvc_file], check=True)
            subprocess.run(["git", "commit", "-m", f"Track {os.path.basename(file_path)} with DVC"], check=False)
        print(f"✅ DVC tracking completed for {file_path}")
    except Exception as e:
        print(f"⚠️ Skipped or failed DVC tracking for {file_path}: {e}")

# =========================================================
# Utility: capture git commit (optional)
# =========================================================
def get_git_commit_hash(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None

# =========================================================
# Utility: read md5 from .dvc file (no external deps)
# =========================================================
def read_dvc_md5_from_file(dvc_path: Path) -> str | None:
    """
    Parse a .dvc file and extract the md5 value for the first output.
    Returns the md5 string or None if not found.
    """
    if not dvc_path.exists():
        return None
    try:
        text = dvc_path.read_text()
        # Match lines like: md5: <hex>
        m = re.search(r"md5:\s*['\"]?([0-9a-fA-F]+)['\"]?", text)
        if m:
            return m.group(1)
        # older/newer formats might use 'etag' or 'hash', try a fallback for 'hash:'
        m2 = re.search(r"hash:\s*['\"]?([0-9a-fA-F:]+)['\"]?", text)
        if m2:
            return m2.group(1)
    except Exception:
        return None
    return None

def collect_dvc_hashes(files: list[Path]) -> dict:
    """
    For each data file path, look for a .dvc file next to it and return dict { filename: md5 }.
    """
    hashes = {}
    for p in files:
        if p is None:
            continue
        dvc_file = Path(str(p) + ".dvc")
        if dvc_file.exists():
            md5 = read_dvc_md5_from_file(dvc_file)
            hashes[p.name] = md5
        else:
            hashes[p.name] = None
    return hashes

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-news-only", action="store_true", help="Only download news CSV")
    parser.add_argument("--download-stock-only", action="store_true", help="Only download stock CSV")
    args = parser.parse_args()

    CI_MODE = os.getenv("CI", "false").lower() == "true"
    DOCKER_MODE = os.getenv("DOCKER", "false").lower() == "true"
    USE_DVC = os.getenv("USE_DVC", "true").lower() == "true" and not CI_MODE

    ticker = os.getenv("TICKER", TICKER_DEFAULT)
    query = os.getenv("QUERY", QUERY_DEFAULT)

    today = date.today()
    end_date = (today - timedelta(days=1)).isoformat()
    start_date = (today - timedelta(days=2)).isoformat()
    print(f"📅 Date range automatically set: {start_date} → {end_date}")

    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    stock_csv_path = raw_dir / "stock.csv"
    news_csv_path = raw_dir / "news_NIFTY.csv"
    summary_path = raw_dir / "ingestion_summary.json"

    # Run ingestion
    stock_path, news_path = None, None
    try:
        if args.download_stock_only:
            stock_path = fetch_stock_data(stock_csv_path, ticker, start_date, end_date)
        elif args.download_news_only:
            news_path = fetch_news_data(news_csv_path, query)
        else:
            stock_path = fetch_stock_data(stock_csv_path, ticker, start_date, end_date)
            news_path = fetch_news_data(news_csv_path, query)

        # DVC tracking (best-effort)
        if stock_path:
            track_with_dvc(stock_path, USE_DVC)
        if news_path:
            track_with_dvc(news_path, USE_DVC)

        # Prepare summary with git commit for traceability
        df_stock = pd.read_csv(stock_path) if (stock_path and stock_path.exists()) else pd.DataFrame()
        df_news = pd.read_csv(news_path) if (news_path and news_path.exists()) else pd.DataFrame()

        summary = {
            "ticker": ticker,
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
            "num_stock_records": len(df_stock),
            "num_news_articles": len(df_news),
            "use_dvc": bool(USE_DVC),
            "git_commit": get_git_commit_hash(repo_root)
        }

        # collect md5 hashes from .dvc files (if present)
        dvc_hashes = collect_dvc_hashes([stock_path, news_path])
        summary["dvc_hashes"] = dvc_hashes

        # write summary
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"🗓️ Summary saved to {summary_path}")
        print(f"🗃️ Summary: {json.dumps(summary, indent=2)}")

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
