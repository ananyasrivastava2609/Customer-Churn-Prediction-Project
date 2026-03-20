"""
Train NLP pipeline on ticket data: preprocess, TF-IDF, classifier; save model and vectorizer.
"""
import logging
import sys
from pathlib import Path

from src.data_processing import load_tickets, ensure_ticket_priority, get_ticket_mapping
from src.nlp_model import train_nlp_pipeline, preprocess_ticket_text

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    data_path = Path("data/sample_tickets.csv")
    if not data_path.exists():
        logger.error("Ticket data not found at %s. Run data/generate_demo_data.py first.", data_path)
        sys.exit(1)

    df = load_tickets(str(data_path))
    df = ensure_ticket_priority(df)

    # Use ticket_priority as target (or a derived column)
    target_col = "ticket_priority"
    if target_col not in df.columns:
        logger.error("ticket_priority column missing after ensure_ticket_priority.")
        sys.exit(1)

    model, vectorizer, metrics, classes = train_nlp_pipeline(
        df, target_col=target_col, save_dir="models"
    )

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/nlp_classification_report.txt", "w") as f:
        import json
        f.write(json.dumps(metrics.get("classification_report", {}), indent=2))
    logger.info("NLP pipeline trained. Macro F1: %.4f. Saved to models/nlp_model.pkl", metrics.get("macro_f1", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
