"""Pipeline runner — orchestrates fetch, normalise, and feature steps.

Usage:
    python -m src.pipeline --step fetch
    python -m src.pipeline --step fetch-results
    python -m src.pipeline --step fetch-racecards
    python -m src.pipeline --step fetch-betfair
    python -m src.pipeline --step fetch-odds
    python -m src.pipeline --step normalise
    python -m src.pipeline --step features
    python -m src.pipeline --step parse-comments
    python -m src.pipeline --step train
    python -m src.pipeline --step predict
    python -m src.pipeline --step all
"""

import argparse
import logging
import sys

from src.ingest.fetch_racecards import fetch_racecards
from src.ingest.fetch_results import fetch_all_results
from src.transform.normalise import normalise

logger = logging.getLogger(__name__)

STEPS = {
    "fetch", "fetch-results", "fetch-racecards", "fetch-betfair", "fetch-odds",
    "normalise", "features", "parse-comments",
    "train", "predict", "all",
}

FETCH_STEPS = {"fetch", "fetch-results", "fetch-racecards", "fetch-betfair", "fetch-odds"}


def run_fetch_results() -> None:
    logger.info("=== Fetching results ===")
    results = fetch_all_results()
    logger.info("Fetched %d result files", len(results))


def run_fetch_racecards() -> None:
    logger.info("=== Fetching racecards ===")
    racecards = fetch_racecards()
    logger.info("Fetched %d racecard files", len(racecards))


def run_normalise() -> None:
    logger.info("=== Normalising data ===")
    normalise()


def run_features() -> None:
    from src.features.build_features import build_features

    logger.info("=== Building features ===")
    build_features()


def run_train() -> None:
    from src.model.train import train

    logger.info("=== Training model ===")
    train()


def run_predict() -> None:
    from src.model.predict import predict

    logger.info("=== Generating predictions ===")
    predict()


def run_parse_comments() -> None:
    from src.features.parse_comments import parse_comments

    logger.info("=== Parsing comments ===")
    parse_comments()


def run_fetch_betfair() -> None:
    from src.ingest.fetch_betfair_historical import fetch_betfair_historical

    logger.info("=== Fetching Betfair historical BSP ===")
    fetch_betfair_historical()


def run_fetch_odds() -> None:
    from src.ingest.fetch_betfair_odds import fetch_cheltenham_odds

    logger.info("=== Fetching Betfair live odds ===")
    fetch_cheltenham_odds()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cheltenham data pipeline")
    parser.add_argument(
        "--step",
        choices=sorted(STEPS),
        required=True,
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.step in ("fetch", "fetch-results", "all"):
        run_fetch_results()

    if args.step in ("fetch", "fetch-racecards", "all"):
        run_fetch_racecards()

    if args.step in ("normalise", "all"):
        run_normalise()

    if args.step in ("features", "all"):
        run_features()

    if args.step in ("train", "all"):
        run_train()

    if args.step in ("predict", "all"):
        run_predict()

    if args.step == "parse-comments":
        run_parse_comments()

    if args.step in ("fetch-betfair", "all"):
        run_fetch_betfair()

    if args.step == "fetch-odds":
        run_fetch_odds()

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
