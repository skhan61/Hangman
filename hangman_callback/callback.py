"""Lightning callback for evaluating Hangman agents during training."""

from __future__ import annotations

import logging
from functools import partial
from typing import List, Sequence

from lightning.pytorch.callbacks import Callback

from api.offline_api import HangmanOfflineAPI
from api.guess_strategies import neural_guess_strategy

logger = logging.getLogger(__name__)


class CustomHangmanEvalCallback(Callback):
    """Evaluate a model on Hangman games and optionally stop training early."""

    def __init__(
        self,
        *,
        val_word_list: Sequence[str] | None = None,
        val_words_path: str | None = None,
        dictionary_path: str | None = None,
        verbose: bool = False,
        max_words: int | None = None,
        patience: int = 0,
        min_delta: float = 0.0,
        mode: str = "max",
        parallel: bool = True,
        frequency: int = 1,
    ) -> None:
        super().__init__()

        if val_words_path is not None:
            with open(val_words_path, "r", encoding="utf-8") as handle:
                words = [line.strip().lower() for line in handle if line.strip()]
            self.val_word_list = words[:max_words] if max_words else words
        elif val_word_list is not None:
            self.val_word_list = (
                list(val_word_list[:max_words])
                if max_words is not None
                else list(val_word_list)
            )
        else:
            raise ValueError("Either val_words_path or val_word_list must be provided.")

        if not self.val_word_list:
            raise ValueError("No testing words provided for evaluation callback.")

        self.dictionary_path = dictionary_path or val_words_path
        if self.dictionary_path is None:
            raise ValueError("dictionary_path must be provided or inferable.")
        self.verbose = verbose
        self.parallel = parallel and not verbose
        self.frequency = max(1, frequency)

        if mode not in {"max", "min"}:
            raise ValueError("mode must be either 'max' or 'min'")
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best_metric_value = float("-inf") if mode == "max" else float("inf")
        self.bad_epochs = 0
        self.latest_results = None

        # Track best win rate and epoch for benchmark reporting
        self.best_win_rate = 0.0
        self.best_epoch = 0

        logger.info("%d words loaded for testing.", len(self.val_word_list))

    # ------------------------------------------------------------------
    # Lightning callback hooks
    # ------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):  # noqa: D401
        """Run evaluation before training starts to see untrained model performance."""
        logger.info("Running hangman evaluation BEFORE training (untrained model)...")

        # Save current training mode
        pl_module.eval()
        # Run evaluation (sets model to eval mode)
        summary = self._run_evaluation(pl_module.model)
        self.latest_results = summary

        win_rate = summary["win_rate"]
        avg_tries = summary["average_tries_remaining"]

        logger.info(
            "BEFORE - Win rate: %.2f%%, Avg tries remaining: %.2f",
            win_rate * 100,
            avg_tries,
        )

        # Restore training mode
        # if was_training:
        pl_module.train()

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: D401
        """Run hangman evaluation at the end of each training epoch."""
        current_epoch = trainer.current_epoch

        # Check if we should run evaluation this epoch based on frequency
        if current_epoch % self.frequency != 0:
            logger.debug(
                "Skipping hangman evaluation at epoch %d (frequency=%d)",
                current_epoch,
                self.frequency,
            )
            return

        logger.info("Running hangman evaluation at epoch %d", current_epoch)

        # Set model to eval mode for evaluation
        pl_module.eval()

        summary = self._run_evaluation(pl_module.model)
        self.latest_results = summary

        win_rate = summary["win_rate"]
        avg_tries = summary["average_tries_remaining"]

        # Log metrics for checkpoint monitoring - must use on_epoch=True for ModelCheckpoint
        pl_module.log(
            "hangman_win_rate",
            win_rate,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        pl_module.log(
            "hangman_avg_tries_remaining",
            avg_tries,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,
            sync_dist=True,
        )

        logger.info(
            "Epoch %d - Win rate: %.2f%%, Avg tries remaining: %.2f",
            current_epoch,
            win_rate * 100,
            avg_tries,
        )

        # Check for improvement and early stopping
        if self._is_improvement(win_rate):
            self._update_best(win_rate, current_epoch)
        elif self.patience > 0:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                logger.info(
                    "Early stopping triggered by Hangman win rate (patience=%d).",
                    self.patience,
                )
                trainer.should_stop = True

        # Restore training mode
        pl_module.train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_evaluation(self, model, *, parallel_override: bool | None = None):
        strategy = partial(neural_guess_strategy, model=model)
        api = HangmanOfflineAPI(
            dictionary_file_location=self.dictionary_path,
            strategy=strategy,
        )

        parallel = self.parallel if parallel_override is None else parallel_override
        if parallel:
            logger.warning(
                "Parallel mode is disabled for neural strategies; falling back to serial evaluation."
            )
        summary = api.simulate_games_for_word_list(
            self.val_word_list,
            parallel=False,
        )

        overall = summary.get("overall", {})
        win_rate = overall.get("win_rate", 0.0)
        avg_tries = overall.get("average_tries_remaining", 0.0)
        games = summary.get("games", [])

        if self.verbose and games:
            self._log_verbose_results(
                games,
                {
                    "wins": overall.get("wins", 0),
                    "total_games": overall.get("total_games", len(games)),
                },
                win_rate,
                avg_tries,
            )

        return {
            "win_rate": win_rate,
            "average_tries_remaining": avg_tries,
            "games": games,
        }

    def _log_verbose_results(
        self,
        word_results: List[dict],
        overall: dict,
        win_rate: float,
        avg_tries: float,
    ) -> None:
        logger.debug("=" * 60)
        logger.debug("RUNNING TESTING GAMES")
        logger.debug("=" * 60)
        for result in word_results:
            word = result.get("word")
            win = result.get("win")
            tries_remaining = result.get("tries_remaining")
            logger.debug("\nWord: '%s'", word)
            logger.debug("Result: %s", "WIN" if win else "LOSS")
            logger.debug("Tries remaining: %s/6", tries_remaining)
            logger.debug("Game progress:")
            for guess_letter, masked_word, correct in result.get("progress", []):
                status = "✓" if correct else "✗"
                logger.debug("  Guess '%s' %s -> %s", guess_letter, status, masked_word)

        logger.debug("\n" + "=" * 60)
        logger.debug(
            "Win Rate: %s (%s/%s)",
            f"{win_rate:.2%}",
            overall.get("wins", 0),
            overall.get("total_games", len(word_results)),
        )
        logger.debug("Average Tries Remaining: %.2f", avg_tries)
        logger.debug("=" * 60)

    def _is_improvement(self, metric_value: float) -> bool:
        if self.mode == "max":
            return metric_value > self.best_metric_value + self.min_delta
        return metric_value < self.best_metric_value - self.min_delta

    def _update_best(self, metric_value: float, epoch: int = 0) -> None:
        self.best_metric_value = metric_value
        self.best_win_rate = metric_value  # Track for benchmark
        self.best_epoch = epoch  # Track for benchmark
        self.bad_epochs = 0
        logger.info("New best hangman win rate: %.2f%%", metric_value * 100.0)
