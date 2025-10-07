from hangman_game._game_engine import play_a_game_with_a_word, simulate_games_for_word_list
from api.offline_api import guess


class CustomHangmanEvalCallback:
    def __init__(self, val_word_list=None):
        self.val_word_list = val_word_list
        self.best_metric_value = float('-inf')
        self.model = None

    def attach_model(self, model):
        self.model = model

    def on_rollout_end(self):
        results = self.evaluate_custom_metrics()
        print("Evaluation results:", results)

    def evaluate_custom_metrics(self):
        if self.model is None or self.val_word_list is None:
            return {}
        results = simulate_games_for_word_list(word_list=self.val_word_list, 
                                               guess_function=guess,
                                               play_function=play_a_game_with_a_word,
                                               model=self.model, 
                                               solver=None, 
                                               transform=None, 
                                               process_word_fn=None)
        
        win_rate = results['overall']['win_rate']
        average_tries_remaining = results['overall']['average_tries_remaining']

        return {'win_rate': win_rate, 'average_tries_remaining': average_tries_remaining}

    def should_save_best_model(self, eval_results):
        """
        Determine if the current model is the best based on win rate.
        """
        current_metric_value = eval_results['win_rate']  # Example criterion
        if current_metric_value > self.best_metric_value:
            self.best_metric_value = current_metric_value
            return True
        return False


