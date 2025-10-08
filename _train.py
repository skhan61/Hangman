# import torch
# import numpy as np
# import random
# from supervised_learning._data.data_module import HangmanDataModule
# from supervised_learning._model.components.neural_net import EmbeddingGRU
# import os

# from src.utils import read_word_list
# from src.data_generation.dataset_analysis import stratified_sample_from_categories, \
#     classify_words_by_unique_letter_count, summarize_categories, categorize_words_by_length
# import collections

# from lightning import Trainer
# torch.set_float32_matmul_precision('medium') #  | 'high')
# from supervised_learning._model.callback import SimulationCallback
# from lightning.pytorch.callbacks import ModelCheckpoint

# from supervised_learning._model.hangman_model import HangmanModel

# from supervised_learning._model.simulation import play_a_game_with_a_word, \
#                             simulate_games_for_word_list, guess # testing function
# from lightning.pytorch.loggers import WandbLogger
# import wandb
# from supervised_learning._data.data_utils import encode_word, custom_collate_fn
# from torch.utils.data import DataLoader, random_split

# def set_seed(seed):
#     """Set seed for reproducibility."""
#     random.seed(seed)       # Python random module
#     np.random.seed(seed)    # Numpy module
#     torch.manual_seed(seed) # PyTorch
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)          # Sets seed for CUDA (GPU)
#         torch.cuda.manual_seed_all(seed)      # Ensure reproducibility on all GPUs
#         torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
#         torch.backends.cudnn.benchmark = False     # If input sizes do not vary, this should be set to False

# # Example usage: 
# set_seed(42)  # Use any number to seed all libraries

# corpus_path = '/media/sayem/510B93E12554BBD1/Hangman/data/words_250000_train.txt'
# total_samples = 250_000  # This can be adjusted as needed
# # Read a specified number of words from the corpus
# corpus = read_word_list(corpus_path, num_samples=total_samples)
# corpus_string = ''.join(corpus).lower()  # Join words and convert to lowercase
# # Count the frequency of each character
# counter = collections.Counter(corpus_string)
# # Extract the frequency of letters a-z (assuming all letters appear at least once)
# frequencies = np.array([counter.get(chr(i), 1) for i in range(97, 123)])  # Default to 1 to avoid division by zero

# # Calculate inverse frequencies
# inverse_frequencies = 1.0 / frequencies

# # Normalize weights if necessary
# max_freq = inverse_frequencies.max()
# normalized_weights = torch.tensor(inverse_frequencies / max_freq, dtype=torch.float)
# # print(normalized_weights.shape)

# # dataset_dir = '/media/sayem/510B93E12554BBD1/dataset'
# dataset_dir = '/media/sayem/510B93E12554BBD1/dataset'
# config = {
#     'data_dir': corpus_path,
#     'batch_size': 512,
#     'dataset_dir': dataset_dir,
#     'vocab_size': 26,           # Example size of the vocabulary
#     'min_length': 3,            # Example minimum length of words to be processed
#     'max_epochs': 200,
#     'drop_uniformly': False,
#     'integer_indices': True, # False for one-hot encodi
#     'num_workers': os.cpu_count(),
#     'embedding_dim': 256,
#     'hidden_dim': 1024, # rnn hidden dim
#     'num_layers': 4, # number of rnn layers
#     'dropout': 0.3, # dropout
#     'miss_chars_fc_out_dim': 512,
#     'nn_model': EmbeddingGRU,
#     'pos_weight': normalized_weights,
#     'optimizer_config': {
#         'type': torch.optim.Adam,
#         'params': {
#             'lr': 0.0005,
#             'betas': (0.9, 0.999),
#             'eps': 1e-08,
#             'weight_decay': 0.0,
#             'amsgrad': False
#         }
#     },
#     'scheduler_config': {
#         'type': torch.optim.lr_scheduler.StepLR,
#         'params': {
#             'step_size': 1,
#             'gamma': 0.1
#         }
#     }
# }

# # initialise the wandb logger and name your wandb project
# wandb_logger = WandbLogger(name="hangman_run", project="hangman", config=config)

# dm = HangmanDataModule(config)
# dm.prepare_data()
# dm.setup(stage='train')
# dm.setup(stage='validate')

            
# model = HangmanModel(config)

# corpus_path_ = '/media/sayem/510B93E12554BBD1/Hangman/data/20k.txt'
# word_list = read_word_list(corpus_path_, num_samples=1_000)
# simulation_callback = SimulationCallback(word_list=word_list)


# dirpath = "/media/sayem/510B93E12554BBD1/ckpt"
# # Create the directory if it does not exist
# if not os.path.exists(dirpath):
#     os.makedirs(dirpath)

# # Configure the checkpoint callback
# checkpoint_callback = ModelCheckpoint(
#     dirpath="/media/sayem/510B93E12554BBD1/ckpt",  
#     filename="{epoch}-{step}",  # Naming scheme
#     save_top_k=-1,  # Save all checkpoints
#     every_n_epochs=1  # Save every epoch
# )

# trainer = Trainer(max_epochs=config['max_epochs'], \
#             logger=wandb_logger, \
#             enable_checkpointing=True,
#             reload_dataloaders_every_n_epochs=1,
#             log_every_n_steps=10,
#             callbacks=[simulation_callback, checkpoint_callback],)
#             # limit_train_batches=0.1,  # 1% of training data
#             # limit_val_batches=1,)    # 50% of validation data)

# trainer.fit(model, dm)
# # trainer.validate(model, dm)

# wandb.finish()

# model = model.to('cpu')

# final_results = simulate_games_for_word_list(word_list=word_list, guess_function=guess, \
#                                             play_function=play_a_game_with_a_word, \
#                                             model=model) 
# # Print overall statistics
# overall_stats = final_results['overall']
# print("\nOverall Statistics:")
# print(f"Total Games: {overall_stats['total_games']}, Wins: {overall_stats['wins']}, Losses: {overall_stats['losses']}")
# print(f"Win Rate: {overall_stats['win_rate']:.2f}, Average_tries_remaining: {overall_stats['average_tries_remaining']:.2f}")

# # from supervised_learning.data.data_utils import encode_word, custom_collate_fn
# # from torch.utils.data import DataLoader, random_split

# # train_dataset = dm.train_dataset
# # val_dataset = dm.val_dataset

# # for epoch in range(config['max_epochs']):
# #     train_dataset.cur_epoch = epoch
# #     val_dataset.cur_epoch = epoch

# #     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
# #                             collate_fn=custom_collate_fn, shuffle=True, \
# #                             num_workers=config['num_workers'])

# #     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
# #                             collate_fn=custom_collate_fn, shuffle=False, \
# #                             num_workers=config['num_workers'])

# #     trainer = Trainer(max_epochs=1, \
# #                 logger=False, \
# #                 enable_checkpointing=False,
# #                 # reload_dataloaders_every_n_epochs=1,
# #                 callbacks=[simulation_callback],)
# #                 # limit_train_batches=1,  # 1% of training data
# #                 # limit_val_batches=1,)    # 50% of validation data)
    
# #     trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


pass