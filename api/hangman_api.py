import json
import requests
import random
import string
import secrets
import time
import re
import collections

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# from src.datamodule import encode_character, ProcessWordTransform
# from src.model.inference import guess_character

import torch

# class HangmanAPI(object):
#     def __init__(self, model=None, corpus_path=None, solver=None, \
#                  access_token=None, session=None, timeout=None):

from typing import Optional
import torch.nn as nn
# from hangman_game.observation_builder import state_encoder

# from src.datamodule.transforms import ProcessWordTransform

class HangmanAPI:

    def __init__(self, model: Optional[nn.Module] = None, 
                corpus_path: Optional[str] = None, 
                access_token: Optional[str] = None, 
                session: Optional[object] = None, 
                timeout: Optional[float] = None):        
        
        # # ===============api=======================
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        # # ======================================
        
        # self.guessed_letters = [] # api + guess
        
        full_dictionary_location = corpus_path
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        # self.full_dictionary_common_letter_sorted = \
        #     collections.Counter("".join(self.full_dictionary)).most_common()
        
        # print(self.full_dictionary_common_letter_sorted)
        
        # self.current_dictionary = [] # api, dont delet
        
        # # ===============custom=======================
        self.model = model
        # self.solver = solver
        # self.transform = transform
        # self.process_word_fn = process_word_fn

        # self.current_attempt = 0  # Initialize the list to track game results

    # custom function
    def update_game_stage(self, masked_word):
        word_len = len(masked_word)
        # print(word_len)
        # Determine incorrect guesses based on their presence in the masked word
        incorrect_guesses = {letter for letter in self.guessed_letters \
                if all(letter != ch for ch in masked_word if ch != '_')}
        # print(incorrect_guesses)
        incorrect_guesses_count = len(incorrect_guesses)
        remaining_attempts = 6 - incorrect_guesses_count
        # print(remaining_attempts)
        return remaining_attempts


    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # Clean the word so that we strip away the space characters
        # and replace "_" with "." as "." indicates any character in regular expressions
        masked_word = word[::2]  # Removing spaces between characters
        

        # print(f"-------------------------------------------")
        # print(f"Data available to make next guess...")
        tries_remains = self.update_game_stage(masked_word)     
    
        # print(f"-------------------------------------------")
        # print(f"Data available to make next guess...")
        # print(f"current state: {type(word)}")
        # print(f"Gussed letters: {type(self.guessed_letters)}")
        # print(f"Tries remainig: {type(tries_remains)} / {6}")
        # print(f"-------------------------------------------")


        obs, action_masks, recurrent_states_actor, recurrent_states_critic \
            = state_encoder(word, \
                    self.guessed_letters, tries_remains, self.previous_action, \
                    self.recurrent_states_actor, self.recurrent_states_critic, \
                    recurrent_state_size=self.model.recurrent_state_size)

        # print(type(obs))
        # print(obs.shape)
        # print(type(action_masks))
        # print(action_masks.shape)

        (
            value_preds,
            actions,
            action_log_probs,
            recurrent_states_actor,
            recurrent_states_critic,
        
        ) = self.model.predict(
            action_masks.to(self.model.get_device()),
            obs.to(self.model.get_device()),
            recurrent_states_actor.to(self.model.get_device()),
            recurrent_states_critic.to(self.model.get_device()),
        )

        # print(actions)

        guessed_char_index = actions.item()  # Gets the scalar value from the tensor
        guessed_char = chr(ord('a') + guessed_char_index)
        
        # print(f"Gussed Letters: {self.guessed_letters}")
        # # Update the guessed letters list
        # if guessed_char and guessed_char not in self.guessed_letters:
        #     self.guessed_letters.append(guessed_char)

        # print('\n')
        return guessed_char, recurrent_states_actor, recurrent_states_critic

    

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
    
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary 
        # to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        self.previous_action = None
        self.recurrent_states_actor = None
        self.recurrent_states_critic = None
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter, recurrent_states_actor, recurrent_states_critic = self.guess(word)
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                self.previous_action = guess_letter
                self.recurrent_states_actor = recurrent_states_actor
                self.recurrent_states_critic = recurrent_states_critic

                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    # # ===============================
                    # print('\n') # TODO: remove latter
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                # print('\n') # TODO: remove latter
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
                # print('\n') # TODO: remove latter
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)