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
    from urlparse import parse_qs, urlparse # type: ignore
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning # type: ignore #noqa

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from .guess_strategies import GuessStrategy

class HangmanAPI(object):
    def __init__(self, access_token=None, \
                session=None, timeout=None, \
                *, dict_path: str, \
                strategy: GuessStrategy | None = None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []

        full_dictionary_location = "words_250000_train.txt" if dict_path is None else dict_path
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter(
            "".join(self.full_dictionary)
        ).most_common()

        self.current_dictionary = []

        self.strategy: GuessStrategy = strategy

    @staticmethod
    def determine_hangman_url():
        links = ["https://trexsim.com", "https://sg.trexsim.com"]

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += "/trexsim/hangman"
        return link

    def guess(self, word):  # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################
        guess_letter = self.strategy(word, self)
        if guess_letter not in self.guessed_letters:
            self.guessed_letters.append(guess_letter)
        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        response = self.request("/new_game", {"practice": practice})
        if response.get("status") == "approved":
            game_id = response.get("game_id")
            word = response.get("word")
            tries_remains = response.get("tries_remains")
            if verbose:
                print(
                    "Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(
                        game_id, tries_remains, word
                    )
                )
            while tries_remains > 0:
                # get guessed letter from user code
                guess_letter = self.guess(word)

                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))

                try:
                    res = self.request(
                        "/guess_letter",
                        {
                            "request": "guess_letter",
                            "game_id": game_id,
                            "letter": guess_letter,
                        },
                    )
                except HangmanAPIError:
                    print("HangmanAPIError exception caught on request.")
                    continue
                except Exception as e:
                    print("Other exception caught on request.")
                    raise e

                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get("status")
                tries_remains = res.get("tries_remains")
                if status == "success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status == "failed":
                    reason = res.get("reason", "# of tries exceeded!")
                    if verbose:
                        print(
                            "Failed game: {0}. Because of: {1}".format(game_id, reason)
                        )
                    return False
                elif status == "ongoing":
                    word = res.get("word")
        else:
            if verbose:
                print("Failed to start a new game")
        return status == "success"

    def my_status(self):
        return self.request("/my_status", {})

    def request(self, path, args=None, post_args=None, method=None):
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
                    verify=False,
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
        if "json" in headers["content-type"]:
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
            raise HangmanAPIError("Maintype was not text, or querystring")

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
