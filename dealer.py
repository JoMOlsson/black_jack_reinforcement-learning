import random
import numpy as np
import copy
import os
import json


class Dealer:

    MINIMUM_CARD_LIMIT = 52

    def __init__(self):
        self.number_of_decks = 5
        self.stack = Dealer.shuffle(self)
        self.minimum_card_limit = 52
        self.deck_count = 0
        self.active_games = []
        self.q = {}

    def get_new_stack(self):
        """ Will return a stack of cards according to the number of decks specified in the self.number_of_decks
        variable

        :return stack (list): Cards stack
        """
        stack = []
        for _ in range(0, self.number_of_decks):
            for card in range(1, 14):
                stack.extend([card]*4)  # Add one card per card-denomination
        return stack

    def shuffle(self):
        """ Fetches a new stack of card and shuffles the stack. The method also resets the deck count.

        :return stack (list): Cards stack
        """
        stack = Dealer.get_new_stack(self)
        random.shuffle(stack)
        self.deck_count = 0
        return stack

    @staticmethod
    def card_count(cards: list):
        """ Given a list of cards the method will calculate the card-count

        :param cards: (list) List of cards
        :return count, usable_ace: (list, int) maximum card-count, usable ace exist
        """
        cards = [card if card not in [11, 12, 13] else 10 for card in cards]
        sum1 = sum(cards)

        usable_ace = 0
        if 1 in cards:
            sum2 = sum(cards) + 10
            usable_ace = 1
        else:
            sum2 = sum1

        count = sorted(list(set([s for s in [sum1, sum2] if s <= 21])))  # Sort, remove duplicates and fat sums
        return count, usable_ace

    def draw_card(self):
        """ Draws the next card in the stack and updates the deck-count

        :return card: (int) drawn card
        """
        card = self.stack[0]

        # Update the deck count
        if card in [2, 3, 4, 5, 6]:
            self.deck_count += 1
        elif card in [10, 11, 12, 13, 1]:
            self.deck_count -= 1
        else:
            pass
        self.stack.pop(0)
        return card

    def new_game(self):
        """ Sets up a new game of black jack. If the number of remaining cards in the stack is lower than the
        minimum accepted card limit, a new shuffled stack is returned.

        :return player_cards, player_count, dealer_cards, dealer_count, playable_ace_p: (list, int, list, int, int)
        """
        player_cards = []
        dealer_cards = []

        if len(self.stack) < Dealer.MINIMUM_CARD_LIMIT:
            self.stack = Dealer.shuffle(self)

        # Deal
        player_cards.append(Dealer.draw_card(self))
        player_cards.append(Dealer.draw_card(self))

        dealer_cards.append(Dealer.draw_card(self))

        player_count, playable_ace_p = Dealer.card_count(player_cards)
        dealer_count, playable_ace_d = Dealer.card_count(dealer_cards)

        game = {'player_cards': player_cards, 'player_count': player_count, 'dealer_cards': dealer_cards,
                'dealer_count': dealer_count, 'playable_ace_p': playable_ace_p}
        self.active_games.append(game)
        return player_cards, player_count, dealer_cards, dealer_count, playable_ace_p

    def dealer_play(self, dealer_cards: list):
        """ Lets the dealer play until satisfaction. The dealer will draw cards until the card-count is larger or equal
        to 17.

        :param dealer_cards: (list) List of dealer cards
        :return dealer_count, dealer_cards: (list, list)
        """
        dealer_satisfied = False
        dealer_count = None
        while not dealer_satisfied:
            dealer_cards.append(Dealer.draw_card(self))
            dealer_count, playable_ace = Dealer.card_count(dealer_cards)
            if not len(dealer_count) or max(dealer_count) >= 17:
                dealer_satisfied = True
        return dealer_count, dealer_cards

    @staticmethod
    def has_black_jack(player_cards: list, player_count: list):
        """ Checks if black-jack is achieved

        :param player_cards: (list) List of cards
        :param player_count: (list) Player count
        :return:
        """
        return len(player_cards) == 2 and 21 in player_count

    @staticmethod
    def double_down_possible(cards):
        """ Check if double down action is possible. Double down is possible if the number of player cards is 2.

        :param cards: (list)
        :return:
        """
        return int(len(cards) == 2)

    @staticmethod
    def split_possible(cards):
        """ Check if split action is possible. Split is possible if the number of player cards is 2 and they are equal

        :param cards: (list)
        :return:
        """
        card_temp = [card if card not in [10, 11, 12, 13] else 10 for card in cards]
        return int(len(card_temp) == 2 and len(set(card_temp)) == 1)

    @staticmethod
    def construct_count_tag(count: list):
        """ Takes a card count (list) and converts it to a count-tag (str). If multiple card counts are present in the
        count list, the method will return all counts separated with dashes.

        :param count: (list)
        :return count_tag: (str)
        """
        count_tag = ''
        for i, c in enumerate(count):
            if i < len(count) - 1:
                count_tag += str(c) + '-'
            else:
                count_tag += str(c)
        return count_tag

    @staticmethod
    def check_outcome(player_count: list, dealer_count: list):
        """ Takes the player count (list) and the dealer count (list) and computes the game outcome. If a count is
        empty, the count is interpreted as above 21.

        :param player_count: (list)
        :param dealer_count: (list)
        :return outcome: (int)
        """
        if not len(player_count):
            return -1
        elif not len(dealer_count):
            return 1
        else:
            best_player_value = max([c for c in player_count if c <= 21])
            best_dealer_value = max([c for c in dealer_count if c <= 21])

        if best_player_value > best_dealer_value:
            outcome = 1
        elif best_player_value < best_dealer_value:
            outcome = -1
        else:
            outcome = 0
        return outcome

    def take_action(self, action, player_cards, dealer_cards, state=''):
        """ Will take an action i the current game corresponding to the action value. If the action is 1, the player
        will hit, otherwise the player will stay.

        Possible actions: 0 = Stand
                          1 = Hit
                          2 = Double-down
                          3 = Split

        :param action: (int)
        :param player_cards: (list)
        :param dealer_cards: (list)
        :param state: (str)
        :return:
        """
        game_over = False
        if action == 0:
            # ---- Stand ----
            game_over = True
            dealer_count, dealer_cards = Dealer.dealer_play(self, dealer_cards)  # play dealer
            player_count, playable_ace_p = Dealer.card_count(player_cards)       # Receive card-count
            outcome = Dealer.check_outcome(player_count, dealer_count)           # Calculate outcome
            reward = outcome
            if self.has_black_jack(player_cards, player_count):
                reward = 1.5
        elif action == 1:
            # ---- Hit ----
            player_cards.append(Dealer.draw_card(self))                          # Draw cards
            player_count, playable_ace_p = Dealer.card_count(player_cards)       # Receive card-count
            if not len(player_count):                                            # Player got fat
                reward = -1
                game_over = True
            else:
                reward = 0
            dealer_count, dealer_ace = Dealer.card_count(dealer_cards)
        elif action == 2:
            # ---- Double down ----
            if self.double_down_possible(player_cards):
                player_cards.append(Dealer.draw_card(self))                          # Draw card
                player_count, playable_ace_p = Dealer.card_count(player_cards)       # Receive card-count
                dealer_count, dealer_cards = Dealer.dealer_play(self, dealer_cards)  # Play dealer
                outcome = Dealer.check_outcome(player_count, dealer_count)           # Calculate outcome
                reward = 2 * outcome
            else:
                reward = -10  # Action not allowed
                player_count, playable_ace_p = Dealer.card_count(player_cards)       # Receive card-count
                dealer_count, _ = Dealer.card_count(dealer_cards)
            game_over = True
        elif action == 3:
            # ---- Split ----
            if self.split_possible(player_cards):
                player_cards.pop(0)
                dealer_count, _ = Dealer.card_count(dealer_cards)

                player_cards_next_game = copy.deepcopy(player_cards)
                dealer_cards_next_game = copy.deepcopy(dealer_cards)
                dealer_count_next_game = copy.deepcopy(dealer_count)

                player_cards.append(Dealer.draw_card(self))
                player_cards_next_game.append(Dealer.draw_card(self))
                player_count, playable_ace_p = Dealer.card_count(player_cards)
                player_count_next_game, playable_ace_p_next_game = Dealer.card_count(player_cards_next_game)

                if self.has_black_jack(player_cards_next_game, player_count_next_game):
                    reward_next_game = 1.5
                    game_over = True
                else:
                    reward_next_game = 0
                    game_over = False
                # reward_new = 1.5 if self.has_black_jack(player_cards_next_game, player_count_next_game) else 0
                game = {'player_cards': player_cards_next_game, 'player_count': player_count_next_game,
                        'dealer_cards': dealer_cards_next_game, 'dealer_count': dealer_count_next_game,
                        'playable_ace_p': playable_ace_p_next_game, 'state': state, 'action': action,
                        'reward': reward_next_game, 'game_over': game_over}
                self.active_games.append(game)
                reward = 1.5 if self.has_black_jack(player_cards, player_count) else 0
            else:
                reward = -10  # Action not allowed
                player_count, playable_ace_p = Dealer.card_count(player_cards)  # Receive card-count
                dealer_count, _ = Dealer.card_count(dealer_cards)
                game_over = True
        game = {'player_cards': player_cards, 'player_count': player_count, 'dealer_cards': dealer_cards,
                'dealer_count': dealer_count, 'playable_ace_p': playable_ace_p}
        self.active_games[0] = game
        return reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over

    def extract_state_string(self, player_count, dealer_count, player_cards, playable_ace_p):
        count_cards = True
        player_count_tag = self.construct_count_tag(player_count)
        dealer_count_tag = self.construct_count_tag(dealer_count)
        dd_possible = self.double_down_possible(player_cards)
        split_possible = self.split_possible(player_cards)
        if count_cards:
            state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{split_possible}_{dealer_count_tag}" \
                    f"_{self.deck_count}"
        else:
            state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{split_possible}_{dealer_count_tag}"
        return state

    def play_manual(self, num_of_games: int = 1, get_help=False):
        """ Plays a number of interactive black-jack games. The number of games is decided by the input variable
        num_of_games.

        :param num_of_games: (int)
        :param get_help: (Boolean)
        :return:
        """
        # Try to load Q-value
        if get_help:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, 'Q/Q.json')) as f:
                self.q = json.load(f)
                # TODO: FIX FIGURING OUT COUNT-CARD

        def get_help(p_cards, p_count, d_count, help_on):
            if help_on:
                play_ace_p = 1 if 1 in p_cards else 0
                state = self.extract_state_string(p_count, d_count, p_cards, play_ace_p)
                if state in self.q:
                    q_vals = self.q[state]
                    q_vals = np.array(q_vals)
                    a = np.argmax(q_vals)
                    str_action = 'stay' if a == 0 else 'hit' if a == 1 else 'double-down' if a == 3 else 'split'
                    print('Reinforcement-Agent: "I would ' + str_action + '"')
                else:
                    print('Reinforcement-Agent: "Hmm, I do not know what to do??"')

        def get_valid_action():
            user_action = input("Choose a valid action [stay, hit, double-down, split]")
            valid_action = False
            user_sat = False
            a = 0
            while not valid_action:
                if user_action == "stay":
                    a = 0
                    user_sat = True
                    valid_action = True
                elif user_action == "hit":
                    a = 1
                    valid_action = True
                elif user_action == "double-down":
                    a = 2
                    valid_action = True
                elif user_action == "split":
                    a = 3
                    valid_action = True
                else:
                    print("Action not valid! Choose one of these four [stay, hit, double-down, split]: ")
                    user_action = input("Choose a valid action [stay, hit, double-down, split]")
            return a, user_sat

        def show_state(deck_count, p_cards, p_count, d_cards, d_count):
            str_p_cards = [str(card) if card not in [1, 11, 12, 13] else 'ace' if card == 1 else 'jack' if card == 11
                           else 'queen' if card == 12 else 'king' for card in p_cards]
            str_d_cards = [str(card) if card not in [1, 11, 12, 13] else 'ace' if card == 1 else 'jack' if card == 11
                           else 'queen' if card == 12 else 'king' for card in d_cards]

            print(f"Deck count is {deck_count}")
            print(f"Your hand is {str_p_cards}, count is {p_count}")
            print(f"Dealer hand is {str_d_cards}, dealer count is {d_count}")

        for i in range(0, num_of_games):
            print("----- NEW GAME -----")
            player_cards, player_count, dealer_cards, dealer_count, playable_ace_p = Dealer.new_game(self)

            if self.has_black_jack(player_cards, player_count):
                print("You got black-jack")
                user_satisfied = True
            else:
                show_state(self.deck_count, player_cards, player_count, dealer_cards, dealer_count)
                get_help(player_cards, player_count, dealer_count, get_help)
                action, user_satisfied = get_valid_action()
            user_lost = False
            while not user_satisfied and not user_lost:
                reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over = \
                    self.take_action(action, player_cards, dealer_cards)
                if reward < -2:
                    print("Action not valid!")

                show_state(self.deck_count, player_cards, player_count, dealer_cards, dealer_count)
                if not len(player_count):
                    print("Sorry, You lost!")
                    user_lost = True
                elif 21 in player_count:
                    print("congrats, you got 21!")
                    user_satisfied = True
                else:
                    get_help(player_cards, player_count, dealer_count, get_help)
                    action, user_satisfied = get_valid_action()

            if not user_lost:
                dealer_count, dealer_cards = Dealer.dealer_play(self, dealer_cards)
                outcome = Dealer.check_outcome(player_count, dealer_count)
                print(f"Dealer count is {dealer_count}")
                if outcome >= 1:
                    print("Congratulation, you win! ")
                elif outcome <= -1:
                    print("You lost")
                else:
                    print("It's a draw!")
