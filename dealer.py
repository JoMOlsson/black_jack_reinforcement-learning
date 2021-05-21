import random
import numpy as np


class Dealer:

    MINIMUM_CARD_LIMIT = 52

    def __init__(self):
        self.number_of_decks = 5
        self.stack = Dealer.shuffle(self)
        self.minimum_card_limit = 52
        self.deck_count = 0

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

    def take_action(self, action, player_cards, dealer_cards):
        """ Will take an action i the current game corresponding to the action value. If the action is 1, the player
        will hit, otherwise the player will stay.

        Possible actions: 0 = Stand
                          1 = Hit
                          2 = Double-down

        :param action: (int)
        :param player_cards: (list)
        :param dealer_cards: (list)
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
                reward = 1
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
        return reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over

    def play_manual(self, num_of_games: int = 1):
        """ Plays a number of interactive black-jack games. The number of games is decided by the input variable
        num_of_games.

        :param num_of_games: (int)
        :return:
        """
        # TODO: Support additional actions
        for i in range(0, num_of_games):
            print("----- NEW GAME -----")
            player_cards, player_count, dealer_cards, dealer_count, playable_ace_p = Dealer.new_game(self)

            user_satisfied = False
            if self.has_black_jack(player_cards, player_count):
                print("You got black-jack")
                user_satisfied = True
            else:
                print(f"Deck count is {self.deck_count}")
                print(f"Your hand is {player_count}")
                print(f"Dealer hand is {dealer_count}")
                action = input("Do you want to draw a card or stay?")
                if not action == "draw":
                    user_satisfied = True

            user_lost = False
            while not user_satisfied and not user_lost:
                player_cards.append(Dealer.draw_card(self))
                player_count, playable_ace = Dealer.card_count(player_cards)
                print(f"Your hand is {player_count}")
                print(f"Deck count is {self.deck_count}")
                if not len(player_count):
                    print("You lost!")
                    user_lost = True
                elif 21 in player_count:
                    print("congrats, you got 21!")
                    user_satisfied = True
                else:
                    action = input("Do you want to draw a card or stay?")
                    if action == "stay":
                        user_satisfied = True

            if not user_lost:
                dealer_count, dealer_cards = Dealer.dealer_play(self, dealer_cards)
                outcome = Dealer.check_outcome(player_count, dealer_count)

                print(f"Dealer count is {dealer_count}")
                if outcome == 1:
                    print("Congratulation, you win! ")
                elif outcome == -1:
                    print("You lost")
                else:
                    print("It's a draw!")
