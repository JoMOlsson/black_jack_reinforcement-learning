import random


class Dealer:

    MINIMUM_CARD_LIMIT = 52

    def __init__(self):
        self.number_of_decks = 5
        self.stack = Dealer.shuffle(self)
        self.minimum_card_limit = 52
        self.deck_count = 0

    def get_new_stack(self):
        stack = []
        for _ in range(0, self.number_of_decks):
            for card in range(1, 14):
                # Add one card per card-denomination
                stack.extend([card]*4)
        return stack

    def shuffle(self):
        stack = Dealer.get_new_stack(self)
        random.shuffle(stack)
        self.deck_count = 0
        return stack

    @staticmethod
    def card_count(cards):
        cards = [card if card not in [11, 12, 13] else 10 for card in cards]
        sum1 = sum(cards)

        usable_ace = 0
        if 1 in cards:
            sum2 = sum(cards) + 10
            usable_ace = 1
        else:
            sum2 = sum1

        max_card_count = [s for s in [sum1, sum2] if s <= 21]
        if max_card_count:
            count = max(max_card_count)
        else:
            count = min([sum1, sum2])

        return count, usable_ace

    def draw_card(self):
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

    def dealer_play(self, dealer_cards):
        dealer_satisfied = False
        dealer_count = None
        while not dealer_satisfied:
            dealer_cards.append(Dealer.draw_card(self))
            dealer_count, playable_ace = Dealer.card_count(dealer_cards)
            if dealer_count >= 17:
                dealer_satisfied = True
        return dealer_count, dealer_cards

    @staticmethod
    def has_black_jack(player_cards, player_count):
        return len(player_cards) == 2 and player_count == 21

    @staticmethod
    def check_outcome(player_count, dealer_count):
        if player_count > 21:
            player_count = 0

        if dealer_count > 21:
            dealer_count = 0

        if player_count > dealer_count:
            outcome = 1
        elif player_count < dealer_count or player_count == 0:
            outcome = -1
        else:
            outcome = 0
        return outcome

    def take_action(self, action, player_cards, dealer_cards):
        game_over = False
        if action == 1:
            # hit
            player_cards.append(Dealer.draw_card(self))
            player_count, playable_ace_p = Dealer.card_count(player_cards)
            if player_count > 21:
                reward = -1
                game_over = True
            else:
                reward = 0
            dealer_count, dealer_ace = Dealer.card_count(dealer_cards)
        else:
            # Stand
            game_over = True
            dealer_count, dealer_cards = Dealer.dealer_play(self, dealer_cards)
            player_count, playable_ace_p = Dealer.card_count(player_cards)
            outcome = Dealer.check_outcome(player_count, dealer_count)  # Calculate outcome

            if outcome == 1:
                reward = 1
            elif outcome == -1:
                reward = -1
            else:
                reward = 0

            if self.has_black_jack(player_cards, player_count):
                reward = 1

        return reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over

    def play_manual(self, num_of_games=1):
        for i in range(0, num_of_games):
            print("----- NEW GAME -----")
            player_cards, player_count, dealer_cards, dealer_count, playable_ace_p = Dealer.new_game(self)

            user_satisfied = False
            if 1 in player_cards and (10 in player_cards or 11 in player_cards
                                      or 12 in player_cards or 13 in player_cards):
                print("You got black-jack")
                user_satisfied = True
            else:
                print("Deck count is %d" % self.deck_count)
                print("Your hand is %d" % player_count)
                print("Dealer hand is %d" % dealer_count)
                action = input("Do you want to draw a card or stay?")
                if not action == "draw":
                    user_satisfied = True

            user_lost = False
            while not user_satisfied and not user_lost:
                player_cards.append(Dealer.draw_card(self))
                player_count, playable_ace = Dealer.card_count(player_cards)
                print("Your hand is %d" % player_count)
                print("Deck count is %d" % self.deck_count)
                if player_count > 21:
                    print("You lost!")
                    user_lost = True
                elif player_count == 21:
                    print("congrats, you got 21!")
                    user_satisfied = True
                else:
                    action = input("Do you want to draw a card or stay?")
                    if action == "stay":
                        user_satisfied = True

            if not user_lost:
                dealer_count, dealer_cards = Dealer.dealer_play(self, dealer_cards)
                outcome = Dealer.check_outcome(player_count, dealer_count)

                print("Dealer count is %d" % dealer_count)
                if outcome == 1:
                    print("Congratulation, you win! ")
                elif outcome == -1:
                    print("You lost")
                else:
                    print("It's a draw!")
