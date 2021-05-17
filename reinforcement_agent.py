import numpy as np
import random
import matplotlib.pyplot as plt
import math
import re
import json
import os
import datetime
import time

from dealer import Dealer


class Agent:
    def __init__(self, alpha=0.3, discount=0.8, exploration_prob=0.3, count_cards=False, load_q_table=False):
        self.alpha = alpha
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.dealer = Dealer()
        self.COUNT_CARDS = count_cards
        if load_q_table:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, 'Q/Q.json')) as f:
                self.q = json.load(f)
        else:
            self.q = {}

    @staticmethod
    def change_action(action, p):
        rand_val = random.uniform(0, 1)
        action = action if rand_val > p else abs(action - 1)
        return action

    @staticmethod
    def extract_state_vectors(q):
        player_count = []
        playable_ace = []
        dealer_count = []
        deck_count = []
        action = []
        max_q_val = []

        state_vectors = {}
        for state in q:
            q_vals = np.array(q[state])
            action.append(np.argmax(q_vals))
            max_q_val.append(np.max(q_vals))

            delim_index = [m.start() for m in re.finditer('_', state)]
            num_of_vars = len(delim_index) + 1
            delim_index.append(-1)
            for i, delim in enumerate(delim_index):
                if i == 0:
                    pc = state[0:delim]
                    if '-' in pc:
                        pc = pc.split('-')[-1]
                    player_count.append(int(pc))
                elif i == 1:
                    playable_ace.append(int(state[delim_index[i - 1] + 1:delim]))
                elif i == 2:
                    if num_of_vars == 3:
                        dc = state[delim_index[i - 1] + 1:len(state)]
                    else:
                        dc = state[delim_index[i - 1] + 1:delim]
                    if '-' in dc:
                        dc = dc.split('-')[-1]
                    dealer_count.append(int(dc))
                else:
                    deck_count.append(int(state[delim_index[i - 1] + 1:len(state)]))

        state_vectors['player_count'] = player_count
        state_vectors['playable_ace'] = playable_ace
        state_vectors['dealer_count'] = dealer_count
        state_vectors['action'] = action
        state_vectors['max_q_val'] = max_q_val
        if deck_count:
            state_vectors['deck_count'] = deck_count

        return state_vectors

    @staticmethod
    def moving_average(data, window):
        n = math.ceil(len(data) / window)
        mov_average = [0] * n

        for i in range(0, n):
            if i == n - 1:
                avg = sum(data[window * i:]) / len(data[window * i:])
            else:
                d = data[window * i:window * (i + 1) - 1]
                avg = sum(d) / len(d)
            mov_average[i] = avg
        return mov_average

    def train_agent(self, num_of_episodes=1000000, azim_start=0, extract_performance_every=np.inf):
        azim = azim_start                            # Azim angle start, used for performance visualisation
        q = self.q                                   # Get Q-table
        explore_stop = round(num_of_episodes * 0.8)  # Stop exploration after 80% of training

        print_tick_lim = 50000                       # Number of episodes between status print-out
        n = 0                                        # Print-out tick count
        print_tick = 0                               # Episode tick count
        outcome_list = [0] * num_of_episodes         # List holding episode outcomes
        performance_count = 0
        for i_episode in range(0, num_of_episodes):
            n += 1
            performance_count += 1
            if n >= print_tick_lim:
                print_tick += n
                n = 0
                print("Executed another %d episodes, total episode count %d" % (print_tick_lim, print_tick))

            player_cards, player_count, dealer_cards, dealer_count, playable_ace_p = self.dealer.new_game()

            player_count_tag = self.dealer.construct_count_tag(player_count)
            dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
            if self.COUNT_CARDS:
                state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}_{self.dealer.deck_count}"
            else:
                state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}"

            # Add state to Q-table
            if state not in q:
                q[state] = [0, 0]

            q_vals = np.array(q[state])
            action = np.argmax(q_vals)

            # Epsilon greedy
            if i_episode < explore_stop:
                action = self.change_action(action, self.exploration_prob)

            game_over = False
            reward = 0
            while not game_over:
                # Take action
                reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over = \
                    self.dealer.take_action(action, player_cards, dealer_cards)
                player_count_tag = self.dealer.construct_count_tag(player_count)
                dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
                if self.COUNT_CARDS:
                    new_state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}_{self.dealer.deck_count}"
                else:
                    new_state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}"

                if game_over:
                    q_new_state = [reward, reward]
                else:
                    if new_state not in self.q:
                        q_new_state = [0, 0]
                        q[new_state] = q_new_state
                    else:
                        q_new_state = q[new_state]

                q[state][action] = q[state][action] + self.alpha * (reward + self.discount * max(q_new_state)
                                                                    - q[state][action])
                # Move to new state
                state = new_state
                q_vals = np.array(q_new_state)
                action = np.argmax(q_vals)
                if i_episode < explore_stop:
                    action = self.change_action(action, self.exploration_prob)
            outcome_list[i_episode] = reward
            if performance_count > extract_performance_every:
                performance_count = 0
                o = outcome_list[i_episode - extract_performance_every + 1:i_episode]
                self.visualize_progress(o, i_episode, azim)
                azim += 1

        # ---- Save q-table ----
        with open('Q.json', 'w') as outfile:
            json.dump(q, outfile)
        self.q = q

        # TODO: Put this in a function
        # bin_size = 1000
        # n = math.ceil(len(outcome_list) / bin_size)
        # bin_count = 0
        # X = []
        # Y = []
        # for i in range(n):
        #     if i < n - 1:
        #         outcome = np.array(outcome_list[bin_count:bin_count + bin_size])
        #     else:
        #         outcome = np.array(outcome_list[bin_count:-1])
        #     idx = np.where(outcome != 0)[0]
        #     outcome = np.array([o for i, o in enumerate(outcome) if i in idx])
        #     n_win = np.where(outcome > 0)[0]
        #     win_ratio = len(n_win) / len(outcome)
        #     bin_count += bin_size
        #     X.append(bin_count)
        #     Y.append(win_ratio)
        # plt.plot(X, Y)
        # plt.show()

        # ---- Visualize outcome ----
        # avg = self.moving_average(outcome_list, 10000)
        # plt.plot(avg)
        # #plt.show()

        self.visualize_action_surface()
        self.evaluate_performance()
        return q

    @staticmethod
    def sample_matching_deck_count(i_sample, deck_count, states):
        return states["playable_ace"][i_sample] == 0 \
               and states["deck_count"][i_sample] == deck_count \
               and states["player_count"][i_sample] <= 21 \
               and states["dealer_count"][i_sample] <= 11

    @staticmethod
    def get_samples(i_sample, states):
        return states["playable_ace"][i_sample] == 0 \
               and states["player_count"][i_sample] <= 21 \
               and states["dealer_count"][i_sample] <= 11

    def visualize_action_surface(self):
        # TODO: Add visualization for not count cards
        # TODO: Refactor
        show_actions = False  # Show either actions or w-values
        azim = 0

        state_vectors = self.extract_state_vectors(self.q)
        has_deck_count = len(state_vectors) > 5
        n_states = len(state_vectors['player_count'])
        if has_deck_count:
            n_image = 0
            deck_c_start = -20
            deck_c_end = 20
            deck_c = list(np.linspace(deck_c_start, deck_c_end, abs(deck_c_start - deck_c_end), dtype=int))
            for i, dc in enumerate(deck_c):
                valid_index = list(filter(lambda i_sample: self.sample_matching_deck_count(i_sample, dc, state_vectors),
                                          range(0, n_states)))
                player_count = np.array(state_vectors["player_count"])[valid_index]
                dealer_count = np.array(state_vectors["dealer_count"])[valid_index]
                action = np.array(state_vectors["action"])[valid_index]
                q_val = np.array(state_vectors["max_q_val"])[valid_index]

                player_count_unique = sorted(list(set(player_count)))
                dealer_count_unique = sorted(list(set(dealer_count)))
                hit_stay = np.zeros(len(player_count_unique) * len(dealer_count_unique)).reshape(
                    len(player_count_unique), len(dealer_count_unique))

                player_count = list(player_count)
                dealer_count = list(dealer_count)
                action = list(action)
                for idx_pcount, p_count in enumerate(player_count_unique):
                    for idx_dcount, d_count in enumerate(dealer_count_unique):
                        action_value = [x for i, x in enumerate(action) if
                                        dealer_count[i] == d_count and player_count[i] == p_count]
                        q_value = [x for i, x in enumerate(q_val) if
                                   dealer_count[i] == d_count and player_count[i] == p_count]
                        if show_actions:
                            if action_value:
                                hit_stay[idx_pcount][idx_dcount] = action_value[0]
                            else:
                                hit_stay[idx_pcount][idx_dcount] = np.nan
                        else:
                            if q_value:
                                hit_stay[idx_pcount][idx_dcount] = q_value[0]
                            else:
                                hit_stay[idx_pcount][idx_dcount] = np.nan
                # ---- Visualize grid ----
                x, y = np.meshgrid(dealer_count_unique, player_count_unique)
                ax = plt.axes(projection="3d")
                ax.plot_surface(x, y, hit_stay, rstride=1, cstride=1, cmap='winter', edgecolor='none')
                ax.set_title('Deck-Count: ' + str(dc))
                ax.set_xlabel('dealer_count')
                ax.set_ylabel('player_count')
                ax.set_zlabel('Hit or stay')
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
                ax.set_zlim([-1, 1])
                basename = 'action_surface_' if show_actions else 'q_surface_'
                plt.savefig(basename + str(n_image) + '.png')
                n_image += 1
                ax.view_init(elev=25, azim=azim)
                # for i_im in range(9):
                #     im_name = '00' + str(n_image) if n_image < 10 else '0' + str(n_image) if n_image < 100 else str(n_image)
                #     ax.view_init(elev=25, azim=azim)
                #     if azim != 360:
                #         plt.savefig(basename + im_name + '.png')
                #         #plt.savefig(basename + str(n_image) + '.png')
                #     n_image += 1
                #     azim += 1

    def visualize_progress(self, outcome, episode_count, azim):
        idx = np.where(np.array(outcome) != 0)[0]
        outcome = np.array([o for i, o in enumerate(outcome) if i in idx])
        n_win = np.where(outcome > 0)[0]
        win_ratio = len(n_win) / len(outcome)

        show_actions = False
        state_vectors = self.extract_state_vectors(self.q)
        im_name = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S').replace('-', '_').replace(':', '_')
        n_states = len(state_vectors['player_count'])

        valid_index = list(filter(lambda i_sample: self.get_samples(i_sample, state_vectors), range(0, n_states)))
        player_count = np.array(state_vectors["player_count"])[valid_index]
        dealer_count = np.array(state_vectors["dealer_count"])[valid_index]
        action = np.array(state_vectors["action"])[valid_index]
        q_val = np.array(state_vectors["max_q_val"])[valid_index]

        player_count_unique = sorted(list(set(player_count)))
        dealer_count_unique = sorted(list(set(dealer_count)))
        hit_stay = np.zeros(len(player_count_unique) * len(dealer_count_unique)).reshape(
            len(player_count_unique), len(dealer_count_unique))

        player_count = list(player_count)
        dealer_count = list(dealer_count)
        action = list(action)
        for idx_pcount, p_count in enumerate(player_count_unique):
            for idx_dcount, d_count in enumerate(dealer_count_unique):
                action_value = [x for i, x in enumerate(action) if
                                dealer_count[i] == d_count and player_count[i] == p_count]
                q_value = [x for i, x in enumerate(q_val) if
                           dealer_count[i] == d_count and player_count[i] == p_count]
                if show_actions:
                    if action_value:
                        hit_stay[idx_pcount][idx_dcount] = action_value[0]
                    else:
                        hit_stay[idx_pcount][idx_dcount] = np.nan
                else:
                    if q_value:
                        hit_stay[idx_pcount][idx_dcount] = q_value[0]
                    else:
                        hit_stay[idx_pcount][idx_dcount] = 0
        # ---- Visualize grid ----
        x, y = np.meshgrid(dealer_count_unique, player_count_unique)
        ax = plt.axes(projection="3d")
        ax.plot_surface(x, y, hit_stay, rstride=1, cstride=1, cmap='winter', edgecolor='none')
        ax.set_title('Win-ratio: ' + str(round(win_ratio*100)/100) + ', episode count: ' + str(episode_count))
        ax.set_xlabel('dealer_count')
        ax.set_ylabel('player_count')
        ax.set_zlabel('Hit or stay')
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        ax.set_zlim([-1, 1])
        ax.view_init(elev=25, azim=azim)
        plt.savefig(im_name + '.png')
        time.sleep(1)

    def evaluate_performance(self):
        policy = ['optimal', 'strict', 'random']

        for iPolicy in range(0, len(policy)):
            num_of_black_jack = 0
            num_of_eval_episodes = 100000
            win_list = [0] * num_of_eval_episodes
            unexplored_states = 0

            print("===== Running with policy: " + policy[iPolicy] + " =====")
            for i in range(0, num_of_eval_episodes):
                player_cards, player_count, dealer_cards, dealer_count, playable_ace_p = self.dealer.new_game()
                player_count_tag = self.dealer.construct_count_tag(player_count)
                dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
                if self.dealer.has_black_jack(player_cards, player_count):
                    num_of_black_jack += 1

                if self.COUNT_CARDS:
                    state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}_{self.dealer.deck_count}"
                else:
                    state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}"

                # Add state to Q-table
                if state not in self.q:
                    unexplored_states += 1
                    self.q[state] = [0, 0]
                q_vals = np.array(self.q[state])
                action = np.argmax(q_vals)

                # Adapt action based on policy
                if policy[iPolicy] == 'strict':
                    if len(player_count) and max(player_count) < 18:
                        action = 1
                    else:
                        action = 0
                elif policy[iPolicy] == 'random':
                    action = random.randint(0, 1)
                else:
                    pass

                game_over = False
                while not game_over:
                    # Take action
                    reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over = \
                        self.dealer.take_action(action, player_cards, dealer_cards)
                    player_count_tag = self.dealer.construct_count_tag(player_count)
                    dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
                    if self.COUNT_CARDS:
                        new_state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}_{self.dealer.deck_count}"
                    else:
                        new_state = f"{player_count_tag}_{playable_ace_p}_{dealer_count_tag}"

                    if game_over:
                        q_new_state = [reward, reward]
                    elif new_state not in self.q:
                        q_new_state = [0, 0]
                        self.q[new_state] = q_new_state
                    else:
                        q_new_state = self.q[new_state]

                    # Move to new state
                    q_vals = np.array(q_new_state)
                    action = np.argmax(q_vals)

                    if policy[iPolicy] == 'strict':
                        if len(player_count) and max(player_count) < 18:
                            action = 1
                        else:
                            action = 0
                    elif policy[iPolicy] == 'random':
                        action = random.randint(0, 1)
                    else:
                        pass
                win_list[i] = reward

            wins = [x for x in win_list if x == 1]
            loose = [x for x in win_list if x == -1]
            draw = [x for x in win_list if x == 0]
            r_wins = round((len(wins) / len(win_list)) * 1000) / 10
            r_draw = round((len(draw) / len(win_list)) * 1000) / 10
            r_loose = round((len(loose) / len(win_list)) * 1000) / 10
            r_black_jack = round((num_of_black_jack / len(win_list)) * 1000) / 10
            num_episodes = len(win_list)
            win_factor = (len(wins) - len(loose) + 0.5 * num_of_black_jack) / num_episodes

            # Print evaluation outcome
            print("Evaluate algorithm")
            print("Found %d new states" % unexplored_states)
            print("Number of wins: %d , %f" % (len(wins), r_wins))
            print("Number of draw: %d , %f" % (len(draw), r_draw))
            print("Number of lost games: %d , %f" % (len(loose), r_loose))
            print("Number of black jacks : %d , %f" % (num_of_black_jack, r_black_jack))
            print("The win factor is: %f" % win_factor)
