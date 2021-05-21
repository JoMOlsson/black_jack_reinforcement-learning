import numpy as np
import random
import matplotlib.pyplot as plt
import math
import re
import json
import os
import datetime

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
    def change_action(action, p, q_values):
        """ Used to implement an epsilon greedy policy. Given an action the method will change the
        action with a given probability specified by the provided probability value [0-1]. If the action is changed,
        it will be changed to any valid action given by the provided set of q-values. Q-values of value -10 are
        interpreted as a banned action.

        :param action: (int) Action value
        :param p: (float) Probability value between [0-1] determining the rate of action change
        :param q_values: (list) List of possible actions
        :return action: (int)
        """
        rand_val = random.uniform(0, 1)
        change_action = rand_val < p

        if change_action:
            valid_actions = np.arange(0, len(q_values), 1)
            additional_actions = np.delete(valid_actions, np.where(valid_actions == action))  # Remove given action
            additional_actions = np.delete(additional_actions, np.where(additional_actions == -10))  # Remove invl act
            action = int(additional_actions[random.randint(0, len(additional_actions)-1)])
        return action

    @staticmethod
    def extract_state_vectors(q):
        player_count = []  # List storing player counts
        playable_ace = []  # List storing usable ave
        dd_possible = []   # Double down possible
        dealer_count = []  # Dealer count
        deck_count = []    # Deck count
        action = []        # Storage for actions
        max_q_val = []     # Storing max_q_vals

        state_vectors = {}  # Dictionary storing state variables
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
                    dd_possible.append(int(state[delim_index[i - 1] + 1:delim]))
                elif i == 3:
                    if num_of_vars == 4:
                        dc = state[delim_index[i - 1] + 1:len(state)]
                    else:
                        dc = state[delim_index[i - 1] + 1:delim]
                    if '-' in dc:
                        dc = dc.split('-')[-1]
                    dealer_count.append(int(dc))
                else:
                    deck_count.append(int(state[delim_index[i - 1] + 1:len(state)]))
        # Adding state vectors
        state_vectors['player_count'] = player_count
        state_vectors['playable_ace'] = playable_ace
        state_vectors['dealer_count'] = dealer_count
        state_vectors['action'] = action
        state_vectors['max_q_val'] = max_q_val
        state_vectors['dd_possible'] = dd_possible
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
        """
        States: - Player cards
                - Playable ace
                - Double down possible
                - Dealer count
                - Deck count

        :param num_of_episodes:
        :param azim_start:
        :param extract_performance_every:
        :return:
        """
        # TODO: Refactor state extraction
        # TODO: Clean Code
        win_loss_ratio_list = []                     # List holding the win/loss ratio values
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
            dd_possible = self.dealer.double_down_possible(player_cards)
            if self.COUNT_CARDS:
                state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}_{self.dealer.deck_count}"
            else:
                state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}"

            # Add state to Q-table
            if state not in q:
                q[state] = [0, 0, 0]

            q_vals = q[state]
            action = np.argmax(q_vals)

            # Epsilon greedy
            if i_episode < explore_stop:
                action = self.change_action(action, self.exploration_prob, q_vals)

            # Check if black-jack
            if self.dealer.has_black_jack(player_cards, player_count):
                # Black Jack
                action = 0
                # Take action
                reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over = \
                    self.dealer.take_action(action, player_cards, dealer_cards)

                q_new_state = [reward, reward, reward]
                q[state][action] = q[state][action] + self.alpha * (reward + self.discount * max(q_new_state)
                                                                    - q[state][action])
            else:
                game_over = False

            reward = 0
            while not game_over:
                # Take action
                reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over = \
                    self.dealer.take_action(action, player_cards, dealer_cards)
                player_count_tag = self.dealer.construct_count_tag(player_count)
                dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
                dd_possible = self.dealer.double_down_possible(player_cards)
                if self.COUNT_CARDS:
                    new_state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}" \
                                f"_{self.dealer.deck_count}"
                else:
                    new_state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}"

                if game_over:
                    q_new_state = [reward, reward, reward]
                else:
                    if new_state not in self.q:
                        q_new_state = [0, 0, 0]
                        q[new_state] = q_new_state
                    else:
                        q_new_state = q[new_state]

                q[state][action] += self.alpha * (reward + self.discount * max(q_new_state) - q[state][action])

                # Move to new state
                if not game_over:
                    state = new_state
                    q_vals = np.array(q_new_state)
                    action = np.argmax(q_vals)
                    if i_episode < explore_stop:
                        action = self.change_action(action, self.exploration_prob, q_vals)
            outcome_list[i_episode] = reward

            # Evaluate performance
            if performance_count > extract_performance_every:
                self.q = q             # Assign q-table
                performance_count = 0  # re-initialize counter
                win_loss_ratio = self.get_performance(number_of_episodes=100_000)
                win_loss_ratio_list.append(win_loss_ratio)
                self.visualize_progress(win_loss_ratio, win_loss_ratio_list, i_episode, azim)
                azim += 1

        # ---- Save q-table ----
        with open('Q.json', 'w') as outfile:
            json.dump(q, outfile)
        self.q = q

        self.evaluate_performance()
        self.visualize_action_surface()
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

    def visualize_action_surface(self, rotation=True):
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
                if not rotation:
                    plt.savefig(basename + str(n_image) + '.png')
                    n_image += 1
                    ax.view_init(elev=25, azim=azim)
                else:
                    for i_im in range(9):
                        im_name = '00' + str(n_image) if n_image < 10 \
                            else '0' + str(n_image) if n_image < 100 \
                            else str(n_image)
                        ax.view_init(elev=25, azim=azim)
                        if azim != 360:
                            plt.savefig(basename + im_name + '.png')
                        n_image += 1
                        azim += 1

    def visualize_progress(self, win_ratio, win_loss_ratio_list, episode_count, azim, include_evaluation_plot=True):
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
        if not include_evaluation_plot:
            x, y = np.meshgrid(dealer_count_unique, player_count_unique)
            axs = plt.axes(projection="3d")
            axs.plot_surface(x, y, hit_stay, rstride=1, cstride=1, cmap='winter', edgecolor='none')
            axs.set_zlim(-1, 1)
            axs.set_title(f'Win-ratio: {round(win_ratio * 100) / 100} , episode count: {episode_count}', fontsize=18)
            axs.set_xlabel('dealer_count', fontsize=12)
            axs.set_ylabel('player_count', fontsize=12)
            axs.set_zlabel('Hit or stay', fontsize=12)
            axs.view_init(elev=25, azim=azim)
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.savefig(im_name + '.png')
        else:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            axs = fig.add_subplot(1, 2, 1, projection="3d")
            x, y = np.meshgrid(dealer_count_unique, player_count_unique)
            surf = axs.plot_surface(x, y, hit_stay, rstride=1, cstride=1, cmap='winter',
                                    linewidth=0, antialiased=False)
            axs.set_zlim(-1, 1)
            axs.set_title(f'Win-ratio: {round(win_ratio*100)/100} , episode count: {episode_count}',  fontsize=18)
            axs.set_xlabel('dealer_count',  fontsize=12)
            axs.set_ylabel('player_count',  fontsize=12)
            axs.set_zlabel('Hit or stay',  fontsize=12)
            axs.view_init(elev=25, azim=azim)
            fig.colorbar(surf, shrink=0.5, aspect=10)

            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')

            axs = fig.add_subplot(1, 2, 2)
            evaluation_rounds = list(range(len(win_loss_ratio_list)))
            axs.plot(evaluation_rounds, win_loss_ratio_list)
            axs.set_xlabel('Evaluation round',  fontsize=12)
            axs.set_ylabel('Win/loss ratio (%)',  fontsize=12)
            axs.set_ylim(30, 50)
            axs.set_xlim(0, len(evaluation_rounds))
            plt.savefig(im_name + '.png')
            plt.close(fig)

    def get_performance(self, number_of_episodes=100_000):
        win_list = [0] * number_of_episodes
        for i in range(0, number_of_episodes):
            player_cards, player_count, dealer_cards, dealer_count, playable_ace_p = self.dealer.new_game()
            player_count_tag = self.dealer.construct_count_tag(player_count)
            dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
            dd_possible = self.dealer.double_down_possible(player_cards)
            if self.COUNT_CARDS:
                state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}_{self.dealer.deck_count}"
            else:
                state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}"

            # Add state to Q-table
            if state not in self.q:
                self.q[state] = [0, 0, 0]

            q_vals = np.array(self.q[state])
            action = np.argmax(q_vals)

            if self.dealer.has_black_jack(player_cards, player_count):
                game_over = True
                reward = 1.5
            else:
                game_over = False
            while not game_over:
                # Take action
                reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over = \
                    self.dealer.take_action(action, player_cards, dealer_cards)
                player_count_tag = self.dealer.construct_count_tag(player_count)
                dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
                dd_possible = self.dealer.double_down_possible(player_cards)
                if self.COUNT_CARDS:
                    new_state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}_{self.dealer.deck_count}"
                else:
                    new_state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}"

                if game_over:
                    q_new_state = [reward, reward, reward]
                elif new_state not in self.q:
                    q_new_state = [0, 0, 0]
                    self.q[new_state] = q_new_state
                else:
                    q_new_state = self.q[new_state]

                # Move to new state
                q_vals = np.array(q_new_state)
                action = np.argmax(q_vals)

            win_list[i] = reward
        # Calculate win-loss ratio
        win_loss = np.where(np.array(win_list) != 0)[0]
        wins = np.where(np.array(win_list)[win_loss] == 1)[0]
        win_loss_ratio = round((len(wins) / len(win_loss)) * 1000) / 10
        return win_loss_ratio

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
                dd_possible = self.dealer.double_down_possible(player_cards)
                if self.dealer.has_black_jack(player_cards, player_count):
                    num_of_black_jack += 1

                if self.COUNT_CARDS:
                    state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}" \
                            f"_{self.dealer.deck_count}"
                else:
                    state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}"

                # Add state to Q-table
                if state not in self.q:
                    unexplored_states += 1
                    self.q[state] = [0, 0, 0]
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

                if self.dealer.has_black_jack(player_cards, player_count):
                    game_over = True
                    reward = 1.5
                else:
                    game_over = False
                while not game_over:
                    # Take action
                    reward, player_cards, player_count, dealer_cards, dealer_count, playable_ace_p, game_over = \
                        self.dealer.take_action(action, player_cards, dealer_cards)
                    player_count_tag = self.dealer.construct_count_tag(player_count)
                    dealer_count_tag = self.dealer.construct_count_tag(dealer_count)
                    dd_possible = self.dealer.double_down_possible(player_cards)
                    if self.COUNT_CARDS:
                        new_state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}" \
                                    f"_{self.dealer.deck_count}"
                    else:
                        new_state = f"{player_count_tag}_{playable_ace_p}_{dd_possible}_{dealer_count_tag}"

                    if game_over:
                        q_new_state = np.array([reward, reward, reward])
                    elif new_state not in self.q:
                        q_new_state = [0, 0, 0]
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

            # ----- Calculate performance metrics -----
            wins = [x for x in win_list if x >= 1]
            loose = [x for x in win_list if -2 <= x < 0]
            draw = [x for x in win_list if x == 0]
            black_jack = [x for x in win_list if x == 1.5]
            dd_win = [x for x in win_list if x == 2]
            dd_loose = [x for x in win_list if x == -2]
            num_valid_episodes = len([x for x in win_list if x != 10])

            r_wins = round((len(wins) / num_valid_episodes) * 1000) / 10
            r_draw = round((len(draw) / num_valid_episodes) * 1000) / 10
            r_loose = round((len(loose) / num_valid_episodes) * 1000) / 10
            r_black_jack = round((len(black_jack) / num_valid_episodes) * 1000) / 10
            r_dd = 0 if not len(dd_win) else round((len(dd_win) / (len(dd_win) + len(dd_loose))) * 1000) / 10

            # Calculate win-factor
            unique_outcomes = list(set(win_list))
            win_factor = 0
            for outcome_value in unique_outcomes:
                if outcome_value >= -2:
                    outcome = [x for x in win_list if x == outcome_value]
                    win_factor += len(outcome) * outcome_value
            win_factor = win_factor / num_valid_episodes

            # Print evaluation outcome
            print("Evaluate algorithm")
            print(f"Found {unexplored_states} new states")
            print(f"Number of wins: {len(wins)}, {r_wins}")
            print(f"Number of draw: {len(draw)} , {r_draw}")
            print(f"Number of lost games: {len(loose)} , {r_loose}")
            print(f"Number of black jacks : {len(black_jack)} , {r_black_jack}")
            print(f"Number of postive/negative double down: {len(dd_win)} / {len(dd_loose)} , {r_dd}")
            print(f"The win factor is: {win_factor}")
