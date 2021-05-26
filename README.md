# Black-jack reinforcement-learning

The Black-Jack reinforcement learning project solves the game of black-jack training an epsilon greedy Q-learning agent to learn an optimal policy of the game. It proves that the Nash equilibrium of the game lies in the favor of the dealer and shows that a policy exist that gains a statistical advantage over the dealer if the ability of counting cards is introduced to the agent.

![rendered_network](/assets/q_surface.gif)

The animation above shows the resulting Q-surface post the training session for different deck-counts ranging from -20 to 20. Below is an animation showing the forming of the q-surface during the training of the Q-agent.

![rendered_network](/assets/q_surface_training.gif)


# Possible actions
- **Hit** - Draw another card from the stack
- **Stay** - Stay and let dealer play
- **Double-down** - Double the bet and take one, and only one additional card. If the game is a win, the reward will be double. If the game results in a loss, the loss will be double. This action is only allowed if the player has not taken any previous actions.
- **Split**  - Split the hand in two and create one new game. The action is only allowed if no previous actions are made and the two cards on hand are equal.

