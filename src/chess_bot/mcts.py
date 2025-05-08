import numpy as np
from .utils import board_to_tensor, move_to_index

class Node:
    def __init__(self, state, parent=None, prior_prob=1.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior_prob = prior_prob

    def expand(self, legal_moves, policy_probs):
        """
        Expande el nodo con los movimientos legales y probabilidades de política.
        """
        for move, prob in zip(legal_moves, policy_probs):
            if move not in self.children:
                next_state = self.state.copy()  # Crear un nuevo estado a partir del movimiento
                next_state.push(move)
                self.children[move] = Node(next_state, parent=self, prior_prob=prob)

    def value(self):
        """
        Calcula el valor actual del nodo.
        """
        return self.value_sum / max(1, self.visits)

    def select_child(self, c_puct=0.7):
     """
     Selecciona el mejor hijo usando PUCT.
     """
     selected_child = max(
         self.children.items(),
         key=lambda item: item[1].value() + c_puct * item[1].prior_prob * (self.visits ** 0.5) / (1 + item[1].visits)
     )
     #print (f"selected move: {selected_child[0]}, visits: {selected_child[1].visits}")
     return selected_child
    
def predict_policy_and_value(network, board):
    """
    Convierte el tablero a tensor y utiliza la red neuronal para predecir política y valor.
    """
    tensor_input = board_to_tensor(board)
    policy, value = network.predict(tensor_input[None, :, :, :])
    return policy[0], value[0][0]

def mcts(root, network, iterations=20, dirichlet_alpha=0.1):
    for _ in range(iterations):
        node = root
        while node.children:
          _, node = node.select_child()

        if not node.state.is_game_over():
            legal_moves = list(node.state.legal_moves)
            policy_probs, value = predict_policy_and_value(network, node.state)

            legal_move_probs = np.zeros(len(legal_moves))
            for i, move in enumerate(legal_moves):
                move_index = move_to_index(move)
                legal_move_probs[i] = policy_probs[move_index]
            legal_move_probs /= np.sum(legal_move_probs) + 1e-8  # Normalizar y evitar división por cero

          # Añade 'Dirichlet noise' al root node para asegurar que se juegan todos los movimientos
            if node == root:
                noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
                legal_move_indexes = [move_to_index(move) for move in legal_moves]
                policy_probs[legal_move_indexes] = policy_probs[legal_move_indexes] + 0.25 * noise

                node.expand(legal_moves, policy_probs)
        else:
            value = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[node.state.result()]

      # Retropropagación del valor
        while node:
            node.visits += 1
            node.value_sum += value if node.parent else -value  # Propagación opuesta para minimizar/maximizar
            node = node.parent

def select_move_with_temperature(node, tau=1.0):
    if tau == 0:
        return max(node.children.items(), key=lambda item: item[1].visits)[0]

    visits = np.array([child.visits for child in node.children.values()])
    probabilities = np.exp(visits / tau) / np.sum(np.exp(visits / tau))
    moves = list(node.children.keys())
    return np.random.choice(moves, p=probabilities)