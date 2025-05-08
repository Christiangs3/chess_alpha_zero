from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import chess
from .mcts import Node, mcts, select_move_with_temperature
from .utils import move_to_index, board_to_tensor

def train_network(network, boards, policies, values, epochs=5, batch_size=32, output_dir=None):
    """
    Entrena la red neuronal
    """

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    if output_dir:
        checkpoint_path = os.path.join(output_dir, "best_model")
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,    # Guarda solo los mejores pesos
                save_weights_only=False, # Guarda todo el modelo, no solo los pesos
                verbose=1
            )
        )

    history = network.fit(
        boards,
        {'policy_output': policies, 'value_output': values},
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks
    )
    return history

def play_game_with_mcts(network, mcts_iterations=20, tau = 1.0):
    """
    Simula una partida utilizando MCTS.
    """
    board = chess.Board()
    states_with_policies = []
    result = None

    while not board.is_game_over():
        if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition() or board.is_insufficient_material() or board.is_stalemate() or board.is_checkmate():
            result = "1/2-1/2"
            break  # Termina si se alcanzan tablas

        if result is not None and result != "1/2-1/2":
          result = board.result()

        root = Node(board)
        mcts(root, network, mcts_iterations)

        # Ajusta tau en función del progreso del juego
        tau = 1.0 if board.fullmove_number < 10 else 0.0
        move = select_move_with_temperature(root, tau=tau)

        policy_probs = np.zeros(4672)
        for move_key, child_node in root.children.items():
            move_index = move_to_index(move_key)
            policy_probs[move_index] = child_node.visits
        policy_probs /= policy_probs.sum()

        states_with_policies.append((board.copy(), policy_probs))
        board.push(move)

        if result is None:
          result = board.result()

    return states_with_policies, result

def calculate_returns(states_with_policies, result):
    """
    Calcula la puntuación a partir de los resultados del juego.
    """
    returns = []
    result_value = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(result, 0)

    # Añadir factor de perspectiva del jugador
    current_player = 1  # 1 para blancas, -1 para negras
    gamma = 0.99
    n = len(states_with_policies)
    for i in range(n):
        discounted_return = 0
        for k in range(i, n):
            discounted_return += (gamma ** (k - i)) * result_value * current_player
        returns.append(discounted_return)
        current_player *= -1  # Alternar entre jugadores

    return returns

def generate_training_data_with_temperature(num_games, network, mcts_iterations=20, tau=1.0):
    """"
    Genera datos de entrenamiento utilizando MCTS con temperatura para regular la exploración/explotación.
    """
    games_data = []
    moves_per_game = []

    for game_num in range(num_games):
        states, result = play_game_with_mcts(network, mcts_iterations, tau)
        returns = calculate_returns(states, result)
        moves_per_game.append(len(states))

        print(f"Game {game_num + 1}: {len(states)} moves, Result: {result}")

        for i, (state, policy) in enumerate(states):
            board_tensor = board_to_tensor(state)
            games_data.append((board_tensor, policy, returns[i]))

    print(f"\nAverage moves per game: {sum(moves_per_game)/len(moves_per_game):.1f}")
    return games_data