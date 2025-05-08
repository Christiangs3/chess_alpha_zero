def board_to_tensor(board):
    """
    Convierte un tablero de ajedrez (chess.Board) a una representación tensorial 8x8x12.
    """
    tensor = np.zeros((8, 8, 12), dtype=int)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = piece_map[piece.piece_type]
        if piece.color == chess.BLACK:
            channel += 6  # Canales 6-11 para piezas negras
        tensor[row, col, channel] = 1

    return tensor

def move_to_index(move):
    """
    Convierte un movimiento de ajedrez en notación UCI a un índice entero.
    Utiliza una codificación de posiciones desde (0, 0) hasta (7, 7) y 73 tipos de movimientos.
    """
    uci_move = move.uci()  # Convertir a notación UCI
    from_square = chess.SQUARE_NAMES.index(uci_move[:2])  # Índice de casilla de origen
    to_square = chess.SQUARE_NAMES.index(uci_move[2:4])  # Índice de casilla de destino

    if len(uci_move) == 5:  # Movimientos con promoción
        promotion_piece = uci_move[4]
        promotion_index = {'q': 0, 'r': 1, 'b': 2, 'n': 3}[promotion_piece]
        return from_square * 64 + to_square * 4 + promotion_index
    else:
        return from_square * 64 + to_square

def index_to_move(index, board):
    """
    Convierte un índice a un movimiento de ajedrez usando el estado del tablero.
    """
    from_square = index // 64
    to_square = (index % 64) // 4
    promotion_piece = index % 4
    promotion_dict = {0: 'q', 1: 'r', 2: 'b', 3: 'n'}

    if promotion_piece in promotion_dict:
        return chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(promotion_dict[promotion_piece]))
    return chess.Move(from_square, to_square)

def save_training_data_to_drive(training_data, output_dir, file_counter):
    """
    Guarda los datos de entrenamiento en Google Drive.
    """
    os.makedirs(output_dir, exist_ok=True)

    if training_data:
        boards, policies, values = zip(*training_data)
        boards = np.array(boards)
        policies = np.array(policies)
        values = np.array(values)
        file_path = os.path.join(output_dir, f'training_batch_{file_counter}.npz')
        np.savez_compressed(file_path, boards=boards, policies=policies, values=values)
        print(f"Datos de entrenamiento guardados en Google Drive en el archivo: {file_path}!")
    else:
        print("No hay datos de entrenamiento para guardar.")


def load_training_data_from_drive(data_dir):
    """
    Carga los datos de entrenamiento desde Google Drive.
    """
    boards_list = []
    policies_list = []
    values_list = []

    for filename in os.listdir(data_dir):
        if filename.startswith("training_batch_") and filename.endswith(".npz"):
            file_number = filename.split("_")[2].split(".")[0]
            try:
               file_number = int(file_number)
            except ValueError:
                continue

            data = np.load(os.path.join(data_dir, filename))
            boards_list.append((file_number,data["boards"]))
            policies_list.append((file_number,data["policies"]))
            values_list.append((file_number,data["values"]))

    boards_list.sort(key=lambda x: x[0])
    policies_list.sort(key=lambda x: x[0])
    values_list.sort(key=lambda x: x[0])

    boards = np.concatenate([x[1] for x in boards_list], axis=0) if boards_list else np.array([])
    policies = np.concatenate([x[1] for x in policies_list], axis=0) if policies_list else np.array([])
    values = np.concatenate([x[1] for x in values_list], axis=0) if values_list else np.array([])


    print(f'Boards data shape: {boards.shape}')
    print(f'Policies data shape: {policies.shape}')
    print(f'Values data shape: {values.shape}')
    return boards, policies, values

def save_model(model, model_path):
    """
    Guarda el modelo en el sistema de archivos.
    """
    model.save(model_path)
    print(f"Model saved at: {model_path}")

def load_model(model_path):
    """
    Carga el modelo desde el sistema de archivos.
    """
    if os.path.exists(model_path):
       model = models.load_model(model_path)
       print(f"Model loaded from: {model_path}")
       return model
    print(f"No model found at: {model_path}, new model created")
    return create_chess_network()

def data_generation_loop(num_games_per_batch, mcts_iterations, output_dir, file_counter, chess_network, tau):
    """"
    Bucle de generación de datos de entrenamiento.
    """
    training_data = generate_training_data_with_temperature(num_games_per_batch, chess_network, mcts_iterations, tau)
    print(f"Se han generado {len(training_data)} registros de entrenamiento.")

    save_training_data_to_drive(training_data, output_dir, file_counter)


def save_day_counter(day, counter_file):
    """
    Guarda el día actual en el archivo de conteo.
    """
    with open(counter_file, "w") as f:
        f.write(str(day))

def load_day_counter(counter_file):
    """
    Carga el día actual desde el archivo de conteo.
    """
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            return int(f.read())
    return 0


def main_training_loop(output_dir, epochs, batch_size, mcts_iterations):
    """
    Bucle principal de entrenamiento.
    """
    model_path = os.path.join(output_dir, "chess_model")
    chess_network = load_model(model_path)

    counter_file = os.path.join(output_dir, "day_counter.txt")
    day = load_day_counter(counter_file)

    if day == 0:
       print("Initial training data generation and training")
       num_games = 50
       file_counter = 0
       tau = 1.0
       data_generation_loop(num_games, mcts_iterations, output_dir, file_counter, chess_network, tau)
       try:
            boards, policies, values = load_training_data_from_drive(output_dir)

            if len(boards) == 0:
                print("No hay datos de entrenamiento disponibles")
                return None

            history = train_network(chess_network, boards, policies, values, epochs, batch_size, output_dir)
            print(f"Initial training completed")
            save_model(chess_network, model_path)

       except (FileNotFoundError, ValueError) as e:
          print(f"Error during training: {str(e)}")
          return None

    for current_day in range(day, 5):
        print(f"Starting training day: {current_day+1}")
        file_counter = current_day + 1
        num_games = 100
        tau = max(0.1, 1.0 - (current_day / 2.0))
        data_generation_loop(num_games, mcts_iterations, output_dir, file_counter, chess_network, tau)

        try:
            boards, policies, values = load_training_data_from_drive(output_dir)

            if len(boards) == 0:
                print("No hay datos de entrenamiento disponibles")
                continue

            history = train_network(chess_network, boards, policies, values, epochs, batch_size, output_dir)
            print(f"Training completed for day: {current_day+1}")
            save_model(chess_network, model_path)


        except (FileNotFoundError, ValueError) as e:
            print(f"Error during training on day {current_day+1}: {str(e)}")
        save_day_counter(current_day + 1, counter_file)

    return chess_network