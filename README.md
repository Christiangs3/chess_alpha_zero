# chess_alpha_zero
Master's Final Project

# MCTS Chess Bot - AlphaZero Inspired

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.14.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This project is a chess bot that uses an AlphaZero-inspired architecture, combining Monte Carlo Tree Search (MCTS) with a deep neural network for decision-making. It was developed as the final project for the Master's in Deep Learning at MIOTI | Tech & Business School and serves as a demonstration of skills in reinforcement learning, neural networks, and software development.

## Table of Contents
*   [Project Overview](#project-overview)
*   [Key Features](#key-features)
*   [Technologies Used](#technologies-used)
*   [Repository Structure](#repository-structure)
*   [Installation](#installation)
*   [Usage](#usage)
    *   [Model Training](#model-training)
    *   [Playing Against the Bot (Coming Soon)](#playing-against-the-bot-coming-soon)
*   [Neural Network Architecture](#neural-network-architecture)
*   [Results and Observations](#results-and-observations)
*   [Potential Future Improvements](#potential-future-improvements)
*   [License](#license)
*   [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to implement a chess agent capable of learning to play at a competent level through self-play, similar to DeepMind's AlphaZero approach.

The system consists of two main components:
1.  **Monte Carlo Tree Search (MCTS):** A search algorithm that explores the tree of possible moves, guided by the neural network's predictions.
2.  **Deep Neural Network:** A neural network with residual blocks that takes the current board state as input and produces two outputs:
    *   **Policy:** A probability distribution over possible moves.
    *   **Value:** An estimate of the probability of winning from the current position.

The agent learns by playing games against itself. Data from these games (states, MCTS policies, and game outcomes) are used to train the neural network, iteratively improving its prediction capabilities and, consequently, the MCTS playing quality.

## Key Features

*   **MCTS with PUCT:** Implementation of the MCTS algorithm using the Upper Confidence Bound for Trees (PUCT) formula to balance exploration and exploitation.
*   **Residual Neural Network:** Neural network architecture with 10 residual blocks for processing the board representation.
*   **Policy and Value Outputs:** The network predicts both the probability of moves and the position's value.
*   **Board Representation:** The board state is represented as an 8x8x12 tensor, encoding the position of each piece type for both colors.
*   **Self-Play Data Generation:** The bot plays against itself to generate training data.
*   **Iterative Training:** The model is trained in cycles, alternating between game data generation and adjusting the neural network weights.
*   **Temperature Handling (Tau):** Parameter to control the exploration level in the early stages of the game during data generation.
*   **Integration with `python-chess`:** Uses the `python-chess` library for chess game logic, move validation, and board state management.

## Technologies Used

*   **Python 3.10+**
*   **TensorFlow 2.14.0:** For implementing and training the neural network.
*   **`python-chess` 1.11.1:** For chess game logic.
*   **NumPy:** For numerical operations and tensor manipulation.
*   **Google Colab:** Used for initial training and notebook execution (code is being refactored into Python modules).
*   **Google Drive:** Used for storing training data and models during development in Colab.

## Repository Structure
Use code with caution.
Markdown
chess_alpha_zero/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── notebooks/
│ └── MCTS_bot_final.ipynb # Original notebook with development and testing
├── src/
│ ├── chess_bot/
│ │ ├── init.py
│ │ ├── mcts.py # MCTS logic
│ │ ├── network.py # Neural Network definition
│ │ ├── train.py # Main training loop and data generation
│ │ └── utils.py # Utility functions (board handling, moves, data)
├── data/
│ ├── training_data/ # For generated training data (.npz)
│ │ └── .gitkeep
│ └── models/ # For saved models
## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Christiangs3/chess_alpha_zero.git
    cd chess_alpha_zero
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Google Drive Setup:**
    If you wish to use the functionality to save/load training data or models from Google Drive (as shown in the notebook), ensure you have access configured and modify the paths in `src/chess_bot/utils.py` or in the training scripts as needed. By default, training and data saving will be done locally in the `data/` directory.

## Usage

### Model Training

Training is performed using the provided notebook `MCTS_bot_final.ipynb` (or eventually a dedicated script, once refactored). The process iterates through self-play cycles to generate data and then uses that data to train the neural network.

Key parameters (within the code/notebook):
*   `output_dir`: Directory to save training data and models.
*   `epochs`: Number of epochs to train the network in each cycle.
*   `batch_size`: Batch size for network training.
*   `mcts_iterations`: Number of MCTS simulations per move during data generation.

Training progress, including model loss and generated data files, will be displayed in the notebook output and saved in the `output_dir`.

## Playing Against the Bot (Coming Soon)

Currently, the functionality to play interactively against a trained model is not implemented as a standalone script. It would require loading a saved model and creating an interface (simple CLI or GUI) for a human player to input moves.

## Neural Network Architecture

The neural network consists of:

*   **Input Layer:** Accepts an 8x8x12 tensor representing the board state.
*   **Initial Convolutional Layer:** With 256 filters (3x3) and ReLU activation, followed by Batch Normalization.
*   **Residual Blocks:** 10 residual blocks, each composed of two convolutional layers (256 filters 3x3, ReLU, Batch Norm) with a skip connection.
*   **Flatten Layer.**
*   **Dense Layers:** Several dense layers with ReLU activation, Batch Normalization, and Dropout for regularization.
*   **Policy Head:** Dense layer with 4672 neurons (corresponding to all possible encoded moves) and Softmax activation.
*   **Value Head:** Dense layer with 1 neuron and Tanh activation (range -1 to 1).

## Results and Observations

The `MCTS_bot_final.ipynb` notebook shows the output from the data generation and training process. It is observed that many games end in draws ("1/2-1/2"), which is common in the early stages of training self-play chess agents before they develop more sophisticated strategies to convert advantages.

These types of architectures require very high computational capacity and hundreds of thousands of games to begin showing understanding of basic tactics and piece development. This level could not be reached within the scope of this project due to resource limitations.

## Potential Future Improvements

*   **Hyperparameter Optimization:** Perform a more exhaustive search for optimal hyperparameters for MCTS and the neural network.
*   **Increased Computational Resources:** Train for a longer time and with more MCTS iterations per move.
*   **More Advanced Network Architecture:** Experiment with larger or different network architectures (e.g., Squeeze-and-Excitation blocks).
*   **Distributed Training:** Adapt the code for distributed training and parallel data generation.
*   **Formal Evaluation:** Implement a system to evaluate the bot's strength (estimated ELO) by playing against other chess engines (e.g., Stockfish at different difficulty levels).
*   **User Interface:** Develop a simple GUI to play against the bot more user-friendly.
*   **MCTS Enhancements:** Explore MCTS variants or more advanced pruning techniques.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

*   The original **AlphaZero paper** by DeepMind for the inspiration and fundamental architecture.
*   **MIOTI | Tech & Business School.**

---
*This README is a draft. Feel free to contact me if you have questions or suggestions.*
