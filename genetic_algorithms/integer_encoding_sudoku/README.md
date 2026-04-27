# Genetic Algorithm with Integer Encoding for Sudoku Solving

This project implements a Genetic Algorithm with integer encoding to solve Sudoku puzzles. Each candidate solution is represented as a grid of integers, and its fitness is evaluated based on how well it satisfies Sudoku constraints.

## Module Structure
```text
integer_encoding_sudoku/
├── README.md              # Documentation and usage guide
├── reporte_sudoku.pdf     # Report
├── requirements.txt       # Dependencies
└── sudoku.py              # Sudoku implementation
```
## How it works 

1. A population of Sudoku grids is initialized, preserving the fixed cells of the puzzle.  
2. Each individual is evaluated using a fitness function based on row, column, and subgrid conflicts.  
3. Parents are selected via stochastic binary tournament selection.  
4. Offspring are generated using crossover while maintaining valid row structures.  
5. Mutation is applied by swapping or modifying values within non-fixed cells.  
6. The population is updated over multiple generations.  
7. The algorithm stops when a valid solution or a stopping condition is reached.

## Usage 

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook
```bash
jupyter notebook sudoku.ipynb
```

## Output

The program will:

- Display the evolution of solutions across generations  
- Show the fitness value based on constraint violations  
- Output the best Sudoku grid found  
- Indicate whether a valid solution was reached

## Status
Complete
