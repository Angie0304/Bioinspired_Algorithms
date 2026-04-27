# Genetic Algorithm with Binary Encoding for Nonogram Solving

his project implements a Genetic Algorithm with binary encoding to solve Nonogram puzzles. Each candidate solution is represented as a binary grid, and its fitness is evaluated based on how well it satisfies row and column constraints.

## Module Structure

```text
binary_encoding_nonogram/
├── README.md
├── nonograma.ipynb
├── reporte_nonograma.pdf
└── requirements.txt
```

## How it works 
1. A random population of binary grids is generated as candidate solutions.
2. Each individual is evaluated using a fitness function based on row and column constraints.
3. The best individuals are selected to produce new solutions through crossover and mutation.
4. The population is updated and the process repeats for multiple generations.
5. The algorithm stops when a valid solution or a stopping condition is reached.
