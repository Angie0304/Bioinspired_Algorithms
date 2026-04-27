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
2. Each individual is evaluated using a fitness function based on row and column constraints, penalizing mismatches.  
3. Parents are selected using stochastic binary tournament selection.  
4. Offspring are generated using n-point or uniform crossover.  
5. Bit-flip mutation is applied with a fixed probability to maintain diversity.  
6. The population is updated iteratively over multiple generations.  
7. The algorithm stops when a valid solution is found or a stopping condition is reached.
