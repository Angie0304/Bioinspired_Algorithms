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

1. A random population of binary grids is generated.  
2. Each individual is evaluated using a constraint-based fitness function.  
3. Parents are selected via stochastic binary tournament selection.  
4. Offspring are generated using n-point or uniform crossover.  
5. Bit-flip mutation is applied to maintain diversity in the population.  
6. The population is updated over multiple generations.  
7. The algorithm stops when a solution or a stopping condition is reached.

## Usage 

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook
```bash
jupyter notebook nonograma.ipynb
```

3. Output
- Display intermediate and final Nonogram solutions
- Show the fitness value of individuals across generations
- Indicate the best solution found
- Visualize the final grid configuration

## Status 
Complete

