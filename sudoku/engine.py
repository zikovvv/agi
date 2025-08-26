from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
NB_CATEGORIES = 10


# ============================================================
# Sudoku dataset primitives (9x9 classic)
# - puzzle: numbers with 0 for blanks (for human-readable datasets)
# - solution: full solved grid (1..9)
# - given_mask: True where puzzle has a clue (nonzero)
# - x_onehot: categorical input features (one-hot over 1..9, blanks -> all zeros)
# - y_onehot: categorical targets      (one-hot over 1..9 for the true digit)
# ============================================================

@dataclass
class SudokuSample:
    puzzle: np.ndarray        # (9, 9) uint8 in {0..9}
    solution: np.ndarray      # (9, 9) uint8 in {1..9}
    given_mask: np.ndarray    # (9, 9) bool
    # x_onehot: np.ndarray      # (81, NB_CATEGORIES) uint8 in {0,1}
    # y_onehot: np.ndarray      # (81, NB_CATEGORIES) uint8 in {0,1}


# -----------------------------
# Validation helpers
# -----------------------------

def _validate_missing(n_missing: int) -> None:
    if n_missing < 0 or n_missing > 81:
        raise ValueError("n_missing must be in [0, 81]")


# -----------------------------
# Complete grid generation
# -----------------------------
# We use the classic base-pattern approach for generating a valid, complete
# 9x9 solution, then randomize by shuffling bands/stacks/rows/cols and relabeling digits.

def _complete_solution(seed: Optional[int] = None) -> np.ndarray:
    """
    Returns a valid solved Sudoku grid (9x9, values 1..9) via pattern + shuffles.
    """
    rng = np.random.default_rng(seed)
    base = 3
    side = base * base  # 9

    def pattern(r: int, c: int) -> int:
        # Base Latin-style pattern that respects 3x3 boxes:
        # (base*(r%base) + r//base + c) % side
        return (base * (r % base) + r // base + c) % side

    # Permute rows/cols within bands/stacks and permute bands/stacks themselves
    def shuffle(s):
        return rng.permutation(s)

    r_base = np.arange(base)
    rows = np.concatenate([base * rb + shuffle(r_base) for rb in shuffle(r_base)])
    cols = np.concatenate([base * cb + shuffle(r_base) for cb in shuffle(r_base)])
    nums = shuffle(np.arange(1, side + 1))

    # Build the solution using the pattern
    grid = np.empty((side, side), dtype=np.uint8)
    for r_i, r in enumerate(rows):
        for c_i, c in enumerate(cols):
            grid[r_i, c_i] = nums[pattern(r, c)]
    return grid


# -----------------------------
# Puzzle masking with exact blanks
# -----------------------------

def _mask_with_exact_blanks(solution: np.ndarray,
                            n_missing: int,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a puzzle by zeroing exactly n_missing positions.
    Returns (puzzle, given_mask).
    - puzzle: (9,9) uint8 with 0 at blanks
    - given_mask: (9,9) bool, True where puzzle has a clue
    """
    _validate_missing(n_missing)
    if solution.shape != (9, 9):
        raise ValueError("solution must be (9,9)")
    total = 81
    if n_missing > total:
        raise ValueError("n_missing cannot exceed 81")

    puzzle = solution.copy()
    all_idxs = np.arange(total, dtype=np.int64)
    # Choose exactly n_missing unique cells to blank
    blank_linear = rng.choice(all_idxs, size=n_missing, replace=False)
    rr = blank_linear // 9
    cc = blank_linear % 9
    puzzle[rr, cc] = 0

    given_mask = puzzle != 0
    return puzzle.astype(np.uint8), given_mask

# -----------------------------
# Single-sample creation
# -----------------------------

def new_sample(n_missing: int,
               seed: Optional[int] = None) -> SudokuSample:
    """
    Create a single Sudoku sample with exactly n_missing blanks.
    - puzzle: zeros for blanks
    - solution: complete grid
    # - x_onehot: one-hot of puzzle (blanks -> all zeros)
    # - y_onehot: one-hot of solution
    """
    _validate_missing(n_missing)
    rng = np.random.default_rng(seed)

    solution = _complete_solution(seed=rng.integers(0, 2**63 - 1))
    puzzle, given_mask = _mask_with_exact_blanks(solution, n_missing, rng)

    # x_onehot = grid_to_one_hot(puzzle)
    # y_onehot = solution_to_one_hot(solution)
    return SudokuSample(
        puzzle=puzzle, solution=solution, given_mask=given_mask,
        # x_onehot=x_onehot, y_onehot=y_onehot
    )


# -----------------------------
# Batch/dataset generation
# -----------------------------
@dataclass
class SudokuDataset:
    puzzles: np.ndarray   # (N, 9, 9) uint8
    solutions: np.ndarray # (N, 9, 9) uint8
    # x_onehot: np.ndarray  # (N, 81, NB_CATEGORIES) uint8
    # y_onehot: np.ndarray  # (N, 81, NB_CATEGORIES) uint8
    n_missing: int


def generate_dataset(n_samples: int,
                     n_missing: int,
                     n_missing_max : Optional[int] = None,
                     seed: Optional[int] = None) -> SudokuDataset:
    """
    Generate a dataset of size N with exactly n_missing blanks per puzzle.
    Shapes:
        puzzles:   (N, 9, 9) uint8   [0..9] (zeros are blanks)
        solutions: (N, 9, 9) uint8   [1..9]
        x_onehot:  (N, 81, NB_CATEGORIES) uint8  [0/1]  (input features)
        y_onehot:  (N, 81, NB_CATEGORIES) uint8  [0/1]  (targets)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    _validate_missing(n_missing)

    rng = np.random.default_rng(seed)

    puzzles = np.empty((n_samples, 9, 9), dtype=np.uint8)
    solutions = np.empty((n_samples, 9, 9), dtype=np.uint8)
    # x_onehot = np.empty((n_samples, 81, NB_CATEGORIES), dtype=np.uint8)
    # y_onehot = np.empty((n_samples, 81, NB_CATEGORIES), dtype=np.uint8)

    for i in range(n_samples):
        # Derive a per-sample seed for reproducibility with a single master seed
        sample_seed = int(rng.integers(0, 2**63 - 1))
        nmis = n_missing
        if n_missing_max is not None and n_missing_max > n_missing:
            nmis = rng.integers(n_missing, n_missing_max + 1)
        s = new_sample(nmis, seed=sample_seed)
        puzzles[i] = s.puzzle
        solutions[i] = s.solution
        # x_onehot[i] = s.x_onehot
        # y_onehot[i] = s.y_onehot

    return SudokuDataset(
        puzzles=puzzles,
        solutions=solutions,
        # x_onehot=x_onehot,
        # y_onehot=y_onehot,
        n_missing=n_missing,
    )


# -----------------------------
# Minimal example / smoke test
# -----------------------------
if __name__ == "__main__":
    N = 100
    MISSING = 50
    ds = generate_dataset(N, MISSING, seed=123)

    print("Dataset shapes:")
    print("  puzzles:", ds.puzzles.shape, ds.puzzles.dtype)
    print("  solutions:", ds.solutions.shape, ds.solutions.dtype)
    # print("  x_onehot:", ds.x_onehot.shape, ds.x_onehot.dtype)
    # print("  y_onehot:", ds.y_onehot.shape, ds.y_onehot.dtype)

    # Verify exact number of blanks per puzzle
    for i in range(N):
        blanks = int((ds.puzzles[i] == 0).sum())
        print(f"Sample {i}: blanks={blanks} (target={MISSING})")

    k = 3
    print("\nPuzzle (zeros = blanks):\n", ds.puzzles[k])
    print("\nSolution:\n", ds.solutions[k])

    nb_missing_each = []
    for i in range(N):
        nb_missing = int((ds.puzzles[i] == 0).sum())
        nb_missing_each.append(nb_missing)
    
    nb_missing_min, nb_missing_max = min(nb_missing_each), max(nb_missing_each)
    assert nb_missing_min == nb_missing_max == MISSING
    


    ds_random_missing = generate_dataset(N, MISSING, MISSING + 10, seed=123)

    nb_missing_each = []
    for i in range(N):
        nb_missing = int((ds_random_missing.puzzles[i] == 0).sum())
        nb_missing_each.append(nb_missing)

    nb_missing_min, nb_missing_max = min(nb_missing_each), max(nb_missing_each)
    assert nb_missing_min >= MISSING
    assert nb_missing_max <= MISSING + 10
    print(f"\nWith random missing in [{MISSING}, {MISSING + 10}]: min={nb_missing_min}, max={nb_missing_max}")
    

