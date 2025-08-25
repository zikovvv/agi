from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from collections import deque


@dataclass
class GameState:
    """
    - bombs:   (H, W) uint8 {0,1}
    - numbers: (H, W) uint8 0..8
    - visible: (H, W) bool
    - lost:    bool  (True if a bomb was revealed)
    - won:     bool  (True if all non-bomb cells are visible)
    """
    bombs: np.ndarray
    numbers: np.ndarray
    visible: np.ndarray
    lost: bool
    won: bool


def _validate_dims(height: int, width: int, n_bombs: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if n_bombs < 0 or n_bombs >= height * width:
        raise ValueError("n_bombs must be in [0, height*width - 1]")


def _place_bombs(height: int, width: int, n_bombs: int,
                 safe_cell: Tuple[int, int],
                 rng: np.random.Generator) -> np.ndarray:
    """
    Place bombs with 'safe_cell' guaranteed bomb-free.
    Returns (H, W) uint8 array of {0,1}.
    """
    _validate_dims(height, width, n_bombs)
    total = height * width
    safe_index = safe_cell[0] * width + safe_cell[1]

    # All indices except the safe one
    candidates = np.delete(np.arange(total, dtype=np.int64), safe_index)
    if n_bombs > candidates.size:
        raise ValueError("Too many bombs given the guaranteed-safe first click.")

    bomb_idxs = rng.choice(candidates, size=n_bombs, replace=False)
    bombs = np.zeros(total, dtype=np.uint8)
    bombs[bomb_idxs] = 1
    return bombs.reshape(height, width)


def _compute_numbers(bombs: np.ndarray) -> np.ndarray:
    """
    Adjacent mine counts via a 3x3 sum (NumPy-only 'convolution').
    """
    H, W = bombs.shape
    P = np.pad(bombs, 1, mode="constant", constant_values=0)
    s = (
        P[0:H,   0:W]   + P[0:H,   1:W+1] + P[0:H,   2:W+2] +
        P[1:H+1, 0:W]   + P[1:H+1, 1:W+1] + P[1:H+1, 2:W+2] +
        P[2:H+2, 0:W]   + P[2:H+2, 1:W+1] + P[2:H+2, 2:W+2]
    )
    numbers = (s - bombs).astype(np.uint8)
    return numbers


def _neighbors(r: int, c: int, H: int, W: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc


def _flood_reveal(numbers: np.ndarray, visible: np.ndarray,
                  start: Tuple[int, int]) -> None:
    """
    Classic Minesweeper reveal:
    - Reveal 'start'. If it's 0, BFS to reveal connected zeros and their bordering numbers.
    - If it's >0, reveal only that cell.
    """
    H, W = numbers.shape
    r0, c0 = start
    if visible[r0, c0]:
        return
    visible[r0, c0] = True
    if numbers[r0, c0] != 0:
        return

    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors(r, c, H, W):
            if visible[rr, cc]:
                continue
            visible[rr, cc] = True
            if numbers[rr, cc] == 0:
                q.append((rr, cc))


def _compute_won(visible: np.ndarray, bombs: np.ndarray) -> bool:
    return np.all(visible | (bombs.astype(bool)))


def new_game(height: int, width: int, n_bombs: int,
             first_click: Optional[Tuple[int, int]] = None,
             seed: Optional[int] = None) -> GameState:
    """
    Create a new game and perform the guaranteed-safe initial click with flood reveal.
    """
    _validate_dims(height, width, n_bombs)
    rng = np.random.default_rng(seed)

    if first_click is None:
        first_click = (int(rng.integers(0, height)), int(rng.integers(0, width)))

    bombs = _place_bombs(height, width, n_bombs, first_click, rng).astype(np.uint8)
    numbers = _compute_numbers(bombs)
    visible = np.zeros((height, width), dtype=bool)

    _flood_reveal(numbers, visible, first_click)

    lost = False
    won = _compute_won(visible, bombs)
    return GameState(bombs=bombs, numbers=numbers, visible=visible, lost=lost, won=won)


def snapshot(state: GameState) -> GameState:
    return GameState(
        bombs=state.bombs.copy(),
        numbers=state.numbers.copy(),
        visible=state.visible.copy(),
        lost=bool(state.lost),
        won=bool(state.won),
    )


def apply_action(numbers: np.ndarray,
                 bombs: np.ndarray,
                 visible: np.ndarray,
                 click: Tuple[int, int]) -> GameState:
    """
    Transition: (numbers, bombs, visible) + click -> new GameState.
    """
    H, W = numbers.shape
    r, c = click
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError("Click out of bounds")

    visible2 = visible.copy()
    lost = False
    if bombs[r, c] == 1:
        visible2[r, c] = True
        lost = True
    else:
        _flood_reveal(numbers, visible2, (r, c))

    won = _compute_won(visible2, bombs)
    return GameState(bombs=bombs, numbers=numbers, visible=visible2, lost=lost, won=won)


def _adjacent_to_visible_safe_candidates(visible: np.ndarray,
                                         bombs: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of safe & currently hidden cells that are
    8-neighbors of ANY already-visible safe cell.
    Vectorized (NumPy-only) adjacency check.
    """
    H, W = visible.shape
    visible_safe = visible & (bombs == 0)

    P = np.pad(visible_safe.astype(np.uint8), 1, mode="constant", constant_values=0)
    # Sum 3x3 neighborhood; subtract the center to exclude self
    s = (
        P[0:H,   0:W]   + P[0:H,   1:W+1] + P[0:H,   2:W+2] +
        P[1:H+1, 0:W]   + P[1:H+1, 1:W+1] + P[1:H+1, 2:W+2] +
        P[2:H+2, 0:W]   + P[2:H+2, 1:W+1] + P[2:H+2, 2:W+2]
    )
    adjacency = (s - visible_safe.astype(np.uint8)) > 0

    safe_hidden = (bombs == 0) & (~visible)
    return adjacency & safe_hidden


def generate_episode(height: int, width: int, n_bombs: int,
                     seed: Optional[int] = None) -> List[GameState]:
    """
    Self-play trajectory for dataset creation using a FRONTIER-FIRST policy:
    1) Start new game with safe first click (flood reveal).
    2) While not won:
       - Prefer a safe hidden cell that is adjacent to any already-visible safe cell.
       - If none exist, pick a random remaining safe hidden cell.
       - Apply action; append snapshot.
    Note: we 'peek' bombs for supervised data synthesis (so we never click a bomb).
    """
    rng = np.random.default_rng(seed)
    state = new_game(height, width, n_bombs, first_click=None, seed=seed)
    history: List[GameState] = [snapshot(state)]

    H, W = state.bombs.shape

    while not state.won and not state.lost:
        # Frontier-preferred candidates (adjacent to a visible safe cell)
        frontier_mask = _adjacent_to_visible_safe_candidates(state.visible, state.bombs)
        candidates = np.argwhere(frontier_mask)

        if candidates.size == 0:
            # Fallback: any remaining safe & hidden cell
            fallback_mask = (state.bombs == 0) & (~state.visible)
            candidates = np.argwhere(fallback_mask)
            if candidates.size == 0:
                break  # nothing left to reveal

        # Pick one candidate at random for variety
        idx = int(rng.integers(0, candidates.shape[0]))
        r, c = map(int, candidates[idx])

        state = apply_action(state.numbers, state.bombs, state.visible, (r, c))
        history.append(snapshot(state))

    return history


def step_with_history(state: GameState,
                      click: Tuple[int, int],
                      history: Optional[List[GameState]] = None) -> Tuple[GameState, List[GameState]]:
    """
    Apply a click, append snapshot to history, and return (new_state, history).
    """
    new_state = apply_action(state.numbers, state.bombs, state.visible, click)
    if history is None:
        history = []
    history.append(snapshot(new_state))
    return new_state, history


# -----------------------------
# Minimal example / smoke test:
# -----------------------------
if __name__ == "__main__":
    H, W, B = 5, 5, 4
    episode = generate_episode(H, W, B, seed=123)
    last = episode[-1]
    print(f"Episode length (states): {len(episode)}")
    print(f"Lost: {last.lost}, Won: {last.won}")
    print("bombs:", episode[0].bombs.shape, episode[0].bombs.dtype)
    print("numbers:", episode[0].numbers.shape, episode[0].numbers.dtype)
    print("visible:", episode[0].visible.shape, episode[0].visible.dtype)

    for j, state in enumerate(episode):
        print(f"State {j}:")
        print(state.bombs)
        print(state.numbers)
        print(state.visible)

        print('=======================')
        print('=======================')
        print('=======================')
