"""Maze generation and solving algorithms (BFS, DFS, Backtracking).

Grid convention: grid[r][c] == 1 is a wall, 0 is open.
Every solver returns a Result with the path, the order cells were
visited in, and the run time.
"""
import random
import time
from collections import deque
from dataclasses import dataclass, field

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


@dataclass
class Result:
    name: str
    path: list = field(default_factory=list)      # start -> end cells
    visited: list = field(default_factory=list)   # cells in exploration order
    time_ms: float = 0.0
    found: bool = False
    capped: bool = False                          # stopped by step limit


def generate_maze(rows, cols, loops=0.0, seed=None):
    """Perfect maze via randomized DFS (recursive backtracker).

    rows, cols are forced odd. `loops` (0..1) knocks out that fraction of
    interior walls, creating multiple routes so the algorithms differ.
    """
    rng = random.Random(seed)
    rows += 1 - rows % 2
    cols += 1 - cols % 2
    grid = [[1] * cols for _ in range(rows)]
    grid[1][1] = 0
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        nbrs = [(r + 2 * dr, c + 2 * dc, dr, dc) for dr, dc in DIRS
                if 0 < r + 2 * dr < rows - 1 and 0 < c + 2 * dc < cols - 1
                and grid[r + 2 * dr][c + 2 * dc] == 1]
        if nbrs:
            nr, nc, dr, dc = rng.choice(nbrs)
            grid[r + dr][c + dc] = 0
            grid[nr][nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()
    if loops > 0:
        walls = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)
                 if grid[r][c] == 1 and
                 ((grid[r - 1][c] == 0 and grid[r + 1][c] == 0) or
                  (grid[r][c - 1] == 0 and grid[r][c + 1] == 0))]
        rng.shuffle(walls)
        for r, c in walls[:int(len(walls) * loops)]:
            grid[r][c] = 0
    return grid, (1, 1), (rows - 2, cols - 2)


def parse_maze(text):
    """Parse a text maze: '#' wall, ' ' or '.' open, 'S' start, 'E' end."""
    lines = [l for l in text.splitlines() if l.strip("\n")]
    width = max(len(l) for l in lines)
    grid, start, end = [], None, None
    for r, line in enumerate(lines):
        row = []
        for c, ch in enumerate(line.ljust(width, "#")):
            if ch == "S":
                start = (r, c)
            elif ch == "E":
                end = (r, c)
            row.append(1 if ch == "#" else 0)
        grid.append(row)
    if start is None or end is None:
        raise ValueError("Maze needs one S (start) and one E (end).")
    return grid, start, end


def _neighbors(grid, r, c):
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
            yield nr, nc


def _build_path(parent, end):
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def solve_bfs(grid, start, end):
    t0 = time.perf_counter()
    parent = {start: None}
    visited = []
    q = deque([start])
    found = False
    while q:
        cur = q.popleft()
        visited.append(cur)
        if cur == end:
            found = True
            break
        for nb in _neighbors(grid, *cur):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    ms = (time.perf_counter() - t0) * 1000
    return Result("BFS", _build_path(parent, end) if found else [], visited, ms, found)


def solve_dfs(grid, start, end):
    """Iterative DFS with a global visited set (each cell expanded once)."""
    t0 = time.perf_counter()
    parent = {start: None}
    seen = set()
    visited = []
    stack = [start]
    found = False
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        visited.append(cur)
        if cur == end:
            found = True
            break
        for nb in _neighbors(grid, *cur):
            if nb not in seen:
                parent[nb] = cur  # latest discoverer wins, which is fine for DFS
                stack.append(nb)
    ms = (time.perf_counter() - t0) * 1000
    return Result("DFS", _build_path(parent, end) if found else [], visited, ms, found)


def solve_backtracking(grid, start, end, step_limit=200_000):
    """Classic backtracking: extend the path, undo (unmark) on a dead end.

    Only cells on the CURRENT path are marked, so a cell can be re-entered
    from a different route after backtracking. That is why it can blow up on
    mazes with loops, and why a step cap is needed.
    """
    t0 = time.perf_counter()
    on_path = {start}
    path = [start]
    visited = [start]
    iters = [iter(_neighbors(grid, *start))]
    found, capped = False, False
    while path:
        if path[-1] == end:
            found = True
            break
        if len(visited) >= step_limit:
            capped = True
            break
        advanced = False
        for nb in iters[-1]:
            if nb not in on_path:          # constraint check
                on_path.add(nb)            # choose
                path.append(nb)
                visited.append(nb)
                iters.append(iter(_neighbors(grid, *nb)))
                advanced = True
                break
        if not advanced:                   # dead end: undo the choice
            on_path.discard(path.pop())
            iters.pop()
    ms = (time.perf_counter() - t0) * 1000
    return Result("Backtracking", list(path) if found else [], visited, ms, found, capped)


SOLVERS = {"BFS": solve_bfs, "DFS": solve_dfs, "Backtracking": solve_backtracking}
