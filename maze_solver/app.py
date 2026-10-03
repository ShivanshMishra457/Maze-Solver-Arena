"""Maze Solver Arena: compare BFS, DFS and Backtracking.

Run:  streamlit run app.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from maze import SOLVERS, generate_maze, parse_maze

st.set_page_config(page_title="Maze Solver Arena", layout="wide")

COLORS = {
    "wall": (0.10, 0.11, 0.15), "open": (0.93, 0.94, 0.96),
    "visited": (0.45, 0.65, 0.95), "path": (1.00, 0.76, 0.10),
    "start": (0.20, 0.75, 0.40), "end": (0.90, 0.25, 0.25),
}

WHY = {
    "BFS": "Explores level by level, so the first time it reaches the end is by the fewest steps. Always shortest, but it visits a lot of cells.",
    "DFS": "Dives down one corridor before trying others. Often visits fewer cells, but the path it finds can be long. No shortest-path guarantee.",
    "Backtracking": "Builds a path, undoes it at dead ends and tries again. It only blocks cells on the current path, so on mazes with loops it re-explores cells and can be very slow.",
}


def draw(grid, start, end, visited_upto, path=None):
    img = np.zeros((len(grid), len(grid[0]), 3))
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            img[r, c] = COLORS["wall"] if v else COLORS["open"]
    for r, c in visited_upto:
        img[r, c] = COLORS["visited"]
    for r, c in path or []:
        img[r, c] = COLORS["path"]
    img[start] = COLORS["start"]
    img[end] = COLORS["end"]
    fig, ax = plt.subplots(figsize=(6, 6 * len(grid) / len(grid[0])))
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0)
    return fig


# ---------- sidebar: maze source ----------
st.sidebar.header("Maze")
mode = st.sidebar.radio("Source", ["Generate", "Paste my own"])
if mode == "Generate":
    size = st.sidebar.slider("Size (cells)", 11, 61, 31, step=2)
    loops = st.sidebar.slider("Extra loops", 0.0, 0.5, 0.05, 0.01,
                              help="0 = one route only. More loops = several routes, where the algorithms really differ.")
    seed = st.sidebar.number_input("Seed", 0, 99999, 7)
    if st.sidebar.button("New random maze"):
        seed = int(time.time()) % 99999
        st.session_state["seed_override"] = seed
    seed = st.session_state.get("seed_override", seed)
    grid, start, end = generate_maze(size, size, loops, seed)
else:
    sample = "#########\n#S    # #\n# ### # #\n#   #   #\n# # ### #\n# #     E\n#########"
    text = st.sidebar.text_area("# = wall, S = start, E = end", sample, height=220)
    try:
        grid, start, end = parse_maze(text)
    except ValueError as e:
        st.error(str(e))
        st.stop()

step_limit = st.sidebar.number_input("Backtracking step limit", 1000, 1_000_000, 200_000, step=10_000,
                                     help="Safety cap so backtracking can't run forever.")

# ---------- solve all ----------
results = {}
for name, fn in SOLVERS.items():
    results[name] = fn(grid, start, end, step_limit) if name == "Backtracking" else fn(grid, start, end)

st.title("Maze Solver Arena")
st.caption("BFS vs DFS vs Backtracking on the same maze. Green = start, red = end, blue = explored, gold = final path.")

# ---------- comparison ----------
rows = []
for n, r in results.items():
    rows.append({
        "Algorithm": n,
        "Solved": "Hit step limit" if r.capped else ("Yes" if r.found else "No path"),
        "Path length": len(r.path) - 1 if r.found else None,
        "Cells visited": len(r.visited),
        "Time (ms)": round(r.time_ms, 3),
    })
df = pd.DataFrame(rows).set_index("Algorithm")

st.subheader("Comparison")
st.dataframe(df, width="stretch")

c1, c2, c3 = st.columns(3)
c1.caption("Path length (shorter is better)")
c1.bar_chart(df["Path length"])
c2.caption("Cells visited")
c2.bar_chart(df["Cells visited"])
c3.caption("Time (ms)")
c3.bar_chart(df["Time (ms)"])

solved = {n: r for n, r in results.items() if r.found}
if solved:
    shortest = min(len(r.path) for r in solved.values())
    winners = [n for n, r in solved.items() if len(r.path) == shortest]
    fewest = min(solved, key=lambda n: len(solved[n].visited))
    st.success(f"Shortest path: {', '.join(winners)} ({shortest - 1} steps). "
               f"Fewest cells explored: {fewest} ({len(solved[fewest].visited)}).")
else:
    st.error("No algorithm found a path from start to end.")

# ---------- step-by-step viewer ----------
st.subheader("Watch it solve")
left, right = st.columns([2, 1])
with right:
    algo = st.selectbox("Algorithm", list(SOLVERS))
    res = results[algo]
    st.info(WHY[algo])
    n = len(res.visited)
    speed = st.select_slider("Speed", ["Slow", "Medium", "Fast"], "Medium")
    play = st.button("Play animation")
    k = st.slider("Step", 0, n, n, key=f"step_{algo}")
with left:
    holder = st.empty()
    if play:
        per_frame = {"Slow": max(1, n // 80), "Medium": max(1, n // 40), "Fast": max(1, n // 15)}[speed]
        for i in range(0, n + per_frame, per_frame):
            i = min(i, n)
            fig = draw(grid, start, end, res.visited[:i], res.path if i == n else None)
            holder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.05)
    else:
        fig = draw(grid, start, end, res.visited[:k], res.path if k == n else None)
        holder.pyplot(fig)
        plt.close(fig)
