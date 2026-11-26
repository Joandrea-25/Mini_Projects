"""Solve a shuffled square-tile image puzzle using only pixel information.

Usage:
  python puzzle_solver.py solve   jumbled.png solved.png --rows 8 --cols 8 [--rotate]
  python puzzle_solver.py shuffle photo.jpg  jumbled.png --rows 8 --cols 8 [--rotate]
"""
import argparse
import numpy as np
from PIL import Image


def split(img, rows, cols):
    h, w = img.shape[0] // rows, img.shape[1] // cols
    return [img[r*h:(r+1)*h, c*w:(c+1)*w] for r in range(rows) for c in range(cols)]


def right_cost(left, right):
    """cost[i, j] = how badly right[j] fits to the right of left[i] (MGC measure).
    It checks that the colour change across the seam matches the colour trend
    on each side of the seam."""
    def one_side(a, b):
        edge_a, edge_b = a[:, :, -1], b[:, :, 0]          # (N, height, 3)
        grad = edge_a - a[:, :, -2]
        # Dummy gradients keep the covariance stable in flat areas.
        dummy = np.array([[0,0,0],[1,1,1],[-1,-1,-1],[1,0,0],[-1,0,0],
                          [0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
        out = np.empty((len(a), len(b)))
        for i in range(len(a)):
            samples = np.vstack([grad[i], dummy])
            mean = grad[i].mean(0)
            inv = np.linalg.inv(np.cov(samples, rowvar=False))
            diff = edge_b - edge_a[i] - mean               # (N, height, 3)
            out[i] = np.einsum('nhi,ij,nhj->n', diff, inv, diff)
        return out
    forward = one_side(left, right)
    backward = one_side(right[:, :, ::-1], left[:, :, ::-1]).T
    return np.sqrt(np.maximum(forward, 0)) + np.sqrt(np.maximum(backward, 0))


def build_costs(variants, n, turns):
    """D[d, a, b]: cost of variant b being next to variant a.
    d: 0=right, 1=down, 2=left, 3=up. Variant id = piece * turns + rotation."""
    right = right_cost(variants, variants)
    down_view = np.stack([np.rot90(v) for v in variants])  # bottom becomes right
    down = right_cost(down_view, down_view)
    D = np.stack([right, down, right.T, down.T])
    owner = np.arange(len(variants)) // turns
    D[:, owner[:, None] == owner[None, :]] = np.inf        # no self matches
    return D


def normalise(D, turns):
    """Divide each cost by the best competing cost. Below 1 means a distinctive match."""
    def runner_up(M, axis):
        s = np.sort(M, axis=axis)
        first = np.take(s, [0], axis=axis)
        second = np.take(s, [1], axis=axis)
        return np.where(M == first, second, first)
    S = np.empty_like(D)
    for d in range(4):
        denom = np.minimum(runner_up(D[d], 1), runner_up(D[d], 0))
        S[d] = D[d] / (denom + 1e-9)
    return S


STEP = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def fits(cells, rows, cols, rotate):
    rs, cs = [p[0] for p in cells], [p[1] for p in cells]
    h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    return (h <= rows and w <= cols) or (rotate and h <= cols and w <= rows)


def assemble(S, n, turns, rows, cols):
    """Join the most reliable pairs first (Kruskal style), without overlaps."""
    finite = np.where(np.isfinite(S.reshape(-1)))[0]
    # Only right and down are needed: left and up are the same pairs reversed.
    finite = finite[finite < 2 * S.shape[1] * S.shape[2]]
    order = finite[np.argsort(S.reshape(-1)[finite], kind='stable')]

    clusters = {p: {(0, 0): (p, 0)} for p in range(n)}   # cell -> (piece, rotation)
    where = {p: (p, (0, 0)) for p in range(n)}           # piece -> (cluster, cell)

    for flat in order:
        d, a, b = np.unravel_index(flat, S.shape)
        pa, ra = divmod(int(a), turns)
        pb, rb = divmod(int(b), turns)
        ca, cell_a = where[pa]
        cb, cell_b = where[pb]
        if ca == cb:
            continue
        # Rotate cluster A's frame to agree with this match, if rotation is allowed.
        k = (clusters[ca][cell_a][1] - ra) % turns
        d2, rb2 = int(d), rb
        if k:
            d2 = (int(d) - k) % 4      # a CCW turn maps right->up, down->right
            rb2 = (rb + k) % 4
        target = (cell_a[0] + STEP[d2][0], cell_a[1] + STEP[d2][1])
        turn = (rb2 - clusters[cb][cell_b][1]) % turns

        def spin(cell):
            r, c = cell
            for _ in range(turn):
                r, c = -c, r
            return r, c

        origin = spin(cell_b)
        dr, dc = target[0] - origin[0], target[1] - origin[1]
        moved = {}
        for cell, (p, rot) in clusters[cb].items():
            r, c = spin(cell)
            moved[(r + dr, c + dc)] = (p, (rot + turn) % turns)
        if moved.keys() & clusters[ca].keys():
            continue
        if not fits(list(clusters[ca]) + list(moved), rows, cols, turns == 4):
            continue
        clusters[ca].update(moved)
        for cell, (p, _) in moved.items():
            where[p] = (ca, cell)
        del clusters[cb]
        if len(clusters) == 1:
            break
    return max(clusters.values(), key=len)


def neighbour_cost(grid, r, c, v, D):
    """Average cost of putting variant v at (r, c), and the number of neighbours."""
    total, count = 0.0, 0
    for d, (dr, dc) in enumerate(STEP):
        y, x = r + dr, c + dc
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] >= 0:
            total += D[d, v, grid[y, x]]
            count += 1
    return (total / count if count else np.inf), count


def fill(cluster, n, turns, rows, cols, D):
    """Crop the largest cluster to the board, then place the remaining pieces."""
    rs, cs = [p[0] for p in cluster], [p[1] for p in cluster]
    if max(rs) - min(rs) + 1 > rows or max(cs) - min(cs) + 1 > cols:
        rows, cols = cols, rows          # assembled sideways (rotation mode)
    best = None
    for top in range(min(rs), max(min(rs), max(rs) - rows + 1) + 1):
        for lft in range(min(cs), max(min(cs), max(cs) - cols + 1) + 1):
            inside = {(r-top, c-lft): v for (r, c), v in cluster.items()
                      if 0 <= r-top < rows and 0 <= c-lft < cols}
            if best is None or len(inside) > len(best):
                best = inside
    grid = np.full((rows, cols), -1)
    for (r, c), (p, rot) in best.items():
        grid[r, c] = p * turns + rot
    free = set(range(n)) - {p for p, _ in best.values()}
    while free:
        empty = [(r, c) for r, c in zip(*np.where(grid < 0))]
        most = max(neighbour_cost(grid, r, c, 0, D)[1] for r, c in empty)
        options = [(neighbour_cost(grid, r, c, p*turns+k, D)[0], int(r), int(c), p*turns+k)
                   for r, c in empty if neighbour_cost(grid, r, c, 0, D)[1] == most
                   for p in free for k in range(turns)]
        _, r, c, v = min(options)
        grid[r, c] = v
        free.remove(v // turns)
    return grid


def solve(img, rows, cols, rotate=False):
    """Return the reconstructed image as a uint8 array."""
    tiles = split(img, rows, cols)
    h, w = tiles[0].shape[:2]
    if rotate and h != w:
        raise ValueError("Rotation mode needs square pieces.")
    if min(h, w) < 2 or len(tiles) < 2:
        raise ValueError("Need at least 2 pieces, each at least 2x2 pixels.")
    turns = 4 if rotate else 1
    variants = np.stack([np.rot90(t, k) for t in tiles for k in range(turns)])
    D = build_costs(variants.astype(float), len(tiles), turns)
    cluster = assemble(normalise(D, turns), len(tiles), turns, rows, cols)
    grid = fill(cluster, len(tiles), turns, rows, cols, D)
    return np.vstack([np.hstack([variants[v] for v in row]) for row in grid])


def shuffle(img, rows, cols, rotate=False, seed=None):
    rng = np.random.default_rng(seed)
    tiles = split(img, rows, cols)
    if rotate and tiles[0].shape[0] != tiles[0].shape[1]:
        raise ValueError("Rotation mode needs square pieces.")
    tiles = [tiles[i] for i in rng.permutation(len(tiles))]
    if rotate:
        tiles = [np.rot90(t, rng.integers(4)) for t in tiles]
    return np.vstack([np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(rows)])


def main():
    ap = argparse.ArgumentParser(description="Image puzzle solver")
    ap.add_argument("mode", choices=["solve", "shuffle"])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rotate", action="store_true", help="pieces may be rotated")
    args = ap.parse_args()
    img = np.array(Image.open(args.input).convert("RGB"))
    if img.shape[0] % args.rows or img.shape[1] % args.cols:
        print("Warning: size not divisible by grid; extra edge pixels are cropped.")
    func = solve if args.mode == "solve" else shuffle
    Image.fromarray(func(img, args.rows, args.cols, args.rotate)).save(args.output)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
