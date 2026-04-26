"""
Nem Placer v2 - multi-start legal optimization.

This placer keeps the safety of legal hard-macro placements while using several
different starts, local-search refinement, soft-macro relaxation, and true proxy
selection. It does not modify the evaluator.
"""

import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


class NemPlacerV2:
    def __init__(self, seed=7):
        self.seed = seed
        self.gap = 0.001
        self.refine_iters = 250
        self.logs = []

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        start_time = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.logs = []

        print("    v2 stage=load_plc", flush=True)
        plc = self._load_plc(benchmark.name)
        print("    v2 stage=extract_edges", flush=True)
        edges, edge_weights, neighbors = self._extract_edges(benchmark, plc)
        soft_neighbors = self._extract_soft_neighbors(benchmark)

        hard_n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:hard_n].cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:hard_n].cpu().numpy()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        raw = benchmark.macro_positions[:hard_n].cpu().numpy().astype(np.float64)

        print("    v2 stage=generate_candidates", flush=True)
        candidates = []
        if hard_n <= 420:
            candidates.append(("minimal_legal", self._legalize(raw, movable, sizes, canvas_w, canvas_h)))
        else:
            candidates.append(("original_repaired", self._repair_hard_bounds(raw, sizes, canvas_w, canvas_h)))
        candidates.append(("safe_height_rows", self._row_pack(benchmark, mode="height", neighbors=neighbors)))
        candidates.append(("area_rows", self._row_pack(benchmark, mode="area", neighbors=neighbors)))
        candidates.append(("connectivity_rows", self._row_pack(benchmark, mode="connectivity", neighbors=neighbors)))
        if hard_n <= 420:
            candidates.append(("center_cluster", self._center_cluster(benchmark, neighbors)))
        else:
            candidates.append(("center_cluster", self._row_pack(benchmark, mode="connectivity", neighbors=neighbors)))

        print("    v2 stage=refine_candidates", flush=True)
        refined = []
        for name, hard_pos in candidates:
            refined.append((f"{name}_start", hard_pos))
            if hard_n <= 420 and self._hard_overlap_count(hard_pos, sizes) == 0:
                iters = self.refine_iters if name in ("minimal_legal", "center_cluster") else self.refine_iters // 2
                opt = self._local_search(
                    hard_pos,
                    movable,
                    sizes,
                    canvas_w,
                    canvas_h,
                    edges,
                    edge_weights,
                    neighbors,
                    iters=iters,
                )
                refined.append((name, opt))

        print("    v2 stage=true_score", flush=True)
        best_placement = benchmark.macro_positions.clone()
        best_proxy = float("inf")

        for name, hard_pos in refined:
            base = benchmark.macro_positions.clone()
            base[:hard_n] = torch.tensor(hard_pos, dtype=torch.float32)

            variants = [(f"{name}_soft_initial", base)]
            if name == "minimal_legal_start":
                variants.append((f"{name}_soft_relaxed", self._relax_soft_macros(base, benchmark, soft_neighbors)))

            for variant_name, placement in variants:
                placement = self._repair_bounds(placement, benchmark)
                valid, violations = validate_placement(placement, benchmark)
                if not valid:
                    print(f"    candidate={variant_name} invalid: {'; '.join(violations[:3])}")
                    costs = {"proxy_cost": float("inf"), "wirelength_cost": float("inf"),
                             "density_cost": float("inf"), "congestion_cost": float("inf"),
                             "overlap_count": 999999}
                else:
                    costs = compute_proxy_cost(placement, benchmark, plc)

                self._log_candidate(variant_name, costs, time.time() - start_time)
                if costs["overlap_count"] == 0 and costs["proxy_cost"] < best_proxy:
                    best_proxy = costs["proxy_cost"]
                    best_placement = placement.clone()

        if not math.isfinite(best_proxy):
            best_placement = benchmark.macro_positions.clone()
            best_placement[:hard_n] = torch.tensor(
                self._safe_shelf(benchmark, raw, movable, sizes, canvas_w, canvas_h),
                dtype=torch.float32,
            )
            best_placement = self._relax_soft_macros(best_placement, benchmark, soft_neighbors)

        return best_placement

    def _repair_hard_bounds(self, pos, sizes, canvas_w, canvas_h):
        out = pos.copy()
        out[:, 0] = np.clip(out[:, 0], sizes[:, 0] / 2, canvas_w - sizes[:, 0] / 2)
        out[:, 1] = np.clip(out[:, 1], sizes[:, 1] / 2, canvas_h - sizes[:, 1] / 2)
        return out

    def _repair_bounds(self, placement, benchmark):
        out = placement.clone()
        sizes = benchmark.macro_sizes
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        for idx in range(benchmark.num_macros):
            w = sizes[idx, 0]
            h = sizes[idx, 1]
            if w < canvas_w:
                out[idx, 0] = torch.clamp(out[idx, 0], w / 2, canvas_w - w / 2)
            else:
                out[idx, 0] = benchmark.macro_positions[idx, 0]
            if h < canvas_h:
                out[idx, 1] = torch.clamp(out[idx, 1], h / 2, canvas_h - h / 2)
            else:
                out[idx, 1] = benchmark.macro_positions[idx, 1]
        fixed = benchmark.macro_fixed
        out[fixed] = benchmark.macro_positions[fixed]
        return out

    def _load_plc(self, name):
        from macro_place.loader import load_benchmark, load_benchmark_from_dir

        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if root.exists():
            _, plc = load_benchmark_from_dir(str(root).replace("\\", "/"))
            return plc

        ng45 = {
            "ariane133": "ariane133",
            "ariane136": "ariane136",
            "nvdla": "nvdla",
            "mempool_tile": "mempool_tile",
            "ariane133_ng45": "ariane133",
            "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla",
            "mempool_tile_ng45": "mempool_tile",
        }
        design = ng45.get(name)
        if design:
            base = Path("external/MacroPlacement/Flows/NanGate45") / design / "netlist" / "output_CT_Grouping"
            netlist = base / "netlist.pb.txt"
            plc_file = base / "initial.plc"
            if netlist.exists():
                _, plc = load_benchmark(str(netlist).replace("\\", "/"), str(plc_file).replace("\\", "/"), name=name)
                return plc
        return None

    def _extract_edges(self, benchmark, plc):
        hard_n = benchmark.num_hard_macros
        edge_dict = defaultdict(float)

        if plc is not None:
            name_to_hard = {}
            for bidx, pidx in enumerate(benchmark.hard_macro_indices):
                name_to_hard[plc.modules_w_pins[pidx].get_name()] = bidx

            for driver, sinks in plc.nets.items():
                macros = set()
                for pin_name in [driver] + sinks:
                    parent = pin_name.split("/")[0]
                    if parent in name_to_hard:
                        macros.add(name_to_hard[parent])

                if len(macros) < 2:
                    continue

                nodes = sorted(macros)
                # High-fanout nets matter, but a full clique should not swamp
                # two-pin locality. This damps fanout aggressively.
                net_w = 1.0 / math.sqrt(max(1, len(nodes) - 1))
                pair_w = net_w / max(1, len(nodes) - 1)
                if len(nodes) <= 16:
                    for i, a in enumerate(nodes):
                        for b in nodes[i + 1:]:
                            edge_dict[(a, b)] += pair_w
                else:
                    # Bounded expansion for high-fanout nets: a light chain plus
                    # a few anchors preserves locality signal without O(k^2)
                    # explosion on large benchmarks.
                    anchors = nodes[:4]
                    for i in range(len(nodes) - 1):
                        a, b = nodes[i], nodes[i + 1]
                        edge_dict[(a, b)] += pair_w
                    for a in anchors:
                        for b in nodes[4::4]:
                            if a != b:
                                edge_dict[(min(a, b), max(a, b))] += pair_w * 0.5
        else:
            for net in benchmark.net_nodes:
                nodes = sorted(i for i in net.tolist() if i < hard_n)
                if len(nodes) < 2:
                    continue
                pair_w = 1.0 / ((len(nodes) - 1) ** 1.5)
                if len(nodes) <= 16:
                    for i, a in enumerate(nodes):
                        for b in nodes[i + 1:]:
                            edge_dict[(a, b)] += pair_w
                else:
                    for i in range(len(nodes) - 1):
                        edge_dict[(nodes[i], nodes[i + 1])] += pair_w

        neighbors = [[] for _ in range(hard_n)]
        for (a, b), w in edge_dict.items():
            neighbors[a].append((b, w))
            neighbors[b].append((a, w))
        for items in neighbors:
            items.sort(key=lambda x: -x[1])

        if not edge_dict:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64), neighbors

        pairs = np.array(list(edge_dict.keys()), dtype=np.int64)
        weights = np.array([edge_dict[tuple(pair)] for pair in pairs], dtype=np.float64)
        return pairs, weights, neighbors

    def _extract_soft_neighbors(self, benchmark):
        soft = defaultdict(list)
        hard_n = benchmark.num_hard_macros

        for net in benchmark.net_nodes:
            nodes = net.tolist()
            hard_nodes = [i for i in nodes if i < hard_n]
            soft_nodes = [i for i in nodes if hard_n <= i < benchmark.num_macros]
            if not hard_nodes:
                continue
            w = 1.0 / math.sqrt(max(1, len(nodes) - 1))
            for sidx in soft_nodes:
                for hidx in hard_nodes:
                    soft[sidx].append((hidx, w))
        return soft

    def _legalize(self, pos, movable, sizes, canvas_w, canvas_h):
        n = len(pos)
        legal = pos.copy()
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        legal[:, 0] = np.clip(legal[:, 0], half_w, canvas_w - half_w)
        legal[:, 1] = np.clip(legal[:, 1], half_h, canvas_h - half_h)

        order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
        placed = np.zeros(n, dtype=bool)

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue
            if not self._overlaps_placed(idx, legal[idx], legal, sizes, placed):
                placed[idx] = True
                continue

            step = max(sizes[idx]) * 0.22
            best = legal[idx].copy()
            best_dist = float("inf")
            for radius in range(1, 55):
                found = False
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) != radius and abs(dy) != radius:
                            continue
                        cand = np.array([
                            np.clip(pos[idx, 0] + dx * step, half_w[idx], canvas_w - half_w[idx]),
                            np.clip(pos[idx, 1] + dy * step, half_h[idx], canvas_h - half_h[idx]),
                        ])
                        if self._overlaps_placed(idx, cand, legal, sizes, placed):
                            continue
                        dist = float(np.sum((cand - pos[idx]) ** 2))
                        if dist < best_dist:
                            best = cand
                            best_dist = dist
                            found = True
                if found:
                    break
            legal[idx] = best
            placed[idx] = True

        return legal

    def _row_pack(self, benchmark, mode, neighbors):
        hard_n = benchmark.num_hard_macros
        sizes_t = benchmark.macro_sizes[:hard_n]
        sizes = sizes_t.cpu().numpy().astype(np.float64)
        areas = sizes[:, 0] * sizes[:, 1]
        degree = np.array([sum(w for _, w in ns) for ns in neighbors])
        movable = benchmark.get_movable_mask()[:hard_n].cpu().numpy()
        order = [i for i in range(hard_n) if movable[i]]

        if mode == "height":
            order.sort(key=lambda i: (-sizes[i, 1], -areas[i]))
        elif mode == "area":
            order.sort(key=lambda i: (-areas[i], -sizes[i, 1]))
        else:
            seeds = sorted(order, key=lambda i: (-(areas[i] * (1.0 + degree[i])), -degree[i]))
            order = self._cluster_order(seeds, neighbors)

        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        pos = benchmark.macro_positions[:hard_n].cpu().numpy().astype(np.float64)
        x = 0.0
        y = 0.0
        row_h = 0.0
        row_no = 0

        for idx in order:
            w, h = sizes[idx]
            if x + w > canvas_w:
                self._center_previous_row(pos, row_start, row_end, canvas_w)
                x = 0.0
                y += row_h + self.gap
                row_h = 0.0
                row_no += 1

            if row_h == 0.0:
                row_start = idx
                row_end = idx

            if y + h > canvas_h:
                pos[idx] = [
                    np.clip(pos[idx, 0], w / 2, canvas_w - w / 2),
                    np.clip(pos[idx, 1], h / 2, canvas_h - h / 2),
                ]
                continue

            px = x + w / 2
            if row_no % 2 == 1 and mode == "connectivity":
                px = canvas_w - x - w / 2
            pos[idx] = [px, y + h / 2]
            x += w + self.gap
            row_h = max(row_h, h)
            row_end = idx

        return pos

    def _safe_shelf(self, benchmark, raw, movable, sizes, canvas_w, canvas_h):
        pos = raw.copy()
        areas = sizes[:, 0] * sizes[:, 1]
        order = sorted(np.where(movable)[0].tolist(), key=lambda i: (-sizes[i, 1], -areas[i]))
        x = 0.0
        y = 0.0
        row_h = 0.0
        for idx in order:
            w, h = sizes[idx]
            if x + w > canvas_w:
                x = 0.0
                y += row_h + 0.001
                row_h = 0.0
            if y + h > canvas_h:
                pos[idx] = [w / 2, h / 2]
            else:
                pos[idx] = [x + w / 2, y + h / 2]
            x += w + 0.001
            row_h = max(row_h, h)
        return pos

    def _center_previous_row(self, pos, row_start, row_end, canvas_w):
        # Placeholder kept intentionally conservative; row positions are already
        # legal and deterministic. Local search handles spreading later.
        return

    def _cluster_order(self, seeds, neighbors):
        unplaced = set(seeds)
        order = []
        while unplaced:
            if not order:
                current = next(i for i in seeds if i in unplaced)
            else:
                weighted = [(j, w) for j, w in neighbors[order[-1]] if j in unplaced]
                current = max(weighted, key=lambda x: x[1])[0] if weighted else next(i for i in seeds if i in unplaced)
            order.append(current)
            unplaced.remove(current)
            for nxt, _ in neighbors[current][:3]:
                if nxt in unplaced:
                    order.append(nxt)
                    unplaced.remove(nxt)
        return order

    def _center_cluster(self, benchmark, neighbors):
        hard_n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:hard_n].cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:hard_n].cpu().numpy()
        degree = np.array([sum(w for _, w in ns) for ns in neighbors])
        order = sorted([i for i in range(hard_n) if movable[i]], key=lambda i: -degree[i])
        pos = benchmark.macro_positions[:hard_n].cpu().numpy().astype(np.float64)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        cx = canvas_w / 2
        cy = canvas_h / 2
        golden = math.pi * (3.0 - math.sqrt(5.0))
        util_scale = math.sqrt(float((sizes[:, 0] * sizes[:, 1]).sum()) / (canvas_w * canvas_h))
        max_r = max(canvas_w, canvas_h) * (0.18 + 0.42 * util_scale)

        for k, idx in enumerate(order):
            r = max_r * math.sqrt((k + 0.5) / max(1, len(order)))
            theta = k * golden
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            w, h = sizes[idx]
            pos[idx, 0] = np.clip(x, w / 2, canvas_w - w / 2)
            pos[idx, 1] = np.clip(y, h / 2, canvas_h - h / 2)

        return self._legalize(pos, movable, sizes, canvas_w, canvas_h)

    def _local_search(self, pos, movable, sizes, canvas_w, canvas_h, edges, edge_weights, neighbors, iters):
        n = len(pos)
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        pos = pos.copy()
        best = pos.copy()
        current = self._cheap_objective(pos, sizes, canvas_w, canvas_h, edges, edge_weights)
        best_score = current
        max_dim = max(canvas_w, canvas_h)

        for step in range(iters):
            frac = step / max(1, iters - 1)
            temp = max_dim * (0.05 * (1.0 - frac) + 0.001)
            old_score = current
            move_type = random.random()

            if move_type < 0.45:
                touched, old = self._move_shift(pos, movable_idx, sizes, canvas_w, canvas_h, temp, large=(frac < 0.25 and random.random() < 0.12))
            elif move_type < 0.70:
                touched, old = self._move_swap(pos, movable_idx, sizes, canvas_w, canvas_h)
            elif move_type < 0.90:
                touched, old = self._move_toward_centroid(pos, movable_idx, sizes, canvas_w, canvas_h, neighbors, temp)
            else:
                touched, old = self._move_spread(pos, movable_idx, sizes, canvas_w, canvas_h, frac)

            if not touched or self._any_touched_overlap(pos, sizes, touched):
                self._restore(pos, touched, old)
                continue

            new_score = self._cheap_objective(pos, sizes, canvas_w, canvas_h, edges, edge_weights)
            delta = new_score - old_score
            accept = delta <= 0.0 or random.random() < math.exp(-delta / max(temp, 1e-9))
            if accept:
                current = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = pos.copy()
            else:
                self._restore(pos, touched, old)

        return best

    def _move_shift(self, pos, movable_idx, sizes, canvas_w, canvas_h, temp, large=False):
        i = int(random.choice(movable_idx))
        old = {i: pos[i].copy()}
        scale = temp * (3.0 if large else 0.8)
        pos[i, 0] += random.gauss(0.0, scale)
        pos[i, 1] += random.gauss(0.0, scale)
        self._clamp_one(pos, i, sizes, canvas_w, canvas_h)
        return [i], old

    def _move_swap(self, pos, movable_idx, sizes, canvas_w, canvas_h):
        i, j = random.sample(list(movable_idx), 2)
        old = {i: pos[i].copy(), j: pos[j].copy()}
        pos[i], pos[j] = old[j].copy(), old[i].copy()
        self._clamp_one(pos, i, sizes, canvas_w, canvas_h)
        self._clamp_one(pos, j, sizes, canvas_w, canvas_h)
        return [i, j], old

    def _move_toward_centroid(self, pos, movable_idx, sizes, canvas_w, canvas_h, neighbors, temp):
        weighted = [i for i in movable_idx if neighbors[i]]
        i = int(random.choice(weighted if weighted else movable_idx))
        old = {i: pos[i].copy()}
        total = sum(w for _, w in neighbors[i][:8])
        if total <= 0:
            return [], old
        centroid = sum(pos[j] * w for j, w in neighbors[i][:8]) / total
        alpha = random.uniform(0.10, 0.42)
        pos[i] = pos[i] * (1.0 - alpha) + centroid * alpha
        pos[i] += np.random.normal(0.0, temp * 0.12, size=2)
        self._clamp_one(pos, i, sizes, canvas_w, canvas_h)
        return [i], old

    def _move_spread(self, pos, movable_idx, sizes, canvas_w, canvas_h, frac):
        k = min(len(movable_idx), max(4, len(movable_idx) // 12))
        touched = random.sample(list(movable_idx), k)
        old = {i: pos[i].copy() for i in touched}
        center = np.array([canvas_w / 2, canvas_h / 2])
        factor = 0.015 * (1.0 - frac)
        for i in touched:
            pos[i] = pos[i] + (pos[i] - center) * factor
            self._clamp_one(pos, i, sizes, canvas_w, canvas_h)
        return touched, old

    def _cheap_objective(self, pos, sizes, canvas_w, canvas_h, edges, edge_weights):
        if len(edges):
            delta = np.abs(pos[edges[:, 0]] - pos[edges[:, 1]])
            hpwl = float(np.sum(edge_weights * (delta[:, 0] + delta[:, 1])))
        else:
            hpwl = 0.0

        grid_cols = 16
        grid_rows = 16
        grid = np.zeros((grid_rows, grid_cols), dtype=np.float64)
        cong_h = np.zeros_like(grid)
        cong_v = np.zeros_like(grid)
        cell_w = canvas_w / grid_cols
        cell_h = canvas_h / grid_rows
        cell_area = cell_w * cell_h

        x0 = np.clip(((pos[:, 0] - sizes[:, 0] / 2) / cell_w).astype(int), 0, grid_cols - 1)
        x1 = np.clip(((pos[:, 0] + sizes[:, 0] / 2) / cell_w).astype(int), 0, grid_cols - 1)
        y0 = np.clip(((pos[:, 1] - sizes[:, 1] / 2) / cell_h).astype(int), 0, grid_rows - 1)
        y1 = np.clip(((pos[:, 1] + sizes[:, 1] / 2) / cell_h).astype(int), 0, grid_rows - 1)
        for i in range(len(pos)):
            area_share = (sizes[i, 0] * sizes[i, 1]) / max(1, ((x1[i] - x0[i] + 1) * (y1[i] - y0[i] + 1)))
            grid[y0[i]: y1[i] + 1, x0[i]: x1[i] + 1] += area_share / cell_area

        if len(edges):
            cols = np.clip((pos[:, 0] / cell_w).astype(int), 0, grid_cols - 1)
            rows = np.clip((pos[:, 1] / cell_h).astype(int), 0, grid_rows - 1)
            for (a, b), w in zip(edges, edge_weights):
                r0, r1 = sorted((rows[a], rows[b]))
                c0, c1 = sorted((cols[a], cols[b]))
                cong_h[rows[a], c0:c1 + 1] += w
                cong_v[r0:r1 + 1, cols[b]] += w

        density = float(np.mean(np.sort(grid.ravel())[-max(1, grid.size // 10):] ** 2))
        congestion = float(np.mean(np.sort((cong_h + cong_v).ravel())[-max(1, grid.size // 20):]))
        boundary = float(np.sum(np.maximum(0, sizes[:, 0] / 2 - pos[:, 0]) ** 2))
        boundary += float(np.sum(np.maximum(0, pos[:, 0] - (canvas_w - sizes[:, 0] / 2)) ** 2))
        boundary += float(np.sum(np.maximum(0, sizes[:, 1] / 2 - pos[:, 1]) ** 2))
        boundary += float(np.sum(np.maximum(0, pos[:, 1] - (canvas_h - sizes[:, 1] / 2)) ** 2))
        overlap = self._overlap_penalty(pos, sizes)

        return hpwl + 35.0 * density + 8.0 * congestion + 1000.0 * boundary + 1.0e8 * overlap

    def _relax_soft_macros(self, placement, benchmark, soft_neighbors):
        if benchmark.num_soft_macros == 0:
            return placement

        out = placement.clone()
        sizes = benchmark.macro_sizes
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        hard_n = benchmark.num_hard_macros

        for idx in range(hard_n, benchmark.num_macros):
            if benchmark.macro_fixed[idx]:
                continue
            refs = soft_neighbors.get(idx)
            if not refs:
                continue

            total = sum(w for _, w in refs)
            target = sum(out[h] * float(w) for h, w in refs) / max(total, 1e-9)
            # Soft macros are allowed to overlap; blend rather than snap to
            # avoid creating extreme density spikes.
            out[idx] = 0.35 * out[idx] + 0.65 * target
            w = sizes[idx, 0]
            h = sizes[idx, 1]
            if w < canvas_w:
                out[idx, 0] = torch.clamp(out[idx, 0], w / 2, canvas_w - w / 2)
            else:
                out[idx, 0] = benchmark.macro_positions[idx, 0]
            if h < canvas_h:
                out[idx, 1] = torch.clamp(out[idx, 1], h / 2, canvas_h - h / 2)
            else:
                out[idx, 1] = benchmark.macro_positions[idx, 1]

        fixed = benchmark.macro_fixed
        out[fixed] = benchmark.macro_positions[fixed]
        return out

    def _overlap_penalty(self, pos, sizes):
        penalty = 0.0
        n = len(pos)
        for i in range(n):
            dx = np.abs(pos[i + 1:, 0] - pos[i, 0])
            dy = np.abs(pos[i + 1:, 1] - pos[i, 1])
            sx = (sizes[i + 1:, 0] + sizes[i, 0]) / 2 + self.gap
            sy = (sizes[i + 1:, 1] + sizes[i, 1]) / 2 + self.gap
            ox = np.maximum(0.0, sx - dx)
            oy = np.maximum(0.0, sy - dy)
            penalty += float(np.sum(ox * oy))
        return penalty

    def _hard_overlap_count(self, pos, sizes):
        count = 0
        n = len(pos)
        for i in range(n):
            dx = np.abs(pos[i + 1:, 0] - pos[i, 0])
            dy = np.abs(pos[i + 1:, 1] - pos[i, 1])
            sx = (sizes[i + 1:, 0] + sizes[i, 0]) / 2
            sy = (sizes[i + 1:, 1] + sizes[i, 1]) / 2
            count += int(np.sum((dx < sx) & (dy < sy)))
        return count

    def _any_touched_overlap(self, pos, sizes, touched):
        for idx in touched:
            if self._overlaps_any(idx, pos[idx], pos, sizes):
                return True
        return False

    def _overlaps_any(self, idx, cand, pos, sizes):
        dx = np.abs(pos[:, 0] - cand[0])
        dy = np.abs(pos[:, 1] - cand[1])
        sx = (sizes[:, 0] + sizes[idx, 0]) / 2 + self.gap
        sy = (sizes[:, 1] + sizes[idx, 1]) / 2 + self.gap
        hit = (dx < sx) & (dy < sy)
        hit[idx] = False
        return bool(np.any(hit))

    def _overlaps_placed(self, idx, cand, pos, sizes, placed):
        if not np.any(placed):
            return False
        dx = np.abs(pos[:, 0] - cand[0])
        dy = np.abs(pos[:, 1] - cand[1])
        sx = (sizes[:, 0] + sizes[idx, 0]) / 2 + self.gap
        sy = (sizes[:, 1] + sizes[idx, 1]) / 2 + self.gap
        hit = (dx < sx) & (dy < sy) & placed
        hit[idx] = False
        return bool(np.any(hit))

    def _clamp_one(self, pos, idx, sizes, canvas_w, canvas_h):
        pos[idx, 0] = np.clip(pos[idx, 0], sizes[idx, 0] / 2, canvas_w - sizes[idx, 0] / 2)
        pos[idx, 1] = np.clip(pos[idx, 1], sizes[idx, 1] / 2, canvas_h - sizes[idx, 1] / 2)

    def _restore(self, pos, touched, old):
        for idx in touched:
            pos[idx] = old[idx]

    def _log_candidate(self, name, costs, runtime):
        item = {
            "candidate": name,
            "proxy": costs["proxy_cost"],
            "wirelength": costs["wirelength_cost"],
            "density": costs["density_cost"],
            "congestion": costs["congestion_cost"],
            "overlaps": costs["overlap_count"],
            "runtime": runtime,
        }
        self.logs.append(item)
        print(
            f"    candidate={name} proxy={item['proxy']:.4f} "
            f"wl={item['wirelength']:.3f} den={item['density']:.3f} "
            f"cong={item['congestion']:.3f} overlaps={item['overlaps']} "
            f"t={item['runtime']:.2f}s"
        )
