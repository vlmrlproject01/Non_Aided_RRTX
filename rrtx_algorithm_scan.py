"""Simplified 2-D RRTX-style planner for TurtleBot3.

This module supplies the occupancy grid, collision checking, tree growth,
graph repair and path extraction used by ``rrtx_node_fixed.py``.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Set
import heapq

np.random.seed(42)

# CONFIGURATION CONSTANTS

MAP_MIN_X = -15.0
MAP_MAX_X = 15.0
MAP_MIN_Y = -15.0
MAP_MAX_Y = 15.0

GRID_RESOLUTION = 0.05    

ROBOT_RADIUS = 0.105        
SAFETY_MARGIN = 0.08
INFLATION_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN   # ~0.185m

STEP_SIZE = 0.5
NEIGHBOR_RADIUS = 1.2
GOAL_BIAS = 0.1
DUPLICATE_NODE_TOLERANCE = 0.05
MAX_TREE_NODES = 800



# OCCUPANCY GRID -- shape-agnostic obstacle representation

class OccupancyGrid2D:
    
    def __init__(self, min_x=MAP_MIN_X, max_x=MAP_MAX_X, min_y=MAP_MIN_Y, max_y=MAP_MAX_Y,
                 resolution=GRID_RESOLUTION):
        self.resolution = resolution
        self.origin_x = min_x
        self.origin_y = min_y
        self.width = max(1, int((max_x - min_x) / resolution))
        self.height = max(1, int((max_y - min_y) / resolution))
        self.raw = np.zeros((self.height, self.width), dtype=bool)       # true = occupied
        self.inflated = np.zeros((self.height, self.width), dtype=bool)  # used for collision checks

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def in_grid_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def in_world_bounds(self, x: float, y: float) -> bool:
        return (
            self.origin_x <= x < self.origin_x + self.width * self.resolution
            and self.origin_y <= y < self.origin_y + self.height * self.resolution
        )

    def mark_occupied(self, x: float, y: float):
        gx, gy = self.world_to_grid(x, y)
        if self.in_grid_bounds(gx, gy):
            self.raw[gy, gx] = True

    def mark_free(self, x: float, y: float):
        gx, gy = self.world_to_grid(x, y)
        if self.in_grid_bounds(gx, gy):
            self.raw[gy, gx] = False

    def update_from_scan_hit(self, robot_x: float, robot_y: float, hit_x: float, hit_y: float):
        
        dist = math.hypot(hit_x - robot_x, hit_y - robot_y)
        if dist < 1e-6:
            return
        step = self.resolution * 0.5
        steps = max(1, int(dist / step))
        for i in range(steps):
            t = i / steps
            self.mark_free(robot_x + (hit_x - robot_x) * t, robot_y + (hit_y - robot_y) * t)
        self.mark_occupied(hit_x, hit_y)

    def compute_inflated(self, radius_m: float = INFLATION_RADIUS):
        
        radius_cells = max(1, int(round(radius_m / self.resolution)))
        occ = np.argwhere(self.raw)
        inflated = self.raw.copy()
        if len(occ) > 0:
            rr = radius_cells
            offsets = [(dy, dx) for dy in range(-rr, rr + 1) for dx in range(-rr, rr + 1)
                       if dy * dy + dx * dx <= rr * rr]
            offs = np.array(offsets)
            ys = occ[:, 0][:, None] + offs[:, 0][None, :]
            xs = occ[:, 1][:, None] + offs[:, 1][None, :]
            valid = (ys >= 0) & (ys < self.height) & (xs >= 0) & (xs < self.width)
            inflated[ys[valid], xs[valid]] = True
        self.inflated = inflated

    def clear_around(self, x: float, y: float, radius: float):
        
        gx, gy = self.world_to_grid(x, y)
        if not self.in_grid_bounds(gx, gy):
            return
        r_cells = max(1, int(round(radius / self.resolution)))
        y0, y1 = max(0, gy - r_cells), min(self.height, gy + r_cells + 1)
        x0, x1 = max(0, gx - r_cells), min(self.width, gx + r_cells + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - gy) ** 2 + (xx - gx) ** 2 <= r_cells ** 2
        sub = self.inflated[y0:y1, x0:x1]
        sub[mask] = False
        self.inflated[y0:y1, x0:x1] = sub

    def is_free_world(self, x: float, y: float) -> bool:
        if not self.in_world_bounds(x, y):
            return False
        gx, gy = self.world_to_grid(x, y)
        if not self.in_grid_bounds(gx, gy):
            return False
        return not self.inflated[gy, gx]


def is_collision_free(point: Tuple[float, float], grid: OccupancyGrid2D) -> bool:
    return grid.is_free_world(point[0], point[1])


def is_edge_collision_free(p1, p2, grid: OccupancyGrid2D) -> bool:
    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    num_checks = max(10, int(dist / (grid.resolution * 0.5)))
    for i in range(num_checks + 1):
        t = i / num_checks
        point = (p1[0] * (1 - t) + p2[0] * t, p1[1] * (1 - t) + p2[1] * t)
        if not is_collision_free(point, grid):
            return False
    return True



def preload_static_circle(grid: OccupancyGrid2D, center: Tuple[float, float], radius: float):
    cx, cy = center
    n = max(16, int(2 * math.pi * radius / grid.resolution))
    for i in range(n):
        theta = 2 * math.pi * i / n
        for r in np.linspace(0, radius, max(2, int(radius / grid.resolution))):
            grid.mark_occupied(cx + r * math.cos(theta), cy + r * math.sin(theta))



# RRTX NODE & PLANNER

class RRTXNode:
    def __init__(self, position: Tuple[float, float]):
        self.position = position
        self.parent: Optional['RRTXNode'] = None
        self.children: Set['RRTXNode'] = set()
        self.neighbors: Set['RRTXNode'] = set()
        self.g: float = float('inf')
        self.lmc: float = float('inf')

    def __lt__(self, other):
        return (min(self.g, self.lmc), self.g) < (min(other.g, other.lmc), other.g)


class RRTX:
    def __init__(self, start: Tuple[float, float], goal: Tuple[float, float], grid: OccupancyGrid2D):
        self.start_pos = start
        self.goal_pos = goal
        self.grid = grid  # reference -- the node mutates grid.raw/inflated in place

        self.nodes: List[RRTXNode] = []
        self.goal_node = RRTXNode(goal)
        self.goal_node.g = 0.0
        self.goal_node.lmc = 0.0
        self.nodes.append(self.goal_node)

        self.queue: List[Tuple[Tuple[float, float], RRTXNode]] = []
        self.queue_set: Set[RRTXNode] = set()

    def update_start(self, new_start):
        self.start_pos = new_start

    def distance(self, p1, p2) -> float:
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _get_nearby_nodes(self, position: Tuple[float, float], radius: float) -> List[RRTXNode]:
        if not self.nodes:
            return []
        coords = np.array([n.position for n in self.nodes])
        dists = np.hypot(coords[:, 0] - position[0], coords[:, 1] - position[1])
        indices = np.where(dists <= radius)[0]
        return [self.nodes[idx] for idx in indices]

    def steer(self, from_pt, to_pt):
        dist = self.distance(from_pt, to_pt)
        if dist < STEP_SIZE:
            return to_pt
        theta = np.arctan2(to_pt[1] - from_pt[1], to_pt[0] - from_pt[0])
        return (from_pt[0] + STEP_SIZE * np.cos(theta), from_pt[1] + STEP_SIZE * np.sin(theta))

    def get_key(self, node: RRTXNode):
        return (min(node.g, node.lmc), node.g)

    def insert_or_update_queue(self, node: RRTXNode):
        """Insert a node or replace its previous priority-queue entry."""
        if node in self.queue_set:
            self.queue = [
                (key, queued_node)
                for key, queued_node in self.queue
                if queued_node is not node
            ]
            heapq.heapify(self.queue)

        heapq.heappush(self.queue, (self.get_key(node), node))
        self.queue_set.add(node)

    def make_parent(self, child: RRTXNode, parent: Optional[RRTXNode]):
        """Update the parent-child relationship safely."""
        if child is parent:
            return

        if child.parent is not None and child.parent is not parent:
            child.parent.children.discard(child)

        child.parent = parent

        if parent is not None:
            parent.children.add(child)

    def verify_queue(self):
        while self.queue:
            _, u = heapq.heappop(self.queue)
            if u not in self.queue_set:
                continue
            self.queue_set.remove(u)

            if u.g > u.lmc:
                u.g = u.lmc
                for v in list(u.neighbors):
                    if v is self.goal_node:
                        continue
                    if not is_edge_collision_free(v.position, u.position, self.grid):
                        continue
                    candidate = u.g + self.distance(v.position, u.position)
                    if candidate < v.lmc:
                        v.lmc = candidate
                        self.make_parent(v, u)
                        self.insert_or_update_queue(v)
            else:
                u.g = float('inf')
                for child in list(u.children):
                    child.lmc = float('inf')
                    self.insert_or_update_queue(child)

                best_parent, best_lmc = None, float('inf')
                for v in u.neighbors:
                    if v.g == float('inf'):
                        continue
                    if not is_edge_collision_free(u.position, v.position, self.grid):
                        continue
                    c = v.g + self.distance(u.position, v.position)
                    if c < best_lmc:
                        best_lmc, best_parent = c, v

                u.lmc = best_lmc
                if best_parent is not None:
                    self.make_parent(u, best_parent)
                elif u.parent is not None:
                    u.parent.children.discard(u)
                    u.parent = None

                for v in list(u.neighbors):
                    if v.parent == u:
                        v.lmc = float('inf')
                        self.insert_or_update_queue(v)

                if u.g != u.lmc:
                    self.insert_or_update_queue(u)

    def _start_is_reachable(self) -> bool:
        if not is_collision_free(self.start_pos, self.grid):
            return False
        v_start_closest = min(self.nodes, key=lambda n: self.distance(n.position, self.start_pos))
        if v_start_closest.g == float('inf'):
            return False
        return is_edge_collision_free(self.start_pos, v_start_closest.position, self.grid)

    def grow_tree(self, max_iter: int = 100):
        if not self.nodes:
            return

        for _ in range(max_iter):
            if len(self.nodes) >= 150 and self._start_is_reachable():
                break

            if np.random.random() < GOAL_BIAS:
                v_rand = self.start_pos
            else:
                v_rand = (np.random.uniform(MAP_MIN_X, MAP_MAX_X), np.random.uniform(MAP_MIN_Y, MAP_MAX_Y))

            v_nearest = min(self.nodes, key=lambda n: self.distance(n.position, v_rand))
            v_new_pos = self.steer(v_nearest.position, v_rand)

            near_existing = self._get_nearby_nodes(v_new_pos, DUPLICATE_NODE_TOLERANCE)
            if near_existing or not is_collision_free(v_new_pos, self.grid):
                continue

            v_new = RRTXNode(v_new_pos)
            v_near = self._get_nearby_nodes(v_new_pos, NEIGHBOR_RADIUS)

            for neighbor in v_near:
                if is_edge_collision_free(v_new.position, neighbor.position, self.grid):
                    v_new.neighbors.add(neighbor)
                    neighbor.neighbors.add(v_new)

            if not v_new.neighbors:
                continue
            self.nodes.append(v_new)

            for neighbor in list(v_new.neighbors):
                cost = neighbor.g + self.distance(v_new.position, neighbor.position)
                if cost < v_new.lmc:
                    v_new.lmc = cost
                    self.make_parent(v_new, neighbor)

            if v_new.parent:
                v_new.g = v_new.lmc
                for neighbor in list(v_new.neighbors):
                    if neighbor.lmc > v_new.g + self.distance(neighbor.position, v_new.position):
                        neighbor.lmc = v_new.g + self.distance(neighbor.position, v_new.position)
                        self.make_parent(neighbor, v_new)
                        self.insert_or_update_queue(neighbor)

            self.verify_queue()

        self.prune_tree(max_nodes=MAX_TREE_NODES)

    def sync_with_grid(self):
       
        if not self.nodes:
            return

        invalidated_nodes = []
        for node in list(self.nodes):
            if not is_collision_free(node.position, self.grid):
                node.lmc = float('inf')
                invalidated_nodes.append(node)
                self.insert_or_update_queue(node)

        for node in self.nodes:
            if node in invalidated_nodes:
                continue
            for nbr in list(node.neighbors):
                if nbr in invalidated_nodes or not is_edge_collision_free(node.position, nbr.position, self.grid):
                    node.neighbors.discard(nbr)
                    nbr.neighbors.discard(node)
                    if node.parent == nbr:
                        node.lmc = float('inf')
                        self.insert_or_update_queue(node)
                    if nbr.parent == node:
                        nbr.lmc = float('inf')
                        self.insert_or_update_queue(nbr)

        for node in self.nodes:
            potential_nbrs = self._get_nearby_nodes(node.position, NEIGHBOR_RADIUS)
            for p_nbr in potential_nbrs:
                if p_nbr == node or p_nbr in node.neighbors:
                    continue
                if is_edge_collision_free(node.position, p_nbr.position, self.grid):
                    node.neighbors.add(p_nbr)
                    p_nbr.neighbors.add(node)

                    c1 = p_nbr.g + self.distance(node.position, p_nbr.position)
                    if c1 < node.lmc:
                        node.lmc = c1
                        self.make_parent(node, p_nbr)
                        self.insert_or_update_queue(node)

                    c2 = node.g + self.distance(p_nbr.position, node.position)
                    if c2 < p_nbr.lmc:
                        p_nbr.lmc = c2
                        self.make_parent(p_nbr, node)
                        self.insert_or_update_queue(p_nbr)

        self.verify_queue()

    def prune_tree(self, max_nodes: int = 600):
        if len(self.nodes) <= max_nodes:
            return

        valid_nodes = [n for n in self.nodes if n.g != float('inf') or n == self.goal_node]
        excess_nodes = [n for n in self.nodes if n.g == float('inf') and n != self.goal_node]

        while len(valid_nodes) + len(excess_nodes) > max_nodes and excess_nodes:
            dead_node = excess_nodes.pop()
            if dead_node.parent:
                dead_node.parent.children.discard(dead_node)
            for nbr in list(dead_node.neighbors):
                nbr.neighbors.discard(dead_node)
            if dead_node in self.queue_set:
                self.queue_set.remove(dead_node)

        self.nodes = valid_nodes + excess_nodes
        self.queue = [(self.get_key(n), n) for _, n in self.queue if n in self.queue_set]
        heapq.heapify(self.queue)

    def extract_path(self) -> List[Tuple[float, float]]:
        if not self.nodes:
            return []

        if not is_collision_free(self.start_pos, self.grid):
            return []

        v_start_nearest = min(self.nodes, key=lambda n: self.distance(n.position, self.start_pos))
        if v_start_nearest.g == float('inf') or not is_edge_collision_free(self.start_pos, v_start_nearest.position, self.grid):
            return []

        raw_path = []
        if self.distance(self.start_pos, v_start_nearest.position) > 0.02:
            raw_path.append(self.start_pos)

        curr: Optional[RRTXNode] = v_start_nearest
        visited = set()
        while curr is not None and curr not in visited:
            raw_path.append(curr.position)
            visited.add(curr)
            curr = curr.parent

        if len(raw_path) < 3:
            return raw_path

        smoothed_path = [raw_path[0]]
        curr_idx = 0
        while curr_idx < len(raw_path) - 1:
            next_idx = len(raw_path) - 1
            while next_idx > curr_idx + 1:
                if is_edge_collision_free(smoothed_path[-1], raw_path[next_idx], self.grid):
                    break
                next_idx -= 1
            smoothed_path.append(raw_path[next_idx])
            curr_idx = next_idx

        return smoothed_path 
