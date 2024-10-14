import heapq
import math
import random
import time
from typing import List, Tuple, Dict

# Define point_t as a tuple for easier hashing
point_t = Tuple[int, int]


def search_path_with_astar(start: point_t, goal: point_t, accessible_fn, h_fn) -> List[point_t]:
    """A* pathfinding algorithm."""
    open_set = []
    heapq.heappush(open_set, (0 + h_fn(start, goal), start))

    came_from: Dict[point_t, point_t] = {}
    g_score = {start: 0}
    f_score = {start: h_fn(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in accessible_fn(current):
            tentative_g_score = g_score[current] + h_fn(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + h_fn(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Return an empty list if no path found


def h_function(a: point_t, b: point_t) -> float:
    """Heuristic function using Manhattan distance."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy


def get_pixel(bitmap: List[int], dims: List[int], pos: point_t) -> int:
    """Gets the value at the specified position from the bitmap."""
    index = pos[1] * dims[0] + pos[0]
    if 0 <= pos[0] < dims[0] and 0 <= pos[1] < dims[1]:
        return bitmap[index]
    return -1  # Return -1 if out of bounds


def set_pixel(bitmap: List[int], dims: List[int], pos: point_t, value: int) -> None:
    """Sets the value at the specified position in the bitmap."""
    index = pos[1] * dims[0] + pos[0]
    if 0 <= pos[0] < dims[0] and 0 <= pos[1] < dims[1]:
        bitmap[index] = value


def accessible(bitmap: List[int], dims: List[int], p: point_t) -> List[point_t]:
    """Returns a list of accessible neighbors for the point p."""
    neighbors = []
    for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = (p[0] + d[0], p[1] + d[1])
        if get_pixel(bitmap, dims, neighbor) == 0:
            neighbors.append(neighbor)
    return neighbors


def main():
    dims = [10, 10]
    bitmap = [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    start = (1, 1)
    goal = (9, 5)
    set_pixel(bitmap, dims, start, 0)
    set_pixel(bitmap, dims, goal, 0)

    path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), h_function)

    for p in path:
        set_pixel(bitmap, dims, p, 9)

    # Display the grid
    display_map = {9: 'o', 1: '#', 0: '·'}
    for y in range(dims[1]):
        row = ""
        for x in range(dims[0]):
            pixel_value = get_pixel(bitmap, dims, (x, y))
            row += f"{display_map.get(pixel_value, '?')} "
        print(row)

    print(f"Path length1"
          f": {len(path)}")

if __name__ == "__main__":
    main()
