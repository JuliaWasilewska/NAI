import argparse
import math
import random
import time
import numpy as np
from PIL import Image


# Funkcja wyszukiwania ścieżki A* z uwzględnieniem kosztów terenu
def search_path_with_astar(start, goal, accessible_fn, h, callback_fn, get_cost_fn):
    open_set = {tuple(start)}
    closed_set = set()
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): h(start, goal)}

    while open_set:
        callback_fn(closed_set, open_set)

        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == tuple(goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(tuple(start))
            return path[::-1]

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in accessible_fn(current):
            if tuple(neighbor) in closed_set:
                continue

            # Uwzględnienie kosztu przejścia przez teren
            tentative_g_score = g_score.get(current, float('inf')) + h(neighbor, current) + get_cost_fn(neighbor)

            if tuple(neighbor) not in open_set:
                open_set.add(tuple(neighbor))
            elif tentative_g_score >= g_score.get(tuple(neighbor), float('inf')):
                continue

            came_from[tuple(neighbor)] = current
            g_score[tuple(neighbor)] = tentative_g_score
            f_score[tuple(neighbor)] = g_score[tuple(neighbor)] + h(neighbor, goal)

    return []

## Funkcje heurystyczne
def manhattan_heuristic(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

def euclidean_heuristic(start, end):
    return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

def random_heuristic(start, end):
    return random.random()

# Funkcja obliczająca koszt przejścia w zależności od koloru pikseli (jaśniejszy teren jest trudniejszy)
def get_cost(bitmap, neighbor):
    return bitmap[neighbor[1], neighbor[0]][0] / 255.0  # Normalizacja do przedziału [0, 1]

# Funkcja zwracająca sąsiadów, uwzględniająca ruchy na ukos (6.1)
def accessible(bitmap, dims, point):
    neighbors = []
    height, width = dims
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]  # Ruchy na ukos i wzdłuż osi

    for delta in deltas:
        neighbor = (point[0] + delta[0], point[1] + delta[1])

        x, y = neighbor[0], neighbor[1]
        if 0 <= x < width and 0 <= y < height:
            if bitmap[y, x][0] == 0:  # Sprawdzamy, czy punkt jest dostępny (np. biały teren)
                neighbors.append(neighbor)
    return neighbors

# Funkcje pomocnicze do obsługi bitmap
def getpixel(image, dims, position):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return None
    return image[position[1], position[0]]

def setpixel(image, dims, position, value):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return
    image[position[1], position[0]] = value

# Funkcja ładowania mapy
def load_world_map(fname):
    img = Image.open(fname)
    img = img.convert("RGBA")
    pixels = np.array(img)
    dims = pixels.shape[:2]
    return dims, pixels

# Funkcja zapisywania mapy
def save_world_map(fname, image):
    img = Image.fromarray(image)
    img.save(fname)

# Znajdowanie pozycji piksela o określonym kolorze
def find_pixel_position(image, dims, value):
    for y in range(dims[0]):
        for x in range(dims[1]):
            if tuple(image[y, x]) == value:
                return [x, y]
    raise ValueError("Nie znaleziono piksela o podanej wartości!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A* Pathfinding with multiple heuristics')
    parser.add_argument('heuristic', choices=['manhattan', 'euclidean', 'random'], help='Wybór metryki heurystycznej')
    args = parser.parse_args()

    dims, bitmap = load_world_map("img.png")

    start = find_pixel_position(bitmap, dims, (255, 0, 255, 255))  # Punkt startowy (fioletowy)
    goal = find_pixel_position(bitmap, dims, (255, 255, 0, 255))  # Punkt docelowy (żółty)

    heuristics = {
        'manhattan': manhattan_heuristic,
        'euclidean': euclidean_heuristic,
        'random': random_heuristic
    }

    heuristic = heuristics[args.heuristic]

    setpixel(bitmap, dims, start, (0, 0, 0, 255))  # Zmieniamy kolor startu na czarny
    setpixel(bitmap, dims, goal, (0, 0, 0, 255))   # Zmieniamy kolor celu na czarny

    start_time = time.time()
    path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), heuristic, lambda x, y: None, lambda n: get_cost(bitmap, n))
    end_time = time.time()

    if path:
        print(f"Heurystyka: {args.heuristic}, Długość trasy: {len(path)}, Czas obliczeń: {end_time - start_time:.6f} sekund")
        for p in path:
            setpixel(bitmap, dims, p, (255, 0, 0, 255))  # Zaznaczamy ścieżkę na czerwono
    else:
        print(f"Heurystyka: {args.heuristic}, Nie znaleziono trasy.")

    save_world_map(f"result_{args.heuristic}.png", bitmap)
