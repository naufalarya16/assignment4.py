import heapq
import time
import random

class TerrainMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        self.start = None
        self.goal = None

    def set_elevation(self, x, y, elevation):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = elevation
            return True
        return False

    def add_no_fly_zone(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = '#'
            return True
        return False

    def set_start(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] != '#':
            self.start = (x, y)
            return True
        return False

    def set_goal(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] != '#':
            self.goal = (x, y)
            return True
        return False

    def is_valid_position(self, x, y):
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.grid[y][x] != '#')

    def get_elevation_cost(self, current, neighbor):
        x1, y1 = current
        x2, y2 = neighbor

        if self.grid[y1][x1] == '#' or self.grid[y2][x2] == '#':
            return float('inf')

        elevation_diff = abs(int(self.grid[y2][x2]) - int(self.grid[y1][x1]))
        if int(self.grid[y2][x2]) > int(self.grid[y1][x1]):
            return 1 + elevation_diff * 1.5
        else:
            return 1 + elevation_diff * 0.5

    def generate_random_terrain(self, max_elevation=9, no_fly_zones=10):
        self.grid = [[random.randint(1, max_elevation) for _ in range(self.width)] 
                     for _ in range(self.height)]

        for _ in range(no_fly_zones):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.add_no_fly_zone(x, y)

        while True:
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            if self.grid[y][x] != '#':
                self.set_start(x, y)
                break

        while True:
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            if self.grid[y][x] != '#' and (x, y) != self.start:
                self.set_goal(x, y)
                break

    def print_map(self, path=None, visited=None):
        visual_grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.start:
                    row.append('S')
                elif pos == self.goal:
                    row.append('G')
                elif path and pos in path:
                    row.append('*')
                elif visited and pos in visited:
                    row.append('+')
                elif self.grid[y][x] == '#':
                    row.append('#')
                else:
                    row.append(str(self.grid[y][x]))
            visual_grid.append(row)

        for row in visual_grid:
            print(' '.join(row))

def a_star_search(terrain_map):
    if terrain_map.start is None or terrain_map.goal is None:
        return None, 0, 0, set()

    start_time = time.time()

    start = terrain_map.start
    goal = terrain_map.goal

    def heuristic(pos):
        x1, y1 = pos
        x2, y2 = goal
        return abs(x2 - x1) + abs(y2 - y1)

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    open_set = []
    closed_set = set()
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    parent = {}
    visited_count = 0

    heapq.heappush(open_set, (f_score[start], start))

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_count += 1

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()

            time_taken = (time.time() - start_time) * 1000
            return path, visited_count, time_taken, closed_set

        closed_set.add(current)

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if not terrain_map.is_valid_position(nx, ny):
                continue

            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + terrain_map.get_elevation_cost(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    time_taken = (time.time() - start_time) * 1000
    return None, visited_count, time_taken, closed_set

def greedy_best_first_search(terrain_map):
    if terrain_map.start is None or terrain_map.goal is None:
        return None, 0, 0, set()

    start_time = time.time()

    start = terrain_map.start
    goal = terrain_map.goal

    def heuristic(pos):
        x1, y1 = pos
        x2, y2 = goal
        return abs(x2 - x1) + abs(y2 - y1)

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    open_set = []
    closed_set = set()
    parent = {}
    visited_count = 0

    heapq.heappush(open_set, (heuristic(start), start))

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_count += 1

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()

            time_taken = (time.time() - start_time) * 1000
            return path, visited_count, time_taken, closed_set

        closed_set.add(current)

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if not terrain_map.is_valid_position(nx, ny):
                continue

            if neighbor in closed_set:
                continue

            if neighbor not in parent:
                parent[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor), neighbor))

    time_taken = (time.time() - start_time) * 1000
    return None, visited_count, time_taken, closed_set

if __name__ == "__main__":
    terrain_map = TerrainMap(10, 10)
    terrain_map.generate_random_terrain()

    print("Peta Terrain:")
    terrain_map.print_map()

    print("\nMencari jalur terbang dengan A*...")
    a_star_path, a_star_visited, a_star_time, a_star_visited_nodes = a_star_search(terrain_map)

    if a_star_path:
        print(f"A* berhasil menemukan jalur dengan {len(a_star_path)} langkah.")
        print(f"Jumlah node yang dikunjungi: {a_star_visited}")
        print(f"Waktu eksekusi: {a_star_time:.2f} ms")
        print("\nPeta dengan jalur A*:")
        terrain_map.print_map(a_star_path, a_star_visited_nodes)
    else:
        print("A* tidak menemukan jalur.")

    print("\nMencari jalur terbang dengan Greedy Best-First Search...")
    gbfs_path, gbfs_visited, gbfs_time, gbfs_visited_nodes = greedy_best_first_search(terrain_map)

    if gbfs_path:
        print(f"GBFS berhasil menemukan jalur dengan {len(gbfs_path)} langkah.")
        print(f"Jumlah node yang dikunjungi: {gbfs_visited}")
        print(f"Waktu eksekusi: {gbfs_time:.2f} ms")
        print("\nPeta dengan jalur GBFS:")
        terrain_map.print_map(gbfs_path, gbfs_visited_nodes)
    else:
        print("GBFS tidak menemukan jalur.")

    print("\nPerbandingan A* vs GBFS:")
    if a_star_path and gbfs_path:
        print(f"Panjang jalur A*: {len(a_star_path)} langkah")
        print(f"Panjang jalur GBFS: {len(gbfs_path)} langkah")
        print(f"Node yang dikunjungi A*: {a_star_visited}")
        print(f"Node yang dikunjungi GBFS: {gbfs_visited}")
        print(f"Waktu eksekusi A*: {a_star_time:.2f} ms")
        print(f"Waktu eksekusi GBFS: {gbfs_time:.2f} ms")
