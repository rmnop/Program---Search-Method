#Given a list of cities and their adjacencies—from city A, what cities are 
# next to it—can a route be found from city A to city X?

# Brute force approaches: 
# - Breadth-first search
# - depth-first search
# - ID-DFS search

#Hueristic Approaches 
# - best-first search
# - A* search


import csv
from collections import defaultdict, deque
import heapq
from queue import PriorityQueue
import time
from math import radians, atan2, sqrt, cos, sin  

#FIRST STEP - Parse files into data CSV FILE AND VERIFY IF TRUE
coordinates = {}
with open('coordinates.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        city, lat, lon = row
        coordinates[city] = (float(lat), float(lon))

graph = {}
with open('Adjacencies.txt', 'r') as file:
    for line in file:
        city1, city2 = line.strip().split()
        if city1 not in graph:
            graph[city1] = []
        if city2 not in graph:
            graph[city2] = []
        graph[city1].append(city2)
        graph[city2].append(city1)

#https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
#we need to calculate distance between two nodes - longitude/latitude: using haversine formula - assume earth is sphere
def distance(coord1, coord2):
    radius = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return (radius * c) * .621371

# ------------- BRUTE FORCE APPROACHES --------------
#GeeksforGeeks BFS example implemented with text file 
def bfs(start, destination):
    visited = set()
    queue = deque([(start, [start])])  # Initialize queue with start node and path
    while queue:
        #empty queue and start path from start
        start, path = queue.popleft() #popleft is O(1), pop() is O(n)
        if start == destination:
            return path
        #if start not destination, place start in path
        visited.add(start)
        for neighbor in graph[start]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None  # If no path found

# Depth-first search
#example from Favtutor.com and chatgpt on how dfs connected nodes with the ID-DFS search
def dfs(start, destination):
    visited = set()
    stack = [(start, [])]

    while stack:
        current, path = stack.pop()
        if current == end:
            return path + [current] #return current path
        if current not in visited:
            visited.add(current)
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    stack.append((neighbor, path + [current]))
    return None

    
#Iterative Deepening Depth-First Search: https://www.hackerearth.com/practice/algorithms/graphs/depth-first-search/tutorial/
def iddfs(start, destination):
    depth = 0
    while True:
        result = dfs_recursive(start, destination, [], depth)
        if result is not None:
            return result
        depth += 1

def dfs_recursive(current, destination, path, depth):
    if current == destination:
        return path + [current]
    if depth == 0:
        return None
    for neighbor in graph.get(current, []):
        if neighbor not in path:
            #use recursive to call teh neighbor and reduce depth
            result = dfs_recursive(neighbor, destination, path + [current], depth - 1)
            if result is not None:
                return result
    return None

# ------------- HUERISTIC APPROACHES --------------
def best_first_search(start, destination):
    visited = set()
    heap = [(distance(coordinates[start], coordinates[destination]), start, [])]

    while heap:
        _, current, path = heapq.heappop(heap)
        if current == destination:
            return path + [current]
        if current not in visited:
            visited.add(current)
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    heapq.heappush(heap, (distance(coordinates[neighbor], coordinates[destination]), neighbor, path + [current]))
    return None

def a_star_search(start, destination):
    visited = set()
    #use a heap to store nodes based on their total hueristic cost to get to the goal
    heap = [(0, distance(coordinates[start], coordinates[destination]), start, [])]

    while heap:
        _, _, current, path = heapq.heappop(heap)
        if current == destination:
            return path + [current]
        if current not in visited:
            visited.add(current)
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    g = len(path) + 1  # Cost from start to current node(neighbor)
                    h = distance(coordinates[neighbor], coordinates[destination])  # calculates hueristic cost to destination
                    f = g + h  # total cost
                    heapq.heappush(heap, (f, h, neighbor, path + [current]))
    return None
            

while True:
     
    start = input("Location: ")
    end = input("Destination: ")
    if start not in coordinates or end not in coordinates:
        print("Invalid cities, Please enter again.")
        continue
        
        
    print("Select a search method:")
    print("1. Breadth-first search")
    print("2. Depth-first search")
    print("3. ID-DFS search")
    print("4. Best-first search")
    print("5. A* search")

    S_Method = int(input("1-5: "))

    start_time = time.time()
    end_time = time.time()
    if(S_Method == 1):
        path = (bfs(start, end))
    elif(S_Method == 2):
        path = dfs(start, end) 
    elif(S_Method == 3):
        path = iddfs(start, end)
    elif(S_Method == 4):
        path = best_first_search(start, end)
    elif(S_Method == 5):
        path = a_star_search(start, end)
    else:
        ("Not a valid Algorithm")
        continue
    
    if path:
        print("Route: ", (path))
        total_distance = sum(distance(coordinates[path[i]], coordinates[path[i+1]]) for i in range(len(path)-1))
        print("Total Distance: ", round(total_distance), "miles")
        print("Total Time", end_time - start_time, "seconds")
    else:
        print("No route found")

    Input = input("Do you want to reroute? (yes/no): ")
    if Input.lower() != "yes":
        break




