#Given a list of cities and their adjacencies—from city A, what cities are 
# next to it—can a route be found from city A to city X?

# Brute force approaches: 
# - Breadth-first search
# - depth-first search
# - ID-DFS search

#Hueristic Approaches 
# - best-first search
# - A* search

#You'll want to think about the initial conditions and 
#how to check for a "valid" sequence and 
# GOAL check at each step of the search process.

import csv
from collections import defaultdict, deque
import heapq
from queue import PriorityQueue
import time



#FIRST STEP - Parse files into data CSV FILE AND VERIFY IF TRUE
def load_csv(filename):
    mylist = []
    with open(filename) as file:
        file_data = csv.reader(file, delimiter=',')
        for row in file_data:
            mylist.append([cell.lower() for cell in row])
        return mylist
    
def read_txt_file(filename):
    mylist = []
    with open(filename) as file:
        file_data = csv.reader(file, delimiter=',')
        for row in file_data:
            mylist.append([cell.lower() for cell in row])
        return mylist   
    
#Modified graph to adjacent cities
graph = {
        'Anthony': ['Bluff_City', 'Argonia', 'Harper'], 
        'Bluff_City': ['Kiowa', 'South_Haven', 'Mayfield'],
        'Kiowa': ['Bluff_City', 'Attica', 'Coldwater'],
        'Attica': ['Harper', 'Medicine_Lodge', 'Kiowa', 'Attica'],
        'Augusta': ['Winfield', 'Andover', 'Emporia'],
        'Winfield': ['Augusta', 'Andover'],
        'Andover': ['Winfield', 'Leon', 'Towanda', 'Augusta'],        
        'Leon': ['Andover', 'Wichita'],
        'Caldwell': ['South_Haven', 'Wellington', 'Argonia'], 
        'South_Haven': ['Caldwell', 'Bluff_City', 'Mulvane'],
        'Bluff_City': ['Anthony', 'Kiowa', 'South_Haven', 'Mayfield'],
        'El_Dorado': ['Towanda', 'Hillsboro'],
        'Towanda': ['El_Dorado', 'Andover'], 
        'Florence': ['McPherson', 'Hutchinson'],
        'McPherson': ['Hillsboro', 'Florence', 'Newton', 'Hutchinson', 'Salina'], 
        'Hillsboro': ['McPherson', 'El_Dorado', 'Lyons'],
        'Greensburg': ['Coldwater'],
        'Coldwater': ['Pratt', 'Greensburg', ],
        'Harper': ['Anthony', 'Attica', ],
        'Argonia': ['Anthony', 'Rago', 'Caldwell'],
        'Hutchinson': ['Newton', 'Pratt', 'McPherson', 'Florence'],
        'Newton': ['Haven', 'Hutchinson', 'McPherson', 'Emporia', 'El_Dorado'],
        'Junction_City': ['Abilene'],
        'Abilene': ['Marion', 'Junction_City', 'Salina', 'Hays'], 
        'Marion': ['Manhattan', 'Abilene', 'McPherson'],
        'Manhattan': ['Topeka', 'Marion'],
        'Kingman': ['Cheney'],
        'Cheney': ['Pratt', 'Kingman'], 
        'Pratt': ['Hutchinson', 'Coldwater', 'Cheney', 'Sawyer', 'Zenda'],
        'Mayfield': ['Wellington', 'Bluff_City', 'Oxford', 'Mulvane'],
        'Wellington': ['Caldwell', 'Mayfield', 'Oxford'], 
        'Caldwell': ['Argonia', 'South_Haven', 'Wellington'],
        'Salina': ['Lyons', 'McPherson', 'Abilene'], 
        'Lyons': ['Hillsboro', 'Salina'],
        'Medicine_Lodge': ['Attica', 'Attica'],
        'Rago': ['Viola', 'Argonia'],
        'Viola': ['Sawyer', 'Rago'],
        'Sawyer': ['Pratt', 'Viola',],
        'Oxford': ['Mayfield', 'Wellington'], 
        'Mayfield': ['Mulvane', 'Bluff_City', 'Oxford', 'Wellington'],
        'Mulvane': ['South_Haven', 'Mayfield', 'Cheney', 'Andover'],
        'Wichita': ['Leon', 'Derby'],
        'Derby': ['Wichita', 'Clearwater'], 
        'Clearwater': ['Cheney', 'Derby'],
        'Cheney': ['Kingman', 'Pratt', 'Clearwater', 'Mulvane'], 
        }


# ------------- BRUTE FORCE APPROACHES --------------
def btfs(graph, start, end):
    queue = []
    # push the path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node == end:
            return path
        #construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)

# Depth-first search
def dfs(graph, start, end, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path = path + [start]

    if start == end:
        return path

    for neighbor in graph[start]:
        if neighbor not in visited:
            new_path = dfs(graph, neighbor, end, visited, path)
            if new_path:
                return new_path

    return None

class Graph:
 
    def __init__(self,vertices):
 
        # No. of vertices
        self.V = vertices
 
        # default dictionary to store graph
        self.graph = defaultdict(list)
 
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # A function to perform a Depth-Limited search
    # from given source 'src'
def ID_DFS(self,graph, end, maxDepth):

    if graph == end : return True
 
        # If reached the maximum depth, stop recursing.
    if maxDepth <= 0 : return False
 
        # Recur for all the vertices adjacent to this vertex
    for i in self.graph[graph]:
            if(self.DLS(i,end,maxDepth-1)):
                 return True
            return False
 
    # IDDFS to search if target is reachable from v.
    # It uses recursive DLS()
def IDDFS(self,graph, end, maxDepth):
 
        # Repeatedly depth-limit search till the
        # maximum depth
    for i in range(maxDepth):
        if (self.DLS(graph, end, i)):
            return True
    return False
 

# ------------- HUERISTIC APPROACHES --------------
def best_first_search(graph, start, end):
    visited = [False] * end
    pq = PriorityQueue()
    pq.put((0, graph))
    visited[graph] = True
     
    while pq.empty() == False:
        u = pq.get()[1]
        # Displaying the path having lowest cost
        print(u, end=" ")
        if u == start:
            break
 
        for v, c in graph[u]:
            if visited[v] == False:
                visited[v] = True
                pq.put((c, v))

# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination
 
# Check if a cell is unblocked
def is_unblocked(graph, row, col):
    return graph[row][col] == 1
 
# Check if a cell is the destination
def is_destination(row, col, end):
    return row == end[0] and col == end[1]
 
# Calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(row, col, end):
    return ((row - end[0]) ** 2 + (col - end[1]) ** 2) ** 0.5
 
# Trace the path from source to destination
def trace_path(cell_details, end):
    print("The Path is ")
    path = []
    row = end[0]
    col = end[1]
 
    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col
 
    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()
 
    # Print the path
    for i in path:
        print("->", i, end=" ")
    print()


def a_star_search(graph, start, end):
    queue = [(0 + hueristic(start, end), 0, [start])]
    while queue:
        (_, cost, path) = heapq.heappop(queue)
        current = path[-1]
        if current == end:
            return path
        for neighbor in graph[current]:
            if neighbor not in path:
                new_cost = cost + 1
                heapq.heappush(queue, (new_cost + hueristic(neighbor, end), new_cost, path + [neighbor]))
            
# Heuristic function (Euclidean distance between cities)
def hueristic(city1, city2):
    lat1, lon1 = [city1]
    lat2, lon2 = [city2]
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 


while True:
    start = input("Location: ")
    end = input("Destination: ")
    S_Method = input("What Search Method?: ")

    start_time = time.time()
    end_time = time.time()


    if(S_Method == 'BTFS' or 'Breadth First Search'):
        print(btfs(graph, start, end))
        print("Total time:", end_time - start_time, "seconds")
    elif(S_Method == 'DFS' or 'Depth First Search'):
        print(dfs(graph, start, end))
        print("Total time:", end_time - start_time, "seconds") 
    elif(S_Method == 'IDFS' or 'Iterative Deep Search'):
        print(ID_DFS(graph, start, end))
        print("Total time:", end_time - start_time, "seconds") 
    elif(S_Method == 'BFS' or 'Best First Search'):
        print(best_first_search(graph, start, end))
        print("Total time:", end_time - start_time, "seconds") 
    elif(S_Method == 'AS' or 'A* Search'):
        print(a_star_search(graph, start, end))
        print("Total time:", end_time - start_time, "seconds") 
    else:
        print("Not in Kansas")



