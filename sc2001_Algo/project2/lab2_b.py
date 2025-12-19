import heapq # Import heapq to use a min-heap (priority queue)

def dijkstra(V, E, start):
    # Initialize distances and predecessors
    d = {node: float('inf') for node in V} # Distance from start to each node, initially infinity
    pi = {node: None for node in V} # Predecessor of each node in the shortest path
    d[start] = 0 # Distance to start node is 0

    # Priority queue: (distance, node)
    pq = [(0, start)] # Min-heap initialized with start node at distance 0
    S = set() # Set of visited nodes (nodes with final shortest distance)

    print("Initial priority queue:", sorted([(node, d[node]) for node in V], key=lambda x: x[1]))

    while pq:
        dist_u, u = heapq.heappop(pq) # Pop node with smallest distance
        if u in S: # If node already processed, skip
            continue
        S.add(u) # Mark node as processed

        # Update distances if shorter path is found
        for v, w in E.get(u, []):
            if dist_u + w < d[v]: # If new path through u is shorter
                d[v] = dist_u + w # Update distance
                pi[v] = u # Update predecessor
                heapq.heappush(pq, (d[v], v)) # Add neighbor to priority queue

        # Print full priority queue as list of all nodes with current shortest distances
        full_queue = sorted([(node, d[node]) for node in V], key=lambda x: x[1])
        print("Priority queue now:", full_queue)

    # After all nodes processed, print final priority queue sorted by shortest distance
    final_pq = sorted([(node, d[node]) for node in V], key=lambda x: x[1])
    print("Final Priority Queue:", final_pq)

    return d, pi  #Return final distances and predecessors


# --- Example run ---
V = {"A", "B", "C", "D", "E", "F", "G"}
E = {
    "A": [("B", 2), ("C", 1)],
    "B": [("C", 2), ("D", 5), ("E", 12)],
    "C": [("D", 3), ("F", 5)],
    "D": [("E", 1), ("F", 2)],
    "E": [("G", 3)],
    "F": [("G", 1)],
    "G": []
}

distances, predecessors = dijkstra(V, E, "A")