INF = 10**9  # "infinity" value

def dijkstra_matrix_arraypq(W, s):
    n = len(W)                    # number of vertices
    dist = [INF] * n              # best known distances
    parent = [-1] * n             # shortest path tree
    in_queue = [True] * n         # all vertices start in the "queue"

    dist[s] = 0                   # distance to source is 0

    # main loop: repeat n times
    for _ in range(n):
        # ----- extractMin (linear scan) -----
        u = -1
        best = INF
        for i in range(n):
            if in_queue[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1 or dist[u] == INF:
            break  # no more reachable vertices
        in_queue[u] = False       # remove from queue

        # ----- relax neighbors -----
        for v in range(n):
            if W[u][v] == INF:    # skip if no edge
                continue
            alt = dist[u] + W[u][v]
            if alt < dist[v]:     # found a shorter path
                dist[v] = alt
                parent[v] = u

    return dist, parent


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Example graph (same as C++ test):
    n = 5
    W = [[INF] * n for _ in range(n)]
    for i in range(n):
        W[i][i] = 0

    edges = [
        (0, 1, 5),
        (0, 2, 8),
        (0, 4, 7),
        (1, 3, 9),
        (2, 3, 1),
        (3, 4, 2)
    ]
    for u, v, w in edges:
        W[u][v] = w  # directed edge

    source = 0
    dist, parent = dijkstra_matrix_arraypq(W, source)

    print("dist:", dist)
    print("parent:", parent)


