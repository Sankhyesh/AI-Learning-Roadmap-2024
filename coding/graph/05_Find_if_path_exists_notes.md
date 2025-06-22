## 1. IP–OP–PS (Input, Output, Problem Statement)

### 1. Problem Statement

We are given an undirected graph with `n` vertices labeled `0…n–1` and a list of edges.  We need to determine **whether there exists any path** from a given `source` vertex to a given `destination` vertex.

### 2. Input Specification

* **n**: Integer, number of vertices (1 ≤ n ≤ 10⁵).
* **edges**: List of pairs `[u, v]`, each `0 ≤ u, v < n`.  The graph is undirected; no self-loops or parallel edges guaranteed.
* **source**, **destination**: Integers in `0…n–1`.

**Constraints:**

* 1 ≤ n ≤ 10⁵
* 0 ≤ edges.length ≤ 2×10⁵
* Each edge endpoint in `[0, n–1]`

**Edge Cases:**

* **Empty edge list**: graph is totally disconnected
* **Single node**: n=1
* **source == destination**
* **Disconnected components**: multiple isolated “islands” in the graph
* **Maximum size**: n or edge count near the upper bound

### 3. Output Specification

Return a single boolean:

* `true` if there is at least one path from `source` to `destination`
* `false` otherwise

### 4. Examples

| Example | Input                                                                     | Output | Notes                            |
| ------- | ------------------------------------------------------------------------- | ------ | -------------------------------- |
| 1       | n=3, edges=\[\[0,1],\[1,2],\[2,0]], source=0, destination=2               | true   | cycle connects all vertices      |
| 2       | n=6, edges=\[\[0,1],\[0,2],\[3,5],\[5,4],\[4,3]], source=0, destination=5 | false  | two disconnected components      |
| 3       | n=1, edges=\[], source=0, destination=0                                   | true   | trivial single-node path         |
| 4       | n=4, edges=\[\[0,1],\[2,3]], source=1, destination=2                      | false  | 1 and 2 are in different islands |

---

## 2. Identification

### 1. Why This Algorithm/Technique?

* We need to answer a **single connectivity query** on an undirected graph.
* **Union-Find (Disjoint Set)** excels at grouping connected components in near-constant time per edge.

### 2. Key Cues/Characteristics

* Undirected edges
* “Is there any path…” ⇒ connectivity, not shortest path
* Single query after building structure

### 3. Alternative Approaches

1. **BFS/DFS** from `source`, stop if you reach `destination`.
2. **Graph coloring / flood fill** to label components.

### 4. Trade-off Analysis

| Approach   | Time Complexity | Space Complexity | Notes                                                     |
| ---------- | --------------- | ---------------- | --------------------------------------------------------- |
| BFS / DFS  | O(n + E)        | O(n + E)         | Simple to implement, but uses a queue/recursion stack     |
| Union-Find | O(n + E α(n))   | O(n)             | Very fast “find” queries, good if multiple queries needed |

### 5. Justification

* Union-Find groups all vertices in **O(E α(n))** and then answers connectivity in **O(α(n))**—effectively constant.
* Clean separation: build once, then one comparison of roots.

---

## 3. Break Down → Disjoint Set Union (Union-Find)

### 1. High-Level Strategy

1. **Initialize** every vertex as its own set.
2. **Union** the two endpoints for each edge.
3. **Find** the set representative (root) for `source` and `destination`.
4. Compare roots: equal ⇒ path exists, else no.

### 2. Pseudocode

```
function validPath(n, edges, source, destination):
    if source == destination:
        return true

    DSU = new DisjointSet(n)
    for (u, v) in edges:
        DSU.union(u, v)

    return DSU.find(source) == DSU.find(destination)


class DisjointSet:
    init(n):
        parent[i] = i for each i in 0…n-1
        rank[i]   = 0

    find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        else if rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
```

### 3. Step-by-Step Explanation

1. **Same-node check**

   * If `source == destination`, no edges needed.
2. **Initialization**

   * `parent[i] = i`, `rank[i] = 0`.
3. **Union loops**

   * Merge the two sets for each edge, keeping trees shallow via `rank`.
4. **Final find**

   * Path exists iff both endpoints share the same ultimate parent.

---

## 4. Explanations + Code

### 1. Detailed Explanation

* **Initialization**: makes each vertex its own root → no connections yet.
* **Union**: for each edge, we merge by root, using rank to minimize tree height.
* **Find**: path-compression flattens the tree for future calls.
* **Query**: comparing two `find` calls tells us if they became connected.

### 2. Code Implementation (C++)

```cpp
#include <vector>
using namespace std;

class DisjointSet {
public:
    DisjointSet(int n)
      : parent(n), rank_(n, 0)
    {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);  // path compression
        return parent[x];
    }

    void unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return;            // already connected

        // union by rank
        if (rank_[ra] < rank_[rb]) {
            parent[ra] = rb;
        } else if (rank_[ra] > rank_[rb]) {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank_[ra]++;
        }
    }

private:
    vector<int> parent;
    vector<int> rank_;
};

class Solution {
public:
    bool validPath(int n,
                   vector<vector<int>>& edges,
                   int source,
                   int destination)
    {
        // Edge case: same node
        if (source == destination)
            return true;

        DisjointSet ds(n);

        // Build connectivity
        for (auto &e : edges)
            ds.unite(e[0], e[1]);

        // Check if they share the same root
        return ds.find(source) == ds.find(destination);
    }
};
```

### 3. Complexity Analysis

* **Time:**

  * Initialization: O(n)
  * E unions, each O(α(n)) → O(E α(n))
  * Two finds: O(α(n))
  * **Total:** O(n + E α(n))
* **Space:**

  * O(n) for `parent` and `rank_` arrays

### 4. Potential Optimizations

* Path-splitting or path-halving instead of full recursion.
* Early exit in union loop if `find(source) == find(destination)` ever becomes true.

### 5. Common Pitfalls

* Forgetting the `source == destination` check.
* Omitting path compression or union by rank → degrades performance to O(n).
* Mixing 0-indexed vs. 1-indexed vertices.

---

## 5. Animated Visualization

### 1. Visualization Goal

Illustrate how union operations merge sets and how the final `find` calls determine connectivity.

### 2. Proposed Visualization Approach

* **Static Diagram Sequence**: Show successive parent-array snapshots after each `union`.

### 3. Visualization Description / Instructions

1. **State 0**: `parent = [0,1,2,3]`
2. **After unite(0,1)**: draw arrow `1→0` → `parent = [0,0,2,3]`
3. **After unite(2,3)**: draw arrow `3→2` → `parent = [0,0,2,2]`
4. **After unite(1,2)**: connect root(1)=0 to root(2)=2 → `parent = [0,2,2,2]`
5. Final `find(0)=2` and `find(3)=2` ⇒ same root ⇒ path exists.

### 4. (Optional) Sample Code Snippet

```python
# Simple matplotlib demonstration of parent-array evolution
import matplotlib.pyplot as plt

def show_parents(parents, step):
    plt.figure(figsize=(4,1))
    plt.title(f"Step {step}: {parents}")
    plt.axis('off')
    plt.text(0.1, 0.5, str(parents), fontsize=14)
    plt.show()

steps = [
    [0,1,2,3],
    [0,0,2,3],
    [0,0,2,2],
    [0,2,2,2]
]

for i, p in enumerate(steps):
    show_parents(p, i)
 