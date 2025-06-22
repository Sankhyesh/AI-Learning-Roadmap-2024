/*
Problem Description:
- n: total number of vertices labeled from 0 through n-1
- edges: list of pairs [u,v] representing connections
- source: starting vertex
- destination: target vertex
- Goal: Determine if there exists any path (via one or more edges) from source to destination
*/

/*
Algorithm Steps:
1. Initialize Union-Find:
   - For i in 0 to n-1, set parent[i] = i, rank[i] = 0
2. Process edges:
   - For each (u,v):
     * Find roots ru = find(u), rv = find(v)
     * If ru != rv, attach smaller rank root under larger rank
     * If ranks equal, increment one rank
3. Check connectivity:
   - If source == destination, return true immediately
   - Otherwise, check if source and destination are in same set
*/

#include <vector>
#include <cstdio>
using namespace std;

class UnionFind
{
public:
    UnionFind(int n) : parent(n), rank_(n, 0)
    {
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
        }
    }

    int find(int x)
    {
        if (parent[x] != x)
        {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void unite(int a, int b)
    {
        int ra = find(a), rb = find(b);
        if (ra == rb)
            return;

        if (rank_[ra] < rank_[rb])
        {
            parent[ra] = rb;
        }
        else if (rank_[ra] > rank_[rb])
        {
            parent[rb] = ra;
        }
        else
        {
            parent[rb] = ra;
            rank_[ra]++;
        }
    }

private:
    vector<int> parent;
    vector<int> rank_;
};

class Solution
{
public:
    bool validPath(int n, vector<vector<int>> &edges, int source, int destination)
    {
        if (source == destination)
        {
            return true;
        }

        UnionFind uf(n);

        // Build connectivity
        for (const auto &e : edges)
        {
            uf.unite(e[0], e[1]);
        }

        return uf.find(source) == uf.find(destination);
    }
};

int main()
{
    Solution solution;

    // Test case 1: Simple path exists
    vector<vector<int>> edges1 = {{0, 1}, {1, 2}, {2, 3}};
    bool result1 = solution.validPath(4, edges1, 0, 3);
    printf("Test 1 (Expected: true): %s\n", result1 ? "true" : "false");

    // Test case 2: No path exists
    vector<vector<int>> edges2 = {{0, 1}, {2, 3}};
    bool result2 = solution.validPath(4, edges2, 0, 3);
    printf("Test 2 (Expected: false): %s\n", result2 ? "true" : "false");

    return 0;
}
