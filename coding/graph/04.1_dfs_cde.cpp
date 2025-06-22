

#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    void dfs(int node, const vector<vector<int>> &adj, vector<bool> &visited, vector<int> &result)
    {
        visited[node] = true;
        result.push_back(node);

        for (int neighbours : adj[node])
        {
            if (!visited[neighbours])
            {
                dfs(neighbours, adj, visited, result);
            }
        }
    }

    vector<int> dfsTraversal(int v, const vector<vector<int>> &adj)
    {
        vector<bool> visited(v, false);
        vector<int> result;
        for (int i = 0; i < v; ++i)
        {
            if (!visited[i])
            {
                dfs(i, adj, visited, result);
            }
        }
    }
}