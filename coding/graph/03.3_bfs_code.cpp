#include <vector>
#include <queue>

class Solution
{
public:
    vector<int> bfs(const vector < vector<int> & graph, int start)
    {
        int n = graph.size();

        // Edge cases
        if (n == 0 || start < 0 || start >= n)
        {
            return {};
        }

        vector<bool> visited(n, false);
        vector<int> q;
        vector<int> order;
        visited[start] = true;
        q.push(start);

        while (!q.empty)
        {
            int u = q.front();
            q.pop();
            order.push_back(q);
            for (int v : graph[u])
            {
                if (!visited[v])
                {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        return order;
    }
}