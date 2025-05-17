/*
n : total number of vertivves labeled  0, throuhg  n- 1
edges list paris [u,v]
source stargign vertex
destinatin the target vertex
\
GOAL: we  need to determine weather there any paths (via one more edges)
that takses use from source to destination

check conneectivity
*/

/*
1. Intialise for i in 0 to n-1, set parent[i] = i, rank[i] = 0
2. proces edges:
    for each (u,v)
     ru = find(u), rv = find(v)
     if ru!=rv attach smaller rank root under larger - rank; if ranks equal increment one
3. check connectivity:
    if source == destination return tru immedieately
*/

class UnionFind
{
public:
    UnionFind(int n) : parent(n), rank_(n, 0)
    {
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
        }

        int find(int x)
        {
            if (parent[x] != x)
            {
                parent[x] = find(parent[x]);
            }

            return parent[x];
        }

        // union by rank

        void unite(int a, int b)
        {
            int ra = find(a), rb = find(b);
            if (ra == rb)
                return;

            if (rank_[ra] < rank_[rb])
            {
                parent[ra] = rb;
            }
            else if ((rank_[ra] > rank_[rb]))
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
    }

    class Solution
    {
    public:
        bool validPath(int n,
                       vector<vector<int>> &edges,
                       int source,
                       int destination)
        {

            //
            if (source == destination)
            {
                return true;
            }

            UnionFind uf(n);

            // building connectivity
            for (auto &e : edges)
            {
                uf.unite(e[0], e[1]);
            }

            return uf.find(source) == uf.find(destination);
        }
    }
