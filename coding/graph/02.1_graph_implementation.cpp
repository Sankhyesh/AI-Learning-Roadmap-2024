#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

void print_graphs_adjMatrix(vector<vector<int>> adjMatrix)
{
    for (int i = 1; i < adjMatrix.size(); i++)
    {
        cout << "Node " << i << " is connected to : ";
        for (int j = 1; j <= 4; j++)
        {
            if (adjMatrix[i][j] == 1)
            {
                cout << j << " ";
            }
        }
        cout << endl;
    }
}

void print_graph_ajs_list(unordered_map<int, vector<int>> adjList)
{
    for (auto a : adjList)
    {
        cout << "Node " << a.first << "is connected to: ";
        for (auto b : a.second)
        {
            cout << b << " ";
        }
        cout << endl;
    }
}

int main()
{

    vector<vector<int>> edgeList = {
        {1, 2},
        {1, 3},
        {2, 4},
        {3, 4}};
    vector<vector<int>> adjMatrix(5, vector<int>(5, 0));
    for (int i = 0; i < edgeList.size(); i++)
    {
        int u = edgeList[i][0];
        int v = edgeList[i][1];
        adjMatrix[u][v] = 1;
        adjMatrix[v][u] = 1;
    }
    print_graphs_adjMatrix(adjMatrix);

    unordered_map<int, vector<int>> adjList;
    for (int i = 0; i < edgeList.size(); i++)
    {
        int u = edgeList[i][0];
        int v = edgeList[i][1];
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }
    print_graph_ajs_list(adjList);

    return 0;
}