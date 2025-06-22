#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <cassert>
using namespace std;

// Structure to represent a weighted edge
struct Edge {
    int dest;
    int weight;
    Edge(int d, int w) : dest(d), weight(w) {}
};

// Function to find shortest path using Dijkstra's algorithm
vector<int> dijkstra(vector<vector<Edge>>& graph, int source) {
    int V = graph.size();
    vector<int> dist(V, INT_MAX);
    dist[source] = 0;
    
    // Priority queue to store vertices with their distances
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, source});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        
        // If we've already found a shorter path, skip
        if (d > dist[u]) continue;
        
        // Check all neighbors of u
        for (const Edge& edge : graph[u]) {
            int v = edge.dest;
            int weight = edge.weight;
            
            // If we found a shorter path to v through u
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    
    return dist;
}

// Test cases
void runTests() {
    // Test Case 1: Simple graph with 3 vertices
    {
        vector<vector<Edge>> graph(3);
        graph[0].push_back(Edge(1, 1));
        graph[0].push_back(Edge(2, 4));
        graph[1].push_back(Edge(2, 2));
        
        vector<int> distances = dijkstra(graph, 0);
        assert(distances[0] == 0);
        assert(distances[1] == 1);
        assert(distances[2] == 3);
        cout << "Test Case 1 passed!" << endl;
    }
    
    // Test Case 2: Graph with disconnected components
    {
        vector<vector<Edge>> graph(4);
        graph[0].push_back(Edge(1, 1));
        graph[2].push_back(Edge(3, 1));
        
        vector<int> distances = dijkstra(graph, 0);
        assert(distances[0] == 0);
        assert(distances[1] == 1);
        assert(distances[2] == INT_MAX);
        assert(distances[3] == INT_MAX);
        cout << "Test Case 2 passed!" << endl;
    }
    
    // Test Case 3: Graph with negative weights (should still work correctly)
    {
        vector<vector<Edge>> graph(3);
        graph[0].push_back(Edge(1, -1));
        graph[1].push_back(Edge(2, -2));
        
        vector<int> distances = dijkstra(graph, 0);
        assert(distances[0] == 0);
        assert(distances[1] == -1);
        assert(distances[2] == -3);
        cout << "Test Case 3 passed!" << endl;
    }
    
    // Test Case 4: Empty graph
    {
        vector<vector<Edge>> graph(0);
        vector<int> distances = dijkstra(graph, 0);
        assert(distances.empty());
        cout << "Test Case 4 passed!" << endl;
    }
}

int main() {
    // Run test cases
    runTests();
    
    // Example graph with 5 vertices
    int V = 5;
    vector<vector<Edge>> graph(V);
    
    // Add edges to the graph
    graph[0].push_back(Edge(1, 4));
    graph[0].push_back(Edge(2, 2));
    graph[1].push_back(Edge(2, 1));
    graph[1].push_back(Edge(3, 5));
    graph[2].push_back(Edge(3, 8));
    graph[2].push_back(Edge(4, 10));
    graph[3].push_back(Edge(4, 2));
    
    // Find shortest paths from vertex 0
    vector<int> distances = dijkstra(graph, 0);
    
    // Print the shortest distances
    cout << "\nExample graph shortest distances from vertex 0:\n";
    for (int i = 0; i < V; i++) {
        cout << "To vertex " << i << ": " << distances[i] << endl;
    }
    
    return 0;
}