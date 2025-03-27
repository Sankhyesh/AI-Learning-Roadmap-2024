 

# **Coding Problem Solving Guide**

This guide outlines strategies for selecting algorithms and data structures based on problem characteristics. Use it as a quick reference for coding interviews or problem-solving practice.

---

## **If the input array is sorted:**
- **Binary Search**: Find a target value efficiently in a sorted array.
- **Two Pointers**: Solve problems like finding pairs that sum to a target.

---

## **If asked for all permutations/subsets:**
- **Backtracking**: Systematically generate all possible permutations or subsets.

---

## **If given a tree:**
- **Depth-First Search (DFS)**: Explore the tree by going as deep as possible along each branch.
- **Breadth-First Search (BFS)**: Explore the tree level by level.

---

## **If given a graph:**
- **Depth-First Search (DFS)**: Traverse the graph by exploring as far as possible along each path.
- **Breadth-First Search (BFS)**: Traverse the graph by exploring all neighbors level by level.

---

## **If given a linked list:**
- **Two Pointers**: Use techniques like fast and slow pointers to detect cycles or intersections.

---

## **If recursion is banned:**
- **Stack**: Use a stack to simulate recursive calls iteratively.

---

## **If must solve in-place:**
- **Swap Corresponding Values**: Modify the input by swapping elements to avoid extra space.
- **Store One or More Different Values in the Same Pointer**: Use techniques like bit manipulation to save space.

---

## **If asked for maximum/minimum subarray/subset/options:**
- **Dynamic Programming**: Solve by breaking the problem into overlapping subproblems.
- **Sliding Window**: Use a moving window to compute results efficiently.

---

## **If asked for top/least K items:**
- **Heap**: Maintain a priority queue to track the top or least K elements.
- **QuickSelect**: Find the k-th smallest or largest element in linear average time.

---

## **If asked for common strings:**
- **Map**: Use a hash map to track string frequencies or presence.
- **Trie**: Store and retrieve strings with shared prefixes efficiently.

---

## **Else (general cases):**
- **Map/Set for O(1) time & O(n) space**: Use hash maps or sets for fast lookups with linear space.
- **Sort input for O(n log n) time and O(1) space**: Sort the input to simplify the problem with minimal extra space.
 