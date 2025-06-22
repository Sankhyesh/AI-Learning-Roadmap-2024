# Binary Search Concepts - Study Notes

## 📚 Chapter 1: Binary Search Fundamentals

### 🎯 **What is Binary Search?**
Binary Search is an efficient algorithm for finding a target value in a **sorted array** by repeatedly dividing the search interval in half.

### ⚡ **Key Requirements**
1. **Sorted Array**: Array must be sorted (ascending or descending)
2. **Random Access**: Must be able to access elements by index in O(1) time
3. **Comparison**: Elements must be comparable

### 🔍 **Algorithm Steps**
1. Compare target with middle element
2. If target equals middle → Found! Return index
3. If target < middle → Search left half
4. If target > middle → Search right half
5. Repeat until found or search space is empty

### 📊 **Time Complexity**
- **Best Case**: O(1) - target is at middle
- **Average Case**: O(log n) 
- **Worst Case**: O(log n) - target not found or at ends
- **Space Complexity**: O(1) for iterative, O(log n) for recursive

### 💡 **Key Insights**
- Eliminates half of remaining elements in each step
- Much faster than linear search for large datasets
- Forms the basis for many advanced algorithms
- Can be applied to problems beyond simple searching

### 🔧 **Implementation Patterns**

#### **Iterative Approach**
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found
```

#### **Recursive Approach**
```python
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

### 🎯 **Common Variations**
1. **Find First/Last Occurrence**: When duplicates exist
2. **Find Floor/Ceil**: Closest smaller/larger element
3. **Search in Rotated Array**: Modified binary search
4. **Peak Element**: Find local maximum
5. **Search in 2D Matrix**: Extended to 2D space

### 🚨 **Common Pitfalls**
- **Integer Overflow**: Use `mid = left + (right - left) // 2`
- **Infinite Loops**: Ensure loop termination conditions
- **Off-by-One Errors**: Careful with boundary conditions
- **Unsorted Array**: Algorithm fails on unsorted data

### 📝 **Practice Problems**
1. Basic binary search implementation
2. Find first and last position of element
3. Search in rotated sorted array
4. Find peak element
5. Square root using binary search

### 🔄 **Review Schedule**
- **Next Review**: In 3 days (based on difficulty: medium)
- **Total Reviews**: 0
- **Mastery Level**: Beginner

---

**File**: `00_binary_search_concepts_notes.md`  
**Category**: Coding/DSA  
**Difficulty**: Medium  
**Chapter**: 1 of Binary Search Mastery