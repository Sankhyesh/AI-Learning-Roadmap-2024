/*
===========================================================================
1. Introduction and Basic Program Structure
===========================================================================
This section demonstrates a minimal C++ program.

Key Points:
- #include <iostream>: Instructs the preprocessor to include the Input/Output stream library.
- int main(): The main entry point of the program.
- std::cout: Used to print output to the console.
- std::endl: Inserts a newline character and flushes the output stream.
- return 0: Signals successful program termination.
*/

// #include <iostream>

// int main(){
//     std::cout << "hello world" << std::endl;
//     return 0;
// }

/*
===========================================================================
2. Variables, Data Types, and Modifiers
===========================================================================
C++ provides various data types and modifiers.
Data Types:
- int       : Typically 4 bytes on 32-bit systems; may be 8 bytes on 64-bit systems.
- char      : Always 1 byte.
- float     : 4 bytes.
- double    : 8 bytes.
- bool      : 1 byte.

Modifiers:
- unsigned, signed, short, long help define the range of a variable.
  * unsigned int: 0 to 2^32 - 1 (e.g., 0 to 4294967295).
  * signed int  : -2^31 to 2^31 - 1 (e.g., -2147483648 to 2147483647).
  * short int   : Typically 2 bytes.
  * long        : Typically 4 bytes on 32-bit systems.
  * long long   : Typically 8 bytes.

Additional notes:
- 2^32 - 1 equals 4294967295.
- 2^31 - 1 equals 2147483647.
- 8 bytes equals 64 bits; hence, 2^64 - 1 equals 18446744073709551615.
*/

/*
Example demonstrating different data types:

#include <iostream>
int main(){
    int a = 2147483648;    // Warning: May overflow a 32-bit int
    long b = 2147483648;   // Valid if 'long' can handle 2147483648
    double c = 3.14;
    char d = 'A';
    bool e = true;
    long long f = 9223372036854775807; // Maximum value for long long

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
    std::cout << "d: " << d << std::endl;
    std::cout << "f: " << f << std::endl;
    return 0;
}
*/

/*
===========================================================================
3. Operators in C++
===========================================================================
C++ supports a wide variety of operators grouped as follows:
1. Arithmetic Operators:
    +, -, *, /, %
2. Relational Operators:
    ==, !=, >, <, >=, <=
3. Logical Operators:
    && (AND), || (OR), ! (NOT)
4. Bitwise Operators:
    &, |, ^, ~, <<, >>
5. Assignment Operators:
    =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=
6. Increment/Decrement Operators:
    ++, --
7. Conditional (Ternary) Operator:
    ? :
8. Comma Operator:
    ,
9. sizeof Operator:
    sizeof()
10. Pointer Operators:
    * (dereference), & (address-of)
11. Member Selection Operators:
    . (direct access), -> (pointer access)
12. Type Cast Operator:
    (type)
13. Bitwise Shift Operators:
    <<, >>

Usage Example:
*/
/*
#include <iostream>
int main() {
    int a = 10, b = 20;
    int sum = a + b;                    // Arithmetic operation
    bool isEqual = (a == b);              // Relational check
    bool logicalResult = (a < b && b > 15); // Logical operation
    int bitwiseAnd = a & b;               // Bitwise operation
    a += 5;                             // Assignment operation
    a++;                                // Increment operator
    int max = (a > b) ? a : b;            // Ternary operator

    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Is Equal: " << isEqual << std::endl;
    std::cout << "Logical AND: " << logicalResult << std::endl;
    std::cout << "Bitwise AND: " << bitwiseAnd << std::endl;
    std::cout << "Max: " << max << std::endl;

    return 0;
}
*/

// 10 in bits = 0000 1010
// 20 in bits = 0001 0100
// 10 & 20 = 0000 0000 = 0
// 10 | 20 = 0001 1110 = 30
// 10 ^ 20 = 0001 1110 = 30 # XOR
// ~10 = 1111 0101 = -11 # 2's complement
// 10 << 2 = 0010 1000 = 40 # left shift
// 10 >> 2 = 0000 0010 = 2 # right shift

// use of the reference operator

/*
#include <iostream>

int func(int &a){
    a = 40;
    return a;
}

int main(){
    // use of the reference operator
    int a = 10;
    int &b = a; // b is a reference to a
    b = 20; // this will change the value of a
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    func(a);
    std::cout << "a: " << a << std::endl;

    // use of the * operator
    int *p = &a; // p is a pointer to a
    std::cout << "p: " << p << std::endl; // printing the address of a
    std::cout << "*p: " << *p << std::endl; // dereferencing the pointer

    return 0;
}

*/

// control structures in C++
// if else statement

/* #include <iostream>
using namespace std;
int main(){
    int a = 10;

    if(a>10){
        cout << "a is greater than 10" << endl;
    }
    else if(a==10){
        cout << "a is equal to 10" << endl;
    }
    else{
        cout << "a is less than 10" << endl;
    }
    return 0;
}
 */

// loops in C++
/*
#include <iostream>
using namespace std;

int main(){
    int i = 0;
    while(i<5){
        cout << i<< endl;
        i++;
    }

    for(int j = 0; j<5; j++){
        cout<<j<<endl;
    }
    int j = 0;
    do{
        cout<<j<<endl;
        j++;
    }while(j<5);

}
 */

// switch case statement in C++
/*
#include <iostream>
using namespace std;

int main(){
    int a = 2;\
    switch(a){
        case 1:
            cout << "a is 1" << endl;
            break;
        case 2:
            cout << "a is 2" << endl;
            break;
        case 3:
            cout << "a is 3" << endl;
            break;
        default:
            cout << "a is not 1, 2 or 3" << endl;
    }

    return 0;
} */

// functions in C++
/*
#include <iostream>
using namespace std;


int add(int a,int  b){
    return a + b;
}
int main(){
    int a = 10;
    int b = 20;
    int sum = add(a, b);
    cout << "Sum: " << sum << endl;
    return 0;
}
 */

// function overloading in C++
// function overloading is a feature in C++ where two or more functions can have the same name but different parameters.
// function overloading is a compile-time polymorphism
// function overloading is also known as static polymorphism
/*
#include <iostream>
using namespace std;
int add(int a, int b){
    return a + b;
}
int add(int a, int b, int c){
    return a + b + c;
}
int main(){
    int a = 10;
    int b = 20;
    int c = 30;
    cout << "Sum: " << add(a, b) << endl;
    cout << "Sum: " << add(a, b, c) << endl;
    return 0;
} */

// arrays in C++
// arrays are used to store multiple values in a single variable
/* #include <iostream>
using namespace std;

int main(){

    int arr[5] = {1, 2, 3, 4, 5};
    cout << "arr[0]: " << arr[0] << endl;
    cout << "arr[1]: " << arr[1] << endl;
    arr[0] = 10;
    cout << "arr[0]: " << arr[0] << endl;
    return 0;
} */

// strings in C++
// strings are used to store text
// strings are objects of the string class
// strings are defined using double quotes
// strings are immutable
// strings can be concatenated using the + operator
// strings can be concatenated using the append() method
// strings can be accessed using the [] operator
// strings can be accessed using the at() method
// strings can be modified using the insert() method
// strings can be modified using the erase() method
// strings can be modified using the replace() method
// strings can be found using the find() method
// strings can be found using the rfind() method
// strings can be found using the substr() method
// strings can be found using the compare() method
// strings can be found using the size() method
// strings can be found using the length() method
// strings can be found using the empty() method
// strings can be found using the clear() method
// strings can be found using the push_back() method
// strings can be found using the pop_back() method
// strings can be found using the resize() method
// strings can be found using the capacity() method
// strings can be found using the reserve() method
// strings can be found using the shrink_to_fit() method
// strings can be found using the copy() method
// strings can be found using the c_str() method
// strings can be found using the data() method
// strings can be found using the swap() method
// strings can be found using the getline() method
// strings can be found using the find_first_of() method
// strings can be found using the find_last_of() method
// strings can be found using the find_first_not_of() method
// strings can be found using the find_last_not_of() method
// strings can be found using the stoi() method
// strings can be found using the stol() method
// strings can be found using the stoul() method
// strings can be found using the stoll() method
// strings can be found using the stof() method
// strings can be found using the stod() method
// strings can be found using the stold() method
// strings can be found using the to_string() method
// strings can be found using the to_wstring() method
// strings can be found using the atoll() method
/*
#include <iostream>
using namespace std;

int main()
{
    string str = "hello";
    cout << "str: " << str << endl;
    cout << "str[0]: " << str[0] << endl; // same as at()
    cout << "str.at(0): " << str.at(0) << endl; // same as []

    cout << "str.size(): " << str.size() << endl; // same as length()
    cout << "str.length(): " << str.length() << endl; // same as size()
    cout << "str.empty(): " << str.empty() << endl; // 0 = false, 1 = true
    cout << "str: " << str << endl;
    str = "hello";
    cout << "str: " << str << endl;
    cout << "str.append(\" world\"): " << str.append(" world") << endl; // hello world
    cout << "str: " << str << endl; // hello world
    cout << "str.insert(5, \" there\"): " << str.insert(5, " there") << endl; // hello there world
    cout << "str: " << str << endl;
    cout << "str.erase(5, 5): " << str.erase(5, 5) << endl; // hello world
    cout << "str: " << str << endl; // hello world
    cout << "str.replace(5, 5, \"\"): " << str.replace(5, 5, "") << endl; // hello
    cout << "str replace: " << str.replace(2,2, "llo") << endl; // hello
    cout << "str: " << str << endl;
    cout << "str.find(\"world\"): " << str.find("world") << endl;
    cout << "str.rfind(\"world\"): " << str.rfind("world") << endl; //
    cout << "str.substr(0, 5): " << str.substr(0, 5) << endl;
    cout << "str.compare(\"hello\"): " << str.compare("hello") << endl;
    cout << "str.compare(\"hello world\"): " << str.compare("hello world") << endl;
    cout << "str.compare(\"hello there\"): " << str.compare("hello there") << endl;
    cout << "str.capacity(): " << str.capacity() << endl;

    return 0;
} */

//  vectors in C++
// vectors are used to store multiple values in a single variable
// vectors are dynamic arrays
// vectors are part of the STL (Standard Template Library)
// vectors are defined in the vector header file
// vectors are created using the vector class
// vectors are accessed using the [] operator
// vectors are accessed using the at() method
// vectors are modified using the push_back() method
// vectors are modified using the pop_back() method
// vectors are modified using the insert() method
// vectors are modified using the erase() method
// vectors are modified using the clear() method
// vectors are modified using the resize() method
// vectors are modified using the swap() method
// vectors are modified using the assign() method
// vectors are modified using the emplace() method
// vectors are modified using the emplace_back() method
// vectors are modified using the front() method
// vectors are modified using the back() method
// vectors are modified using the begin() method
// vectors are modified using the end() method
// vectors are modified using the rbegin() method
// vectors are modified using the rend() method
// vectors are modified using the cbegin() method
// vectors are modified using the cend() method
// vectors are modified using the crbegin() method
// vectors are modified using the crend() method
// vectors are modified using the empty() method
// vectors are modified using the size() method
// vectors are modified using the capacity() method
// vectors are modified using the reserve() method
// vectors are modified using the shrink_to_fit() method
// vectors are modified using the assign() method
// vectors are modified using the insert() method
// vectors are modified using the erase() method
// vectors are modified using the swap() method
// vectors are modified using the emplace() method
// vectors are modified using the emplace_back() method
// vectors are modified using the clear() method
// vectors are modified using the resize() method

/* #include <iostream>
#include <vector>
using namespace std;

int main(){
    vector<int> vec = {1,2,3,4,5,6};
    cout << "vec[0]: " << vec[0] << endl;
    cout << "vec.at(0): " << vec.at(0) << endl;
    cout << "vec.size(): " << vec.size() << endl;
    cout << "vec.capacity(): " << vec.capacity() << endl;
    cout << "vec.empty(): " << vec.empty() << endl;
    vec.push_back(7);
    cout << "vec.push_back(7): " << vec[6] << endl;
    vec.pop_back();
    cout << "vec.pop_back(): " << vec[5] << endl;
    vec.insert(vec.begin() + 2, 10);
    cout << "vec.insert(vec.begin() + 2, 10): " << vec[2] << endl;
  // how begin works?
    // about begin() and end() functions
    // begin() returns an iterator pointing to the first element in the vector
    // end() returns an iterator pointing to the last element in the vector
    // print the vector
    for(auto i = vec.begin(); i != vec.end(); i++){
        cout << *i << " ";
    }
    cout << endl;
    vec.erase(vec.begin() + 2);
    cout << "vec.erase(vec.begin() + 2): " << vec[2] << endl;
    return 0;
} */

// pointers in the c++
// pointers are used to store the address of variables
// pointers are used to store the address of functions
// pointers are used to store the address of arrays
// pointers are used to store the address of objects
// pointers are used to store the address of classes
// pointers are used to store the address of structures
// pointers are used to store the address of unions
// pointers are used to store the address of other pointers
// pointers are used to store the address of the first element of an array
// pointers are used to store the address of the last element of an array
// pointers are used to store the address of the next element of an array
// pointers are used to store the address of the previous element of an array
// pointers are used to store the address of the current element of an array

/* #include <iostream>
using namespace std;

int main(){
    int a = 10;
    int *p = &a;
    cout << "a: " << a << endl;
    cout << "p: " << p << endl;
    cout << "*p: " << *p << endl;
    *p = 20;
    cout << "a: " << a << endl;
    cout << "p: " << p << endl;
    cout << "*p: " << *p << endl;
    return 0;
} */

// Object-Oriented Programming (OOP) in C++
// OOP is a programming paradigm that uses objects and classes
// OOP is based on the concept of objects
// OOP is based on the concept of classes
// OOP is based on the concept of inheritance
// OOP is based on the concept of polymorphism
// OOP is based on the concept of encapsulation
// OOP is based on the concept of abstraction
// OOP is based on the concept of data hiding
// OOP is based on the concept of modularity
// OOP is based on the concept of reusability
// OOP is based on the concept of extensibility
// OOP is based on the concept of maintainability
// OOP is based on the concept of flexibility
// OOP is based on the concept of scalability
// OOP is based on the concept of security
// OOP is based on the concept of efficiency
// OOP is based on the concept of reliability
// OOP is based on the concept of robustness
// OOP is based on the concept of portability
// OOP is based on the concept of simplicity
// OOP is based on the concept of readability
// OOP is based on the concept of maintainability

/* #include <iostream>
using namespace std;

class Car{
    public: // access specifier
        string brand; // attribute
        string model; // attribute
        int year; // attribute
        // default constructor
        Car(){
            cout << "Car object created" << endl;
        }
        // parameterized constructor
        Car(string x, string y, int z){ // constructor
            brand = x;
            model = y;
            year = z;
        }
        void print(){ // method
            cout << brand << " " << model << " " << year << endl;
        }
};

int main(){
    Car car1;
    Car car2("BMW", "X5", 1999);
    car1.brand = "Ford";
    car1.model = "Mustang";
    car1.year = 1969;
    car1.print();
    car2.print();
    return 0;
}

 */
// Inheritance in C++
// Inheritance is a feature in C++ where a class inherits properties and methods from another class
// Inheritance is used to create a new class from an existing class
// Inheritance is used to create a parent-child relationship between classes
// Inheritance is used to create a base class and a derived class
// Inheritance is used to create a superclass and a subclass
// Inheritance is used to create a parent class and a child class

/* #include <iostream>
using namespace std;

class Vehicle{
    public:
        string brand = "Ford";
        void honk(){
            cout << "Tuut, tuut!" << endl;
        }
};

class Car: public Vehicle{ // Car is the child class of Vehicle,
    // Vehicle is the parent class of Car, public Vehicle means that the public members
    // of the Vehicle class are accessible in the Car class
    // default access specifier is private
    // private members of the Vehicle class are not accessible in the Car class
    // protected members of the Vehicle class are accessible in the Car class

    public:
        string model = "Mustang";
};


int main(){
    Car myCar;
    myCar.honk();
    cout << myCar.brand + " " + myCar.model << endl;
    return 0;
}

 */

//  Advanced Topics & Best Practices

// 1. Namespaces in C++
// Namespaces are used to organize code into logical groups and prevent naming conflicts

// stl algorithms and iterators
// STL (Standard Template Library) is a powerful library in C++ that provides various algorithms and data structures
// STL algorithms are used to perform operations on containers
// STL iterators are used to iterate over containers
// STL containers are used to store data
// STL functions are used to perform operations on containers
// sort and find are examples of STL algorithms

/* #include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main(){
    vector<int> vec = {1,2,3,4,5,6};
    sort(vec.begin(), vec.end());
    for(auto i = vec.begin(); i != vec.end(); i++){
        cout << *i << " ";
    }
    cout << endl;
    auto it = find(vec.begin(), vec.end(), 3);
    if(it != vec.end()){
        cout << "Element found: " << *it << endl;
    }
    else{
        cout << "Element not found" << endl;
    }
    return 0;
}
 */

// Memory Management
// Memory management is the process of allocating and deallocating memory
// Memory management is important to prevent memory leaks and optimize performance
// Memory management is important to prevent memory fragmentation and optimize memory usage
// Memory management is important to prevent memory corruption and optimize

/* #include <iostream>
#include <memory>
using namespace std;

int main()
{
    unique_ptr<int> p1(new int(10));
    cout << *p1 << endl;
    // no need to delete the memory
    shared_ptr<int> p2(new int(20));
    cout << *p2 << endl;
    // no need to delete the memory
    weak_ptr<int> p3(p2);
    cout << *p3.lock() << endl;
    // no need to delete the memory
    return 0;
}
 */