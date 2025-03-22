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



