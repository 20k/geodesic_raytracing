#ifndef PRINT_HPP_INCLUDED
#define PRINT_HPP_INCLUDED

#include <iostream>
#include <stdio.h>

inline bool should_debug = true;

template<typename... T>
inline
void print(T&&... t)
{
    if(should_debug)
    {
        printf(t...);
    }
}

template<typename... T>
inline
void printj(T&&... t)
{
    if(should_debug)
    {
        (std::cout << ... << t) << "\n";
    }
}

#endif // PRINT_HPP_INCLUDED
