/*************************************************************************
	> File Name: main.cpp
	> Author: 
	> Mail: 
	> Created Time: Sun 01 Oct 2023 11:30:15 AM CST
 ************************************************************************/

#include<iostream>
#include "swap.h"
using namespace std;

int main()
{
    int val1 = 10;
    int val2 = 20;

    cout << " Before swap:" << endl;
    cout << "val1 = " << val1 << endl;
    cout << "val2 = " << val2 << endl;
    swap(val1, val2);
    cout << " After swap:" << endl;
    cout << "val1 = " << val1 << endl;
    cout << "val2 = " << val2 << endl;

    return 0;
}
