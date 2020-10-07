#!/bin/bash

read -p 'Filename: ' filename


# Building ComputeCpp integration header file
../bin/compute++ -sycl -O2 -mllvm -inline-threshold=1000 -intelspirmetadata -sycl-target ptx64 -std=c++17 -I"../include"	-I"/usr/include"  -sycl-ih $filename.sycl -o $filename.s -c $filename.cpp


# Building CXX object samples .o file
g++ -isystem ../include -Wall -std=c++17 -include $filename.sycl -x c++ -o $filename.o -c $filename.cpp


# Linking CXX executable 
g++ -Wall $filename.o  -o $filename -Wl,-rpath,../lib: ../lib/libComputeCpp.so


# Delete .o and .s and .sycl
rm $filename.o
rm $filename.s
rm $filename.sycl
