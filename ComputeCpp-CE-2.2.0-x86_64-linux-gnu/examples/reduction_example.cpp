/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  reduction.cpp
 *
 *  Description:
 *    Example of a reduction operation in SYCL.
 *
 **************************************************************************/

#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include "reduction.hpp"




bool isPowerOfTwo(unsigned int x){
  /* If x is a power of two, x & (x - 1) will be nonzero. */
    return ((x != 0) && !(x & (x - 1)));
}


int main(int argc, char* argv[]){
    unsigned int N = std::stoi(argv[1]);
    int init = 100;
    //const unsigned N = 1048576u;

    if(!isPowerOfTwo(N)){
        std::cout << "The SYCL reduction example only works with vector sizes Power of Two \n";
        return 1;
    }

    std::random_device hwRand;
    std::ranlux48 rand(hwRand());
    std::uniform_int_distribution<int> dist(10, 150);
  
    auto f = std::bind(dist, rand);

    std::vector<int> v(N);
    std::generate(v.begin(), v.end(), f);

    auto binaryop = [](unsigned a, unsigned b){ return a + b; };

    //auto resSycl = sycl_reduce(v, init, [=](unsigned a, unsigned b){ return a<b?a:b; });
    auto resSycl = chiu::sycl_reduce(v, init, binaryop);
    std::cout << "SYCL Reduction result: " << resSycl << '\n';

    //auto resStl = std::accumulate(std::begin(v), std::end(v), init, [=](unsigned a, unsigned b){ return a<b?a:b; });
    auto resStl = std::accumulate(std::begin(v), std::end(v), init, binaryop);
    std::cout << " STL Reduction result: " << resStl << '\n';

    assert(resSycl == resStl);
    std::cout << "Result is correct!\n";

    return 0;
}
