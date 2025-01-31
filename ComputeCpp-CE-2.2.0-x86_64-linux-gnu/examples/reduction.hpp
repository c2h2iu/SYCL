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

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>



namespace chiu{

    template<typename T>
    class sycl_reduction;


    template <typename T, typename I, typename C>
    T sycl_reduce(const std::vector<T>& v, I& init, C bop){
        T retVal;

        {
            cl::sycl::queue q([=](cl::sycl::exception_list eL){
                try{
                    for(auto& e : eL)    std::rethrow_exception(e);
                } catch (cl::sycl::exception ex){
                    std::cout << " There is an exception in the reduction kernel\n";
                    std::cout << ex.what() << '\n';
                }
            });

            /* Output device and platform information. */
            auto device = q.get_device();
        
            auto deviceName = device.get_info<cl::sycl::info::device::name>();
            std::cout << "Device Name: " << deviceName << '\n';
        
            auto platformName = device.get_platform().get_info<cl::sycl::info::platform::name>();
            std::cout << "Platform Name " << platformName << '\n';

            /* The buffer is used to initialise the data on the device, but we don't
             * want to copy back and trash it. buffer::set_final_data() tells the
             * SYCL runtime where to put the data when the buffer is destroyed; nullptr
             * indicates not to copy back. The vector's length is used as the global
             * work size (unless that is too large). */
            cl::sycl::buffer<int, 1> bufI(v.data(), cl::sycl::range<1>(v.size()));
            bufI.set_final_data(nullptr);
        
            size_t local = std::min(v.size(), device.get_info<cl::sycl::info::device::max_work_group_size>());
            size_t length = v.size();

            if(length == 0)    return init;
            if(length == 1)    return bop(init, v[0]);

            {
                /* Each iteration of the do loop applies one level of reduction until
                 * the input is of length 1 (i.e. the reduction is complete). */
                do{
                    auto f = [length, local, &bufI, bop](cl::sycl::handler& h) mutable{
                        cl::sycl::nd_range<1> r{ cl::sycl::range<1>{ std::max(length, local) }, cl::sycl::range<1>{ std::min(length, local) } };

                        /* Two accessors are used: one to the buffer that is being reduced,
                         * and a second to local memory, used to store intermediate data. */
                        auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
                        cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, 
                                                 cl::sycl::access::target::local> scratch(cl::sycl::range<1>(local), h);

                        /* The parallel_for invocation chosen is the variant with an nd_item
                         * parameter, since the code requires barriers for correctness. */
                        h.parallel_for<sycl_reduction<T>>(r, [aI, scratch, local, length, bop](cl::sycl::nd_item<1> id){
                            size_t globalid = id.get_global_id(0);
                            size_t localid = id.get_local_id(0);

                            /* All threads collectively read from global memory into local.
                             * The barrier ensures all threads' IO is resolved before
                             * execution continues (strictly speaking, all threads within
                             * a single work-group - there is no co-ordination between
                             * work-groups, only work-items). */
                            if(globalid < length)    scratch[localid] = aI[globalid];
            
                            id.barrier(cl::sycl::access::fence_space::local_space);

                            /* Apply the reduction operation between the current local
                             * id and the one on the other half of the vector. */
                            if(globalid < length){
                                int min = (length < local) ? length : local;
                                for(size_t offset = min / 2; offset > 0; offset /= 2){
                                    if(localid < offset)    scratch[localid] = bop(scratch[localid], scratch[localid + offset]);
                                    id.barrier(cl::sycl::access::fence_space::local_space);
                                }
                  
                                /* The final result will be stored in local id 0. */
                                if(localid == 0)    aI[id.get_group(0)] = scratch[localid];
                            }   
                        });
                    };

                    q.submit(f);

                    /* At this point, you could queue::wait_and_throw() to ensure that
                     * errors are caught quickly. However, this would likely impact
                     * performance negatively. */                
                    length = length / local;

                } while(length > 1);
            }

            {
                /* It is always sensible to wrap host accessors in their own scope as
                 * kernels using the buffers they access are blocked for the length
                 * of the accessor's lifetime. */
                auto hI = bufI.template get_access<cl::sycl::access::mode::read>();
                retVal = bop(init, hI[0]);
            }
        }

        return retVal;
    }
}
