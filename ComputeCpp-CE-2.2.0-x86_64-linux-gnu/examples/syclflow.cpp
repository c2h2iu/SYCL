#include <iostream>
#include <vector>
#include <SYCL/sycl.hpp>

#define N 10

class add;
class mul;
class add2;

class syclTask{
    friend class syclFlow;
    friend std::ostream& operator << (std::ostream&, const syclTask&);

public:
    syclTask() = default;
    
    syclTask(const syclTask&) = default;

    syclTask& operator = (const syclTask&) = default;

    template <typename... Ts>
    syclTask& precede(Ts&&... tasks);
    
    template <typename... Ts>
    syclTask& succeed(Ts&&... tasks);
    
    syclTask& name(const std::string& name);
    
    const std::string& name() const;

    size_t num_successors() const;

//    bool empty() const;

//    syclTaskType type() const;

//    void dump(std::ostream& os) const;

//private:    
//    syclTask(syclNode*);

//    syclNode* _node {nullptr};
};



//inline syclTask::syclTask(syclNode* node) : _node {node} {
//}


template <typename... Ts>
syclTask& syclTask::precede(Ts&&... tasks) {
//    (_node->_precede(tasks._node), ...);
    return *this;
}


template <typename... Ts>
syclTask& syclTask::succeed(Ts&&... tasks) {
//    (tasks._node->_precede(_node), ...);
    return *this;
}


//inline bool syclTask::empty() const {
//    return _node == nullptr;
//}


inline syclTask& syclTask::name(const std::string& name) {
//    _node->_name = name;
    return *this;
}


inline const std::string& syclTask::name() const {
    return "god";
//    return _node->_name;
}


inline size_t syclTask::num_successors() const {
    return 6;
//    return _node->_successors.size();
}


//inline syclTaskType syclTask::type() const {
//    return static_cast<syclTaskType>(_node->_handle.index());
//}


//inline void syclTask::dump(std::ostream& os) const {
//    os << "syclTask ";
//    if(_node->_name.empty()) os << _node;
//    else os << _node->_name;
//    os << " [type=" << sycl_task_type_to_string(type()) << ']';
//}


//inline std::ostream& operator << (std::ostream& os, const syclTask& st) {
//    st.dump(os);
//    return os;
//}



class syclFlow{
//friend class Executor;

private:
//    cl::sycl::handler& cgh;
    cl::sycl::queue gpuQueue{cl::sycl::host_selector{}};
//    auto device;

public:
    template <typename K, typename R, typename C>
    void parallel_for(R&& numItems, C&& callable);

    template <typename C>
    void parallel_for(C&& callable);

    //template<typename kernelname, size_t dimensions = 1, typename C>
    //syclTask parallel_for(cl::sycl::nd_range<dimensions> numWorkItems, C&& callable);

    //template<size_t dimensions = 1, typename C>
    //syclTask parallel_for(cl::sycl::range<dimensions> numWorkItems, C&& callable);

    //template<size_t dimensions = 1, typename C>
    //syclTask parallel_for(cl::sycl::nd_range<dimensions> numWorkItems, C&& callable);




//    const std::string& get_device_name(){ device.get_infor<cl::sycl::info::device::name>(); }
//    const std::string& get_platform_name(){ 
//        device.get_platform().get_info<cl::sycl::info::platform::name>(); 
//    }

    syclFlow(){ 
        std::cout << "constructor\n"; 
    }

    //syclFlow(){ device = deviceQueue.get_device(); }
};

template <typename C>
void syclFlow::parallel_for(C&& callable){
    callable(0);
}


template <typename K, typename R, typename C>
void syclFlow::parallel_for(R&& numItems, C&& callable){
    gpuQueue.submit([&](cl::sycl::handler& cgh){
        cgh.parallel_for<K>(std::forward<R>(numItems), std::forward<C>(callable));
    });
}



int main() {
  
    std::cout << "SYCL VERSION: " << CL_SYCL_LANGUAGE_VERSION << '\n';

    std::vector<int> dA(N), dB(N), dC(N);

    for(size_t i=0; i<N; ++i) {
        dA[i] = 1;
        dB[i] = 2;
        dC[i] = 0;
    }
  
    {
        cl::sycl::queue gpuQueue{cl::sycl::default_selector{}};
    
        auto device = gpuQueue.get_device();
        auto deviceName = device.get_info<cl::sycl::info::device::name>();
        std::cout << "running vector-add on device: " << deviceName << '\n';

        cl::sycl::buffer<int, 1> bufA(dA.data(), cl::sycl::range<1>(dA.size()));
        cl::sycl::buffer<int, 1> bufB(dB.data(), cl::sycl::range<1>(dB.size()));
        cl::sycl::buffer<int, 1> bufC(dC.data(), cl::sycl::range<1>(dC.size()));

        gpuQueue.submit([&](cl::sycl::handler& cgh){
            auto inA = bufA.get_access<cl::sycl::access::mode::read>(cgh);
            auto inB = bufB.get_access<cl::sycl::access::mode::read>(cgh);
            auto out = bufC.get_access<cl::sycl::access::mode::write>(cgh);
       
            cgh.parallel_for<add>(cl::sycl::range<1>(dA.size()), [=](cl::sycl::id<1> i) {
                out[i] = inA[i] + inB[i];
            });
        });
        
        syclFlow sf;
        sf.parallel_for<mul>(cl::sycl::range<1>(dA.size()), [=](cl::sycl::id<1> i) {
        });
    }

  /*
  auto Task = taskflow.emplace([&] (tf::syclFlow& sf) {

    auto inA = bufA.get_access<cl::sycl::access::mode::read>(sf);
    auto inB = bufA.get_access<cl::sycl::access::mode::read>(sf);
    auto out = bufA.get_access<cl::sycl::access::mode::write>(sf);

    auto task1 = sf.parallel_for<add>(c::sycl::range<1>(dA.size()), [=](cl::sycl::id<1> i) {
       out[i] = inA[i] + inB[i]; 
    });
    
    auto task2 = sf.parallel_for<add>(c::sycl::range<1>(dA.size()), [=](cl::sycl::id<1> i) {
       out[i] = inA[i] + inB[i] + out[i]; 
    });

    task2.precede(task1);
  });
  */

    //syclFlow sf;
    //sf.parallel_for<add2>(cl::sycl::range<1>(dA.size()), [=](cl::sycl::id<1> i) {
    //});

    bool correct = true;
    for(int i=0; i<N; i++) {
        if(dC[i] != dA[i] + dB[i]) {
            correct = false;
        }
    }

    std::cout << (correct ? "result is correct" : "result is incorrect") << std::endl;

    return 0;
}



