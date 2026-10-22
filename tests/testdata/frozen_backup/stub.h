#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#define TMS_ROCM_LEGACY_CHUNKED 0
#define SIMPLE_CHECK(x,m) do { if (!(x)) { std::cerr << m << std::endl; std::exit(1); } } while (0)
#define CURESULT_CHECK(x) assert((x)==0)
#define CUDA_ERROR_CHECK(x) assert((x)==0)
using CUdevice=int;
using CUdeviceptr=uintptr_t;
using CUmemGenericAllocationHandle=uint64_t;
using cudaError_t=int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=1;
constexpr int cudaMemcpyDeviceToHost=1, cudaMemcpyHostToDevice=2;
inline std::map<void*,bool> readonly;
inline unsigned writes=0, reads=0, releases=0;
inline std::map<std::string,std::vector<char>> disk;
inline int cudaDeviceSynchronize() {return 0;}
inline int cudaMemGetInfo(size_t* f,size_t*t) {*f=*t=1000000000;return 0;}
inline int cudaMallocHost(void**p,size_t n) {*p=std::malloc(n);return 0;}
inline int cudaFreeHost(void*p) {std::free(p);return 0;}
inline int cudaMemcpy(void*d,const void*s,size_t n,int) {std::memcpy(d,s,n);return 0;}
inline int cuMemAddressReserve(CUdeviceptr*p,size_t n,int,int,int) {*p=(uintptr_t)std::malloc(n);return 0;}
inline int cuMemAddressFree(CUdeviceptr p,size_t) {readonly.erase((void*)p);std::free((void*)p);return 0;}
inline int cuMemMap(CUdeviceptr p,size_t n,int,uint64_t,int) {std::memset((void*)p,0xcc,n);return 0;}
inline int cuMemUnmap(CUdeviceptr p,size_t n) {std::memset((void*)p,0xdd,n);return 0;}
inline int cuMemRelease(uint64_t) {return 0;}
namespace CUDAUtils {
inline CUdevice cu_ctx_get_device() {return 0;}
inline size_t cu_mem_get_allocation_size(size_t n,int) {return n;}
inline int cu_mem_create(uint64_t*h,size_t,int) {*h=1;return 0;}
inline void cu_mem_set_access(void*p,size_t,int) {readonly[p]=false;}
inline void cu_mem_set_readonly(void*p,size_t,int) {readonly[p]=true;}
}
namespace APIForwarder {inline int call_real_cuda_free(void*p) {std::free(p);return 0;}}
