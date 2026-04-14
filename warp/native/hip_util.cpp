// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Filippo Luca Ferretti.
// SPDX-License-Identifier: Apache-2.0

#if WP_ENABLE_HIP

#include "cuda_util.h"
#include "error.h"

#include <set>
#include <stack>

static bool hip_driver_initialized = false;

bool ContextGuard::always_restore = false;

CudaTimingState* g_cuda_timing_state = NULL;

bool init_cuda_driver()
{
    hipError_t result = hipInit(0);
    if (result != hipSuccess)
    {
        fprintf(stderr,
            "Warp HIP warning: Could not initialize HIP runtime. "
            "GPU execution will not be available.\n");
        return false;
    }
    hip_driver_initialized = true;
    return true;
}

bool is_cuda_driver_initialized() { return hip_driver_initialized; }

bool check_cuda_result(cudaError_t code, const char* func, const char* file, int line)
{
    if (code == hipSuccess)
        return true;

    wp::set_error_string(
        "Warp HIP error %u: %s (in function %s, %s:%d)",
        unsigned(code), hipGetErrorString(code), func, file, line);
    return false;
}

bool check_cu_result(CUresult result, const char* func, const char* file, int line)
{
    if (result == hipSuccess)
        return true;

    const char* errString = hipGetErrorString(result);
    if (errString)
        wp::set_error_string(
            "Warp HIP error %u: %s (in function %s, %s:%d)",
            unsigned(result), errString, func, file, line);
    else
        wp::set_error_string(
            "Warp HIP error %u (in function %s, %s:%d)",
            unsigned(result), func, file, line);

    return false;
}

bool get_capture_dependencies(CUstream stream, std::vector<CUgraphNode>& dependencies_ret)
{
    CUstreamCaptureStatus status;
    size_t num_dependencies = 0;
    const CUgraphNode* dependencies = NULL;
    dependencies_ret.clear();
    if (check_cu(cuStreamGetCaptureInfo_f(stream, &status, NULL, NULL, &dependencies, &num_dependencies)))
    {
        if (dependencies && num_dependencies > 0)
            dependencies_ret.insert(dependencies_ret.begin(), dependencies, dependencies + num_dependencies);
        return true;
    }
    return false;
}

bool get_graph_leaf_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t>& leaf_nodes_ret)
{
    if (!graph)
        return false;

    size_t node_count = 0;
    if (!check_cuda(cudaGraphGetNodes(graph, NULL, &node_count)))
        return false;

    std::vector<cudaGraphNode_t> nodes(node_count);
    if (!check_cuda(cudaGraphGetNodes(graph, nodes.data(), &node_count)))
        return false;

    leaf_nodes_ret.clear();
    for (size_t i = 0; i < node_count; i++)
    {
        size_t num_dependents = 0;
        if (check_cu(cuGraphNodeGetDependentNodes_f(nodes[i], NULL, &num_dependents)))
        {
            if (num_dependents == 0)
                leaf_nodes_ret.push_back(nodes[i]);
        }
    }
    return true;
}

CUresult cuDriverGetVersion_f(int* version)
{
    return hipDriverGetVersion(version);
}

CUresult cuGetErrorName_f(CUresult result, const char** pstr)
{
    *pstr = hipGetErrorName(result);
    return hipSuccess;
}

CUresult cuGetErrorString_f(CUresult result, const char** pstr)
{
    *pstr = hipGetErrorString(result);
    return hipSuccess;
}

CUresult cuInit_f(unsigned int flags) { return hipInit(flags); }

CUresult cuDeviceGet_f(CUdevice* dev, int ordinal) { return hipDeviceGet(dev, ordinal); }

CUresult cuDeviceGetCount_f(int* count)
{
    hipError_t r = hipGetDeviceCount(count);
    if (r != hipSuccess && count)
        *count = 0;
    return hipSuccess;
}

CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev)
{
    hipDeviceProp_t prop;
    hipError_t r = hipGetDeviceProperties(&prop, dev);
    if (r != hipSuccess)
        return r;
    strncpy(name, prop.name, len);
    name[len - 1] = '\0';
    return hipSuccess;
}

CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev)
{
    return hipDeviceGetAttribute(value, attrib, dev);
}

CUresult cuDeviceGetUuid_f(CUuuid* uuid, CUdevice dev)
{
    return hipDeviceGetUuid(uuid, dev);
}

CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev)
{
    return hipDevicePrimaryCtxRetain(ctx, dev);
}

CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev)
{
    return hipDevicePrimaryCtxRelease(dev);
}

CUresult cuDeviceCanAccessPeer_f(int* can_access, CUdevice dev, CUdevice peer_dev)
{
    return hipDeviceCanAccessPeer(can_access, dev, peer_dev);
}

CUresult cuMemGetInfo_f(size_t* free, size_t* total)
{
    return hipMemGetInfo(free, total);
}

CUresult cuCtxGetCurrent_f(CUcontext* ctx) { return hipCtxGetCurrent(ctx); }
CUresult cuCtxSetCurrent_f(CUcontext ctx) { return hipCtxSetCurrent(ctx); }
CUresult cuCtxPushCurrent_f(CUcontext ctx) { return hipCtxPushCurrent(ctx); }
CUresult cuCtxPopCurrent_f(CUcontext* ctx) { return hipCtxPopCurrent(ctx); }
CUresult cuCtxSynchronize_f() { return hipDeviceSynchronize(); }

CUresult cuCtxGetDevice_f(CUdevice* dev)
{
    return hipCtxGetDevice(dev);
}

CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev)
{
    return hipCtxCreate(ctx, flags, dev);
}

CUresult cuCtxDestroy_f(CUcontext ctx) { return hipCtxDestroy(ctx); }

CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags)
{
    return hipCtxEnablePeerAccess(peer_ctx, flags);
}

CUresult cuCtxDisablePeerAccess_f(CUcontext peer_ctx)
{
    return hipCtxDisablePeerAccess(peer_ctx);
}

CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags)
{
    return hipStreamCreateWithFlags(stream, flags);
}

CUresult cuStreamDestroy_f(CUstream stream) { return hipStreamDestroy(stream); }
CUresult cuStreamQuery_f(CUstream stream) { return hipStreamQuery(stream); }
CUresult cuStreamSynchronize_f(CUstream stream) { return hipStreamSynchronize(stream); }

CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags)
{
    return hipStreamWaitEvent(stream, event, flags);
}

CUresult cuStreamGetCtx_f(CUstream stream, CUcontext* pctx)
{
    return hipStreamGetCtx(stream, pctx);
}

CUresult cuStreamGetCaptureInfo_f(
    CUstream stream,
    CUstreamCaptureStatus* captureStatus_out,
    cuuint64_t* id_out,
    CUgraph* graph_out,
    const CUgraphNode** dependencies_out,
    size_t* numDependencies_out)
{
    return hipStreamGetCaptureInfo_v2(
        stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
}

CUresult cuStreamUpdateCaptureDependencies_f(
    CUstream stream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags)
{
    return hipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
}

CUresult cuStreamCreateWithPriority_f(CUstream* phStream, unsigned int flags, int priority)
{
    return hipStreamCreateWithPriority(phStream, flags, priority);
}

CUresult cuStreamGetPriority_f(CUstream hStream, int* priority)
{
    return hipStreamGetPriority(hStream, priority);
}

CUresult cuEventCreate_f(CUevent* event, unsigned int flags)
{
    return hipEventCreateWithFlags(event, flags);
}

CUresult cuEventDestroy_f(CUevent event) { return hipEventDestroy(event); }
CUresult cuEventQuery_f(CUevent event) { return hipEventQuery(event); }

CUresult cuEventRecord_f(CUevent event, CUstream stream)
{
    return hipEventRecord(event, stream);
}

CUresult cuEventRecordWithFlags_f(CUevent event, CUstream stream, unsigned int flags)
{
    (void)flags;
    return hipEventRecord(event, stream);
}

CUresult cuEventSynchronize_f(CUevent event)
{
    return hipEventSynchronize(event);
}

CUresult cuGraphNodeGetDependentNodes_f(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes)
{
    return hipGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
}

CUresult cuGraphNodeGetType_f(CUgraphNode hNode, CUgraphNodeType* type)
{
    return hipGraphNodeGetType(hNode, type);
}

CUresult cuModuleLoadDataEx_f(
    CUmodule* module, const void* image, unsigned int numOptions,
    CUjit_option* options, void** optionValues)
{
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return hipModuleLoadData(module, image);
}

CUresult cuModuleUnload_f(CUmodule hmod) { return hipModuleUnload(hmod); }

CUresult cuModuleGetFunction_f(CUfunction* hfunc, CUmodule hmod, const char* name)
{
    return hipModuleGetFunction(hfunc, hmod, name);
}

CUresult cuLaunchKernel_f(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra)
{
    return hipModuleLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuOccupancyMaxPotentialBlockSize_f(
    int* minGridSize, int* blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize,
    size_t dynamicSMemSize, int blockSizeLimit)
{
    return hipModuleOccupancyMaxPotentialBlockSize(
        minGridSize, blockSize, func, dynamicSMemSize, blockSizeLimit);
}

CUresult cuMemcpyPeerAsync_f(
    CUdeviceptr dst_ptr, CUcontext dst_ctx, CUdeviceptr src_ptr,
    CUcontext src_ctx, size_t n, CUstream stream)
{
    (void)dst_ctx;
    (void)src_ctx;
    return hipMemcpyAsync((void*)dst_ptr, (void*)src_ptr, n, hipMemcpyDeviceToDevice, stream);
}

CUresult cuPointerGetAttribute_f(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    return hipPointerGetAttribute(data, attribute, ptr);
}

CUresult cuGraphicsMapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream stream)
{
    return hipGraphicsMapResources(count, resources, stream);
}

CUresult cuGraphicsUnmapResources_f(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    return hipGraphicsUnmapResources(count, resources, hStream);
}

CUresult cuGraphicsResourceGetMappedPointer_f(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
    return hipGraphicsResourceGetMappedPointer((void**)pDevPtr, pSize, resource);
}

CUresult cuGraphicsGLRegisterBuffer_f(CUgraphicsResource* pCudaResource, unsigned int buffer, unsigned int flags)
{
    return hipGraphicsGLRegisterBuffer(pCudaResource, buffer, flags);
}

CUresult cuGraphicsGLRegisterImage_f(
    CUgraphicsResource* pCudaResource, unsigned int image, unsigned int target, unsigned int flags)
{
    return hipGraphicsGLRegisterImage(pCudaResource, image, target, flags);
}

CUresult cuGraphicsSubResourceGetMappedArray_f(
    CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    return hipGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
}

CUresult cuGraphicsUnregisterResource_f(CUgraphicsResource resource)
{
    return hipGraphicsUnregisterResource(resource);
}

CUresult cuModuleGetGlobal_f(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
    return hipModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuFuncSetAttribute_f(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    return hipFuncSetAttribute(hfunc, attrib, value);
}

CUresult cuIpcGetEventHandle_f(CUipcEventHandle* pHandle, CUevent event)
{
    return hipIpcGetEventHandle(pHandle, event);
}

CUresult cuIpcOpenEventHandle_f(CUevent* phEvent, CUipcEventHandle handle)
{
    return hipIpcOpenEventHandle(phEvent, handle);
}

CUresult cuIpcGetMemHandle_f(CUipcMemHandle* pHandle, CUdeviceptr dptr)
{
    return hipIpcGetMemHandle(pHandle, (void*)dptr);
}

CUresult cuIpcOpenMemHandle_f(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int flags)
{
    return hipIpcOpenMemHandle((void**)pdptr, handle, flags);
}

CUresult cuIpcCloseMemHandle_f(CUdeviceptr dptr)
{
    return hipIpcCloseMemHandle((void*)dptr);
}

CUresult cuArrayCreate_f(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray)
{
    return hipArrayCreate(pHandle, pAllocateArray);
}

CUresult cuArrayDestroy_f(CUarray hArray)
{
    return hipArrayDestroy(hArray);
}

CUresult cuArray3DCreate_f(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
    return hipArray3DCreate(pHandle, pAllocateArray);
}

CUresult cuArray3DGetDescriptor_f(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    return hipArray3DGetDescriptor(pArrayDescriptor, hArray);
}

CUresult cuMemcpy2D_f(const CUDA_MEMCPY2D* pCopy)
{
    return hipMemcpyParam2D(pCopy);
}

CUresult cuMemcpy2DAsync_f(const CUDA_MEMCPY2D* pCopy, CUstream hStream)
{
    return hipMemcpyParam2DAsync(pCopy, hStream);
}

CUresult cuMemcpy3D_f(const CUDA_MEMCPY3D* pCopy)
{
    return hipDrvMemcpy3D(pCopy);
}

CUresult cuMemcpy3DAsync_f(const CUDA_MEMCPY3D* pCopy, CUstream hStream)
{
    return hipDrvMemcpy3DAsync(pCopy, hStream);
}

CUresult cuTexObjectCreate_f(
    CUtexObject* pTexObject,
    const CUDA_RESOURCE_DESC* pResDesc,
    const CUDA_TEXTURE_DESC* pTexDesc,
    const CUDA_RESOURCE_VIEW_DESC* pResViewDesc)
{
    return hipTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

CUresult cuTexObjectDestroy_f(CUtexObject texObject)
{
    return hipTexObjectDestroy(texObject);
}

#endif // WP_ENABLE_HIP
