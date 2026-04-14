// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Filippo Luca Ferretti.
// SPDX-License-Identifier: Apache-2.0

// GPU runtime abstraction: maps CUDA types/enums to HIP equivalents so that
// cuda_util.h, cuda_util.cpp, and warp.cu can compile against either backend
// with minimal source changes.

#pragma once

#if WP_ENABLE_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// In device code, set __CUDA_ARCH__ to 700 so that existing GPU code path
// guards activate. Value 700 enables most features but stays below 800,
// which avoids CUDA-specific PTX asm paths (cp.async, etc.) that have
// regular load/store fallbacks.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 700
#endif

typedef hipDevice_t     CUdevice;
typedef hipCtx_t        CUcontext;
typedef hipStream_t     CUstream;
typedef hipEvent_t      CUevent;
typedef hipModule_t     CUmodule;
typedef hipFunction_t   CUfunction;
typedef hipDeviceptr_t  CUdeviceptr;
typedef hipError_t      CUresult;

#define CUDA_SUCCESS                     hipSuccess
#define CUDA_ERROR_NOT_READY             hipErrorNotReady
#define CUDA_ERROR_DEINITIALIZED         hipErrorDeinitialized
#define CUDA_ERROR_NOT_INITIALIZED       hipErrorNotInitialized
#define CUDA_ERROR_INVALID_CONTEXT       hipErrorInvalidContext
#define CUDA_ERROR_INVALID_VALUE         hipErrorInvalidValue
#define CUDA_ERROR_OUT_OF_MEMORY         hipErrorOutOfMemory
#define CUDA_ERROR_NOT_FOUND             hipErrorNotFound
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED hipErrorPeerAccessAlreadyEnabled

typedef hipError_t      cudaError_t;
#define cudaSuccess     hipSuccess

typedef hipStream_t     cudaStream_t;
typedef hipEvent_t      cudaEvent_t;

typedef hipDeviceAttribute_t CUdevice_attribute;
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID              hipDeviceAttributePciDomainID
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                 hipDeviceAttributePciBusId
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID              hipDeviceAttributePciDeviceId
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING         hipDeviceAttributeUnifiedAddressing
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR   hipDeviceAttributeComputeCapabilityMajor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR   hipDeviceAttributeComputeCapabilityMinor
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT       hipDeviceAttributeMultiprocessorCount
#define CU_DEVICE_ATTRIBUTE_INTEGRATED                 hipDeviceAttributeIntegrated
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK hipDeviceAttributeMaxSharedMemoryPerBlock

#define cudaMemcpyHostToHost     hipMemcpyHostToHost
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault        hipMemcpyDefault

typedef hipUUID CUuuid;

typedef hipStreamCaptureStatus CUstreamCaptureStatus;
#define CU_STREAM_CAPTURE_STATUS_NONE   hipStreamCaptureStatusNone
#define CU_STREAM_CAPTURE_STATUS_ACTIVE hipStreamCaptureStatusActive

typedef hipGraph_t          CUgraph;
typedef hipGraphNode_t      CUgraphNode;
typedef hipGraphExec_t      CUgraphExec;
typedef hipGraphNodeType    CUgraphNodeType;

typedef hipArray*               CUarray;
typedef HIP_ARRAY_DESCRIPTOR    CUDA_ARRAY_DESCRIPTOR;
typedef HIP_ARRAY3D_DESCRIPTOR  CUDA_ARRAY3D_DESCRIPTOR;
typedef hipTextureObject_t      CUtexObject;
typedef HIP_RESOURCE_DESC       CUDA_RESOURCE_DESC;
typedef HIP_TEXTURE_DESC        CUDA_TEXTURE_DESC;
typedef HIP_RESOURCE_VIEW_DESC  CUDA_RESOURCE_VIEW_DESC;

typedef hipPointer_attribute CUpointer_attribute;
#define CU_POINTER_ATTRIBUTE_CONTEXT hipPointerAttributeContext

#define CU_EVENT_DEFAULT        hipEventDefault
#define CU_EVENT_DISABLE_TIMING hipEventDisableTiming
#define CU_EVENT_BLOCKING_SYNC  hipEventBlockingSync

#define CU_STREAM_DEFAULT       hipStreamDefault
#define CU_STREAM_NON_BLOCKING  hipStreamNonBlocking

typedef hipIpcMemHandle_t   CUipcMemHandle;
typedef hipIpcEventHandle_t CUipcEventHandle;

#define CU_CTX_SCHED_AUTO 0

#define cudaGetLastError              hipGetLastError
#define cudaGetErrorString            hipGetErrorString
#define cudaMalloc                    hipMalloc
#define cudaMallocAsync               hipMallocAsync
#define cudaFree                      hipFree
#define cudaFreeAsync                 hipFreeAsync
#define cudaMallocHost                hipHostMalloc
#define cudaFreeHost                  hipHostFree
#define cudaMemcpyAsync               hipMemcpyAsync
#define cudaMemset                    hipMemset
#define cudaMemsetAsync               hipMemsetAsync
#define cudaDeviceSynchronize         hipDeviceSynchronize
#define cudaDeviceCanAccessPeer       hipDeviceCanAccessPeer
#define cudaEventElapsedTime          hipEventElapsedTime

#define cudaDeviceGetDefaultMemPool   hipDeviceGetDefaultMemPool
#define cudaMemPoolSetAttribute       hipMemPoolSetAttribute
#define cudaMemPoolGetAttribute       hipMemPoolGetAttribute
#define cudaMemPoolSetAccess          hipMemPoolSetAccess
#define cudaMemPoolGetAccess          hipMemPoolGetAccess
#define cudaDeviceGraphMemTrim        hipDeviceGraphMemTrim

typedef hipMemPool_t                 cudaMemPool_t;
typedef hipMemPoolProps               cudaMemPoolProps;
typedef hipMemAccessDesc             cudaMemAccessDesc;
typedef hipMemAccessFlags            cudaMemAccessFlags;
typedef hipMemAllocationType         cudaMemAllocationType;
typedef hipMemAllocationHandleType   cudaMemAllocationHandleType;
typedef hipMemLocation               cudaMemLocation;
typedef hipMemLocationType           cudaMemLocationType;
typedef hipMemPoolAttr               cudaMemPoolAttr;

#define cudaMemPoolAttrReleaseThreshold     hipMemPoolAttrReleaseThreshold
#define cudaMemAccessFlagsProtReadWrite     hipMemAccessFlagsProtReadWrite
#define cudaMemLocationTypeDevice           hipMemLocationTypeDevice
#define cudaMemAllocationTypePinned         hipMemAllocationTypePinned
#define cudaMemHandleTypeNone               hipMemHandleTypeNone

#define cudaStreamBeginCapture        hipStreamBeginCapture
#define cudaStreamBeginCaptureToGraph  hipStreamBeginCaptureToGraph
#define cudaStreamEndCapture          hipStreamEndCapture
#define cudaStreamIsCapturing         hipStreamIsCapturing
#define cudaStreamCaptureModeGlobal   hipStreamCaptureModeGlobal

typedef hipStreamCaptureMode cudaStreamCaptureMode;

#define cudaGraphInstantiateWithFlags  hipGraphInstantiateWithFlags
#define cudaGraphLaunch               hipGraphLaunch
#define cudaGraphUpload               hipGraphUpload
#define cudaGraphDestroy              hipGraphDestroy
#define cudaGraphExecDestroy          hipGraphExecDestroy
#define cudaGraphGetNodes             hipGraphGetNodes
#define cudaGraphDebugDotPrint        hipGraphDebugDotPrint
#define cudaGraphAddChildGraphNode    hipGraphAddChildGraphNode
#define cudaGraphChildGraphNodeGetGraph hipGraphChildGraphNodeGetGraph
#define cudaGraphAddMemcpyNode1D      hipGraphAddMemcpyNode1D
#define cudaGraphExecMemcpyNodeSetParams1D hipGraphExecMemcpyNodeSetParams1D
#define cudaGraphAddMemFreeNode       hipGraphAddMemFreeNode
#define cudaGraphRetainUserObject     hipGraphRetainUserObject
#define cudaUserObjectCreate          hipUserObjectCreate

typedef hipGraphNode_t          cudaGraphNode_t;
typedef hipGraph_t              cudaGraph_t;
typedef hipGraphExec_t          cudaGraphExec_t;
typedef hipKernelNodeParams     cudaKernelNodeParams;
typedef hipMemcpy3DParms        cudaMemcpy3DParms;
typedef hipUserObject_t         cudaUserObject_t;

// Conditional graphs are not available in HIP yet
inline hipError_t cudaGraphConditionalHandleCreate(void*, void*, unsigned int, unsigned int)
{
    return hipErrorNotSupported;
}
inline hipError_t cudaGraphSetConditional(void*, unsigned int)
{
    return hipErrorNotSupported;
}

typedef unsigned long long cuuint64_t;

typedef int CUjit_option;
#define CU_JIT_ERROR_LOG_BUFFER            0
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 1
#define CU_JIT_INPUT_PTX                   0

typedef hipFuncAttribute CUfunction_attribute;
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES hipFuncAttributeMaxDynamicSharedMemorySize

typedef hip_Memcpy2D     CUDA_MEMCPY2D;
typedef HIP_MEMCPY3D     CUDA_MEMCPY3D;

typedef hipGraphicsResource_t CUgraphicsResource;

typedef size_t (*CUoccupancyB2DSize)(int);

#define cudaErrorCallRequiresNewerDriver hipErrorNotSupported

// Sentinel so that #if CUDA_VERSION >= XXXX guards evaluate to false
#define CUDA_VERSION 0

#endif // WP_ENABLE_HIP
