#include <string>
#include "ClrUtils.h"


BSTR StdStrToBSTR(std::string what)
{
	int wslen = MultiByteToWideChar(CP_ACP, 0, what.data(), strlen(what.data()), 0, 0);
	BSTR bstr = SysAllocStringLen(0, wslen);
	MultiByteToWideChar(CP_ACP, 0, what.data(), strlen(what.data()), bstr, wslen);
	return bstr;
}


BSTR CudaStatusBSTR(cudaError_t cuda_status, std::string func_name)
{
	std::string strRet = "error in " + func_name + ": " +
		CudaStatusToChars(cuda_status);

	return StdStrToBSTR(strRet);
}


BSTR RuntimeErrBSTR(std::string err_msg, std::string func_name)
{
	std::string strRet = "error in " + func_name + ": " + err_msg;
	return StdStrToBSTR(strRet);
}


const char* CudaStatusToChars(cudaError_t status)
{
	switch (status)
	{
	case cudaSuccess:   return "cudaSuccess";
	case cudaErrorMissingConfiguration:   return "cudaErrorMissingConfiguration";
	case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
	case cudaErrorInitializationError: return "cudaErrorInitializationError";
	case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
	case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
	case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
	case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
	case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
	case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
	case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
	case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
	case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
	case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
	case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
	case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
	case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
	case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
	case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";
	case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";
	case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";
	case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
	case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";
	case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";
	case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";
	case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";
	case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";
	case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";
	case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";
	case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
	case cudaErrorUnknown: return "cudaErrorUnknown";
	case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";
	case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";
	case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
	case cudaErrorNotReady: return "cudaErrorNotReady";
	case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";
	case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";
	case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";
	case cudaErrorNoDevice: return "cudaErrorNoDevice";
	case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";
	case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";
	case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";
	case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";
	case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";
	case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";
	case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";
	case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";
	case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";
	case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";
	case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";
	case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";
	case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled";
	case cudaErrorDeviceAlreadyInUse: return "cudaErrorDeviceAlreadyInUse";
	case cudaErrorProfilerDisabled: return "cudaErrorProfilerDisabled";
	case cudaErrorProfilerNotInitialized: return "cudaErrorProfilerNotInitialized";
	case cudaErrorProfilerAlreadyStarted: return "cudaErrorProfilerAlreadyStarted";
	case cudaErrorProfilerAlreadyStopped: return "cudaErrorProfilerAlreadyStopped";
	case cudaErrorAssert: return "cudaErrorAssert";
	case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";
	case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered";
	case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered";
	case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem";
	case cudaErrorPeerAccessUnsupported: return "cudaErrorPeerAccessUnsupported";
	case cudaErrorLaunchMaxDepthExceeded: return "cudaErrorLaunchMaxDepthExceeded";
	case cudaErrorLaunchFileScopedTex: return "cudaErrorLaunchFileScopedTex";
	case cudaErrorLaunchFileScopedSurf: return "cudaErrorLaunchFileScopedSurf";
	case cudaErrorSyncDepthExceeded: return "cudaErrorSyncDepthExceeded";
	case cudaErrorLaunchPendingCountExceeded: return "cudaErrorLaunchPendingCountExceeded";
	case cudaErrorNotPermitted: return "cudaErrorNotPermitted";
	case cudaErrorNotSupported: return "cudaErrorNotSupported";
	case cudaErrorHardwareStackError: return "cudaErrorHardwareStackError";
	case cudaErrorIllegalInstruction: return "cudaErrorIllegalInstruction";
	case cudaErrorMisalignedAddress: return "cudaErrorMisalignedAddress";
	case cudaErrorInvalidAddressSpace: return "cudaErrorInvalidAddressSpace";
	case cudaErrorInvalidPc: return "cudaErrorInvalidPc";
	case cudaErrorIllegalAddress: return "cudaErrorIllegalAddress";
	case cudaErrorInvalidPtx: return "cudaErrorInvalidPtx";
	case cudaErrorInvalidGraphicsContext: return "cudaErrorInvalidGraphicsContext";
	case cudaErrorNvlinkUncorrectable: return "cudaErrorNvlinkUncorrectable";
	case cudaErrorJitCompilerNotFound: return "cudaErrorJitCompilerNotFound";
	case cudaErrorCooperativeLaunchTooLarge: return "cudaErrorCooperativeLaunchTooLarge";
	case cudaErrorStartupFailure: return "cudaErrorStartupFailure";
	case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";
	default:      return "[Unknown cudaError]";
	}
}


BSTR CurandStatusBSTR(curandStatus_t cuda_status, std::string func_name)
{
	std::string strRet = "error in " + func_name + ": " +
		CurandStatusToChars(cuda_status);

	return StdStrToBSTR(strRet);
}

const char* CurandStatusToChars(curandStatus_t status)
{
	switch (status)
	{
	case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS"; ///< No errors
	case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH"; ///< Header file and linked library version do not match
	case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED"; ///< Generator not initialized
	case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED"; ///< Memory allocation failed
	case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR"; ///< Generator is wrong type
	case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE"; ///< Argument out of range
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE"; ///< Length requested is not a multple of dimension
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"; ///< GPU does not have double precision required by MRG32k3a
	case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE"; ///< Kernel launch failure
	case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE"; ///< Preexisting failure on library entry
	case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED"; ///< Initialization of CUDA failed
	case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH"; ///< Architecture mismatch, GPU does not support requested feature
	case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";///< Internal library error
	default:      return "[Unknown curandStatus]";
	}
}
