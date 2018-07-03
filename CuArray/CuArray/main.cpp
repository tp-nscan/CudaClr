#include "stdio.h"
#include "..\..\Common\ClrUtils.h"
#include <string>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv)
//{
//	std::string ws;
//	std::string ws2;
//	std::string ws3;
//	std::string wsR;
//	ws = "yow";
//	ws2 = "\n";
//	ws3 = ws + ws2;
//	cudaError_t et;
//	et = cudaErrorMapBufferObjectFailed;
//	const char *rv = CudaStatusToString(et);
//	wsR = rv + ws3;
//	printf(wsR.data());
//
//
//	printf(CudaStatusToString(et));
//	printf(ws3.data());
//	BSTR bs = SysAllocStringByteLen(ws.data(), ws.size());
//	std::wstring cc(bs, SysStringLen(bs));
//	char mbstr[11];
//	std::wcstombs(mbstr, cc.c_str(), 11);
//	printf(mbstr);
//}

//
//void SysFreeString(
//	BSTR bstrString




//int main(int argc, char **argv)
//{
//	std::string ws;
//	std::string ws2;
//	std::string ws3;
//	std::string wsR;
//	ws = "yow";
//
//	BSTR res = CudaStatusBSTR(ws, ws);
//
//	ws2 = "\n";
//	ws3 = ws + ws2;
//	cudaError_t et;
//	et = cudaErrorMapBufferObjectFailed;
//	const char *rv = CudaStatusToString(et);
//
//
//	int wslen = MultiByteToWideChar(CP_ACP, 0, rv, strlen(rv), 0, 0);
//	BSTR bstr = SysAllocStringLen(0, wslen);
//	MultiByteToWideChar(CP_ACP, 0, rv, strlen(rv), bstr, wslen);
//
//	wsR = rv; // +ws3;
//	printf(wsR.data());
//
//
//	printf(CudaStatusToString(et));
//	printf(ws3.data());
//	BSTR bs = SysAllocStringByteLen(ws.data() , ws.size());
//
//
//
//
//	//std::wstring cc(bs, SysStringLen(bs));
//	//char mbstr[11];
//	//std::wcstombs(mbstr, cc.c_str(), 11);
//	//printf(mbstr);
//}