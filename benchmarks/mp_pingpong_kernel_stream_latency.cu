/****
 * Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ****/

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <mpi.h>
#include <gdsync.h>
#include <mp.h>
#include "mp/device.cuh"
#include <vector>

#include "prof.h"

#define CUDA_CHECK(stmt)                                \
do {                                                    \
    cudaError_t result = (stmt);                        \
    if (cudaSuccess != result) {                        \
        fprintf(stderr, "[%s:%d] cuda failed with %s \n",   \
         __FILE__, __LINE__,cudaGetErrorString(result));\
        exit(-1);                                       \
    }                                                   \
    assert(cudaSuccess == result);                      \
} while (0)

#define CU_CHECK(stmt)                                 \
do {                                                    \
    CUresult result = (stmt);                           \
    if (CUDA_SUCCESS != result) {                        \
        fprintf(stderr, "[%s:%d] cuda failed with %d \n",   \
         __FILE__, __LINE__, result);\
        exit(-1);                                       \
    }                                                   \
    assert(CUDA_SUCCESS == result);                     \
} while (0)

#define MP_CHECK(stmt)                                  \
do {                                                    \
    int result = (stmt);                                \
    if (0 != result) {                                  \
        fprintf(stderr, "[%s:%d] mp call failed \n",    \
         __FILE__, __LINE__);                           \
        exit(-1);                                       \
    }                                                   \
    assert(0 == result);                                \
} while (0)

int enable_debug_prints = 0;
#define mp_dbg_msg(FMT, ARGS...)  do                                    \
{                                                                       \
    if (enable_debug_prints)  {                                              \
        fprintf(stderr, "[%d] [%d] MP DBG  %s() " FMT, getpid(),  my_rank, __FUNCTION__ , ## ARGS); \
        fflush(stderr);                                                 \
    }                                                                   \
} while(0)

#define MAX_SIZE 1*1024*1024 
#define ITER_COUNT_SMALL 1000
#define ITER_COUNT_LARGE 1000

struct prof prof_normal;
struct prof prof_async;
int prof_start = 0;
int prof_idx = 0;

static const int over_sub_factor = 2;
int gpu_num_sm;
int enable_ud = 0;
int gpu_id = -1;

int comm_size, my_rank, peer;
int steps_per_batch = 20, batches_inflight = 4;
int enable_async = 1;
int calc_size = 128*1024;
int use_calc_size = 1;
volatile uint32_t tracking_event = 0;
float *in = NULL;
float *out = NULL;
int num_streams = 1;

unsigned int *sindex_d = NULL;
unsigned int *windex_d = NULL;
__device__ unsigned int rreq_count_d;
__device__ unsigned int sreq_count_d;

__device__ int counter;
__device__ int clockrate;

__global__ void calc_kernel(int n, float c, float *in, float *out)
{
        const uint tid = threadIdx.x;
        const uint bid = blockIdx.x;
        const uint block_size = blockDim.x;
        const uint grid_size = gridDim.x;
        const uint gid = tid + bid*block_size;
        const uint n_threads = block_size*grid_size;
        for (int i=gid; i<n; i += n_threads)
                out[i] = in[i] * c;
}

int gpu_launch_calc_kernel(size_t size, cudaStream_t stream)
{
        const int nblocks = over_sub_factor * gpu_num_sm;
        const int nthreads = 32*2;
        int n = size / sizeof(float);
        static float *in = NULL;
        static float *out = NULL;
        if (!in) {
                CUDA_CHECK(cudaMalloc((void **)&in, size));
                CUDA_CHECK(cudaMalloc((void **)&out, size));

                CUDA_CHECK(cudaMemset((void *)in, 1, size));
                CUDA_CHECK(cudaMemset((void *)out, 1, size));
        }
        calc_kernel<<<nblocks, nthreads, 0, stream>>>(n, 1.0f, in, out);
        CUDA_CHECK(cudaGetLastError());
        return 0;
}

__global__ void dummy_kernel(double time)
{
    long long int start, stop;
    double usec;

    start = clock64();
    do {
        stop = clock64();
	usec = ((double)(stop-start)*1000)/((double)clockrate); 
	counter = usec;
    } while(usec < time);
}

/*application and pack buffers*/
void *buf = NULL, *sbuf_d = NULL, *rbuf_d = NULL;
cudaStream_t *streams;
cudaStream_t main_stream;
cudaEvent_t start_event, stop_event;
size_t buf_size; 

/*mp specific objects*/
mp_request_t *sreq = NULL;
mp_request_t *rreq = NULL;
mp::mlx5::send_desc_t *sdesc = NULL;
mp::mlx5::send_desc_t *sdesc_d = NULL;
mp::mlx5::wait_desc_t *wdesc = NULL;
mp::mlx5::wait_desc_t *wdesc_d = NULL;
cudaGraphExec_t graphExec;
cudaGraph_t graph, subGraph;
cudaGraphNode_t emptyNode; 

mp_reg_t sreg, rreg; 
double time_start, time_stop;

int batch_to_rreq_idx (int batch_idx) { 
     return (batch_idx % (batches_inflight + 1))*steps_per_batch*num_streams;
}

int batch_to_sreq_idx (int batch_idx) { 
     return (batch_idx % batches_inflight)*steps_per_batch*num_streams;
}

void post_recv (int size, int batch_index)
{
    int j;
    int req_idx = batch_to_rreq_idx (batch_index);
 
    for (j=0; j<steps_per_batch*num_streams; j++) {
        MP_CHECK(mp_irecv ((void *)((uintptr_t)rbuf_d), size, peer, &rreg, &rreq[req_idx + j]));
    }
}

void wait_send (int batch_index) 
{
    int j;
    int req_idx = batch_to_sreq_idx (batch_index); 

    for (j=0; j<steps_per_batch*num_streams; j++) {
        MP_CHECK(mp_wait(&sreq[req_idx + j]));
    }
}

void wait_recv (int batch_index) 
{
    int j;
    int req_idx = batch_to_rreq_idx (batch_index);
 
    for (j=0; j<steps_per_batch*num_streams; j++) {
        MP_CHECK(mp_wait(&rreq[req_idx + j]));
    }
}

__global__ void send_op_kernel (mp::mlx5::send_desc_t desc)
{
    mp::device::mlx5::send(desc);
}

__global__ void wait_op_kernel (mp::mlx5::wait_desc_t desc)
{
    mp::device::mlx5::wait(desc);
    mp::device::mlx5::signal(desc);
}

__global__ void send_op_kernel_graph (mp::mlx5::send_desc_t *desc, unsigned int *index)
{
    unsigned int idx = atomicInc(index, sreq_count_d);
    mp::device::mlx5::send(desc[idx]);
}

__global__ void wait_op_kernel_graph (mp::mlx5::wait_desc_t *desc, unsigned int *index)
{
    unsigned int idx = atomicInc(index, rreq_count_d);
    mp::device::mlx5::wait(desc[idx]);
    mp::device::mlx5::signal(desc[idx]);
}

void create_work_async_graph (size_t size, double kernel_size) 
{
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaKernelNodeParams waitParams, sendParams, calcKernelParams, dummyKernelParams;
    cudaGraphNode_t sendNode, waitNode, kernelNode, emptyNode, subGraphNode[2];

    CUDA_CHECK(cudaMalloc((void **)&in, size));
    CUDA_CHECK(cudaMalloc((void **)&out, size));
    CUDA_CHECK(cudaMalloc((void **)&sindex_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&windex_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset((void *)in, 1, size));
    CUDA_CHECK(cudaMemset((void *)out, 1, size));
    CUDA_CHECK(cudaMemset((void *)sindex_d, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset((void *)windex_d, 0, sizeof(unsigned int)));

    int sreq_count = batches_inflight*steps_per_batch*num_streams;
    CUDA_CHECK(cudaMemcpyToSymbol(sreq_count_d, (void *)&sreq_count, sizeof(int), 
			    	0, cudaMemcpyHostToDevice));
    int rreq_count = (batches_inflight + 1)*steps_per_batch*num_streams;
    CUDA_CHECK(cudaMemcpyToSymbol(rreq_count_d, (void *)&rreq_count, sizeof(int), 
			    	0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaGraphCreate(&subGraph, 0));
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    waitParams.func = (void*)wait_op_kernel_graph;
    waitParams.gridDim = 1;
    waitParams.blockDim = 1;
    waitParams.sharedMemBytes = 0;
    void *waitArgs[2] = {(void*)&wdesc_d, (void *)&windex_d};
    waitParams.kernelParams = waitArgs;
    waitParams.extra = NULL;

    sendParams.func = (void*)send_op_kernel_graph;
    sendParams.gridDim = 1;
    sendParams.blockDim = 1;
    sendParams.sharedMemBytes = 0;
    void *sendArgs[2] = {(void*)&sdesc_d, (void *)&sindex_d};
    sendParams.kernelParams = sendArgs;
    sendParams.extra = NULL;

    const float value = 0.1F; 
    int n = kernel_size / sizeof(float);

    calcKernelParams.func = (void*)calc_kernel;
    calcKernelParams.gridDim = over_sub_factor * gpu_num_sm;
    calcKernelParams.blockDim = 32*2;
    calcKernelParams.sharedMemBytes = 0;
    void *calcKernelArgs[4] = {(void*)&n, (void *)&value, (void *)&in, (void *)&out};
    calcKernelParams.kernelParams = calcKernelArgs;
    calcKernelParams.extra = NULL;

    dummyKernelParams.func = (void*)dummy_kernel;
    dummyKernelParams.gridDim = 1;
    dummyKernelParams.blockDim = 1;
    dummyKernelParams.sharedMemBytes = 0;
    void *dummyKernelArgs[1] = {(void*)&kernel_size};
    dummyKernelParams.kernelParams = dummyKernelArgs;
    dummyKernelParams.extra = NULL;
    
    CUDA_CHECK(cudaGraphAddEmptyNode(&emptyNode, subGraph, NULL, 0));

    if (!my_rank) { 
	for(int k=0; k<num_streams; k++) {
           nodeDependencies.clear();
           nodeDependencies.push_back(emptyNode);
     	   CUDA_CHECK(cudaGraphAddKernelNode(&waitNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &waitParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(waitNode);
           if (kernel_size > 0) {
              if (use_calc_size > 0) {
                  CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &calcKernelParams));
              } else { 
                  CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &dummyKernelParams));
              }
              nodeDependencies.clear();
              nodeDependencies.push_back(kernelNode);
           }

           CUDA_CHECK(cudaGraphAddKernelNode(&sendNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &sendParams));
	}
    } else {
	for(int k=0; k<num_streams; k++) {
           nodeDependencies.clear();
           nodeDependencies.push_back(emptyNode);
           CUDA_CHECK(cudaGraphAddKernelNode(&sendNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &sendParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(sendNode);
           CUDA_CHECK(cudaGraphAddKernelNode(&waitNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &waitParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(waitNode);
	   if (kernel_size > 0) {
              if (use_calc_size > 0) {
                  CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &calcKernelParams));
              } else { 
                  CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, subGraph, nodeDependencies.data(), nodeDependencies.size(), &dummyKernelParams));
              }
           }
        }
    }
  
    CUDA_CHECK(cudaGraphAddChildGraphNode (&subGraphNode[0], graph, NULL, 0, subGraph)); 
    for (int k=1; k<steps_per_batch; k++) {
        nodeDependencies.clear();
        nodeDependencies.push_back(subGraphNode[(k-1)%2]);
	CUDA_CHECK(cudaGraphAddChildGraphNode (&subGraphNode[k%2], graph, nodeDependencies.data(), nodeDependencies.size(), subGraph));
    }

    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
}

void prepare_work_async_graphs (int size, int batch_index)
{
    int sreq_idx = batch_to_sreq_idx (batch_index);
    int rreq_idx = batch_to_rreq_idx (batch_index);

    for (int j=0; j<steps_per_batch; j++) {
	for(int k=0; k<num_streams; k++) { 
 	    int s_idx = sreq_idx + j*num_streams + k;
	    int r_idx = rreq_idx + j*num_streams + k;

	    if (!my_rank) {
                MP_CHECK(mp::mlx5::get_descriptors(&wdesc[r_idx], &rreq[r_idx]));

                MP_CHECK(mp_send_prepare((void *)((uintptr_t)sbuf_d), size, peer, &sreg, &sreq[s_idx]));
                MP_CHECK(mp::mlx5::get_descriptors(&sdesc[sreq_idx], &sreq[s_idx]));
   	    } else {
   	        MP_CHECK(mp_send_prepare((void *)((uintptr_t)sbuf_d), size, peer, &sreg, &sreq[s_idx]));
                MP_CHECK(mp::mlx5::get_descriptors(&sdesc[sreq_idx], &sreq[s_idx]));

	        MP_CHECK(mp::mlx5::get_descriptors(&wdesc[r_idx], &rreq[r_idx]));
 	    }	
	}
    }
}

void test_post_work_async_kernels (int size, int batch_index, double kernel_size) 
{
    int sreq_idx = batch_to_sreq_idx (batch_index);
    int rreq_idx = batch_to_rreq_idx (batch_index);

    for (int j=0; j<steps_per_batch; j++) {
	CUDA_CHECK(cudaEventRecord(start_event, main_stream));	
	for(int k=0; k<num_streams; k++) {
	    int s_idx = sreq_idx + j*num_streams + k;
	    int r_idx = rreq_idx + j*num_streams + k;

   	    CUDA_CHECK(cudaStreamWaitEvent(streams[k], start_event, 0));	

	    if (!my_rank) {
	        wait_op_kernel<<<1,1,0,streams[k]>>>(wdesc[r_idx]);
                CUDA_CHECK(cudaGetLastError());

                if (kernel_size > 0) {
                    if (use_calc_size > 0)
                       gpu_launch_calc_kernel(kernel_size, streams[k]);
                    else
                       dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
                }

	        send_op_kernel<<<1,1,0,streams[k]>>>(sdesc[s_idx]);
                CUDA_CHECK(cudaGetLastError());
 	    } else {
	       send_op_kernel<<<1,1,0,streams[k]>>>(sdesc[s_idx]);
               CUDA_CHECK(cudaGetLastError());

	       wait_op_kernel<<<1,1,0,streams[k]>>>(wdesc[r_idx]);
               CUDA_CHECK(cudaGetLastError());

               if (kernel_size > 0) {
                   if (use_calc_size > 0)
                      gpu_launch_calc_kernel(kernel_size, streams[k]);
                   else
                      dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
               }
 	    }

	    CUDA_CHECK(cudaEventRecord(stop_event, streams[k]));
	    CUDA_CHECK(cudaStreamWaitEvent(main_stream, stop_event, 0));
	}
    }
}

void post_work_async_graphs (int size, int batch_index, double kernel_size)
{
    fprintf(stderr, "preparing graph batch %d \n", batch_index);
    prepare_work_async_graphs (size, batch_index);
    //test_post_work_async_kernels (size, batch_index, kernel_size);
    CUDA_CHECK(cudaGraphLaunch(graphExec, main_stream));
}

void post_work_async_kernels (int size, int batch_index, double kernel_size) 
{
    int sreq_idx = batch_to_sreq_idx (batch_index);
    int rreq_idx = batch_to_rreq_idx (batch_index);

    for (int j=0; j<steps_per_batch; j++) {
	CUDA_CHECK(cudaEventRecord(start_event, main_stream));	
	for(int k=0; k<num_streams; k++) {
	    int s_idx = sreq_idx + j*num_streams + k;
	    int r_idx = rreq_idx + j*num_streams + k;

   	    CUDA_CHECK(cudaStreamWaitEvent(streams[k], start_event, 0));	

	    if (!my_rank) {
                MP_CHECK(mp::mlx5::get_descriptors(&wdesc[r_idx], &rreq[r_idx]));
	        wait_op_kernel<<<1,1,0,streams[k]>>>(wdesc[r_idx]);
                CUDA_CHECK(cudaGetLastError());

                if (kernel_size > 0) {
                    if (use_calc_size > 0)
                       gpu_launch_calc_kernel(kernel_size, streams[k]);
                    else
                       dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
                }

                MP_CHECK(mp_send_prepare((void *)((uintptr_t)sbuf_d), size, peer, &sreg, &sreq[s_idx]));
                MP_CHECK(mp::mlx5::get_descriptors(&sdesc[s_idx], &sreq[s_idx]));
	        send_op_kernel<<<1,1,0,streams[k]>>>(sdesc[s_idx]);
                CUDA_CHECK(cudaGetLastError());
 	    } else {
               MP_CHECK(mp_send_prepare((void *)((uintptr_t)sbuf_d), size, peer, &sreg, &sreq[s_idx]));
               MP_CHECK(mp::mlx5::get_descriptors(&sdesc[s_idx], &sreq[s_idx]));
	       send_op_kernel<<<1,1,0,streams[k]>>>(sdesc[s_idx]);
               CUDA_CHECK(cudaGetLastError());

	       MP_CHECK(mp::mlx5::get_descriptors(&wdesc[r_idx], &rreq[r_idx]));
	       wait_op_kernel<<<1,1,0,streams[k]>>>(wdesc[r_idx]);
               CUDA_CHECK(cudaGetLastError());

               if (kernel_size > 0) {
                   if (use_calc_size > 0)
                      gpu_launch_calc_kernel(kernel_size, streams[k]);
                   else
                      dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
               }
 	    }

	    CUDA_CHECK(cudaEventRecord(stop_event, streams[k]));
	    CUDA_CHECK(cudaStreamWaitEvent(main_stream, stop_event, 0));
	}
    }
}

void post_work_async (int size, int batch_index, double kernel_size) 
{
    int sreq_idx = batch_to_sreq_idx (batch_index);
    int rreq_idx = batch_to_rreq_idx (batch_index);
   
    for (int j=0; j<steps_per_batch; j++) {
	CUDA_CHECK(cudaEventRecord(start_event, main_stream));	
	for(int k=0; k<num_streams; k++) {
	    int s_idx = sreq_idx + j*num_streams + k;
	    int r_idx = rreq_idx + j*num_streams + k;

  	    CUDA_CHECK(cudaStreamWaitEvent(streams[k], start_event, 0));	

	    if (!my_rank) { 
                MP_CHECK(mp_wait_on_stream(&rreq[r_idx], streams[k]));

                if (kernel_size > 0) {
                    if (use_calc_size > 0)
                       gpu_launch_calc_kernel(kernel_size, streams[k]);
                    else
                       dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
                }

                MP_CHECK(mp_isend_on_stream ((void *)((uintptr_t)sbuf_d), size, peer, &sreg, 
					&sreq[s_idx], streams[k]));
	    } else {
                MP_CHECK(mp_isend_on_stream ((void *)((uintptr_t)sbuf_d), size, peer, &sreg, 
					&sreq[s_idx], streams[k]));

                MP_CHECK(mp_wait_on_stream(&rreq[r_idx], streams[k]));

                if (kernel_size > 0) {
                    if (use_calc_size > 0)
                       gpu_launch_calc_kernel(kernel_size, streams[k]);
                    else
                       dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
                }
	    }

	    CUDA_CHECK(cudaEventRecord(stop_event, streams[k]));
	    CUDA_CHECK(cudaStreamWaitEvent(main_stream, stop_event, 0));
	}
    }
}

void post_work_sync (int size, int batch_index, double kernel_size) 
{
    int rreq_idx = batch_to_rreq_idx (batch_index);
    int sreq_idx = batch_to_sreq_idx (batch_index);

    for (int j=0; j<steps_per_batch; j++) {
	if (!my_rank) {
	    for(int k=0; k<num_streams; k++) {
		MP_CHECK(mp_wait(&rreq[rreq_idx + j*num_streams + k]));

                if (kernel_size > 0) {
                    if (use_calc_size > 0)
                       gpu_launch_calc_kernel(kernel_size, streams[k]);
                    else
                       dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
                }
	    }

	    for(int k=0; k<num_streams; k++) { 
                CUDA_CHECK(cudaStreamSynchronize(streams[k]));

                MP_CHECK(mp_isend ((void *)((uintptr_t)sbuf_d), size, peer, &sreg, &sreq[sreq_idx + j*num_streams + k]));
	    }
       } else {
	    for(int k=0; k<num_streams; k++) { 
                MP_CHECK(mp_isend ((void *)((uintptr_t)sbuf_d), size, peer, &sreg, &sreq[sreq_idx + j*num_streams + k]));
	    }

	    for(int k=0; k<num_streams; k++) { 
                MP_CHECK(mp_wait(&rreq[rreq_idx + j*num_streams + k]));

                if (kernel_size > 0) {
                    if (use_calc_size > 0)
                       gpu_launch_calc_kernel(kernel_size, streams[k]);
                    else
                       dummy_kernel <<<1, 1, 0, streams[k]>>> (kernel_size);
                }
	    }

	    for(int k=0; k<num_streams; k++) { 
                CUDA_CHECK(cudaStreamSynchronize(streams[k]));
	    }
        }
    }
}

double prepost_latency;

double sr_exchange (MPI_Comm comm, int size, int iter_count, double kernel_size, int use_async, int use_kernel_ops = 0, int use_graphs = 0)
{
    int j;
    double latency;
    double time_start, time_stop;
    int batch_count, wait_send_batch = 0, wait_recv_batch = 0;
    struct prof *prof = NULL;
    static int graph_created = 0;

    prof = (use_async) ? &prof_async : &prof_normal;
 
    if (iter_count%steps_per_batch != 0) { 
	fprintf(stderr, "iter_count must be a multiple of steps_per_batch: %d \n", steps_per_batch);
	exit(-1);
    }
    batch_count = iter_count/steps_per_batch;
    tracking_event = 0;

    if (use_graphs && !graph_created) {
        if (!use_kernel_ops) { 
	    fprintf(stderr, "CUDA graphs cannot be used without kernel ops \n"); 
	    return 0; 
	}
        create_work_async_graph (size, kernel_size);
	graph_created = 1;
    }

    post_recv (size, 0);

    MPI_Barrier(MPI_COMM_WORLD);

    time_start = MPI_Wtime();

    for (j=0; (j<batches_inflight) && (j<batch_count); j++) { 
        if (j<(batch_count-1)) {
	    post_recv (size, j+1);
	}

        if (use_async) { 
            if (use_kernel_ops) {
	        if (use_graphs) { 
		    post_work_async_graphs (size, j, kernel_size);
		} else { 
		    post_work_async_kernels (size, j, kernel_size);
		}
            } else { 
                post_work_async (size, j, kernel_size);
	    }
        } else { 
            post_work_sync (size, j, kernel_size);
	}
    }

    time_stop = MPI_Wtime();

    prepost_latency = ((time_stop - time_start)*1e6);
    
    time_start = MPI_Wtime();

    wait_send_batch = wait_recv_batch = 0;
    prof_idx = 0;
    while (wait_send_batch < batch_count) { 
        if (!my_rank && prof_start) PROF(prof, prof_idx++);

	if (use_async) {
	    wait_recv (wait_recv_batch);
            wait_recv_batch++;
	}

        if (!my_rank && prof_start) PROF(prof, prof_idx++); 

        wait_send (wait_send_batch);
	wait_send_batch++;

        if (!my_rank && prof_start) PROF(prof, prof_idx++);

	if (j < (batch_count-1)) {
	    post_recv (size, j+1);
	}

        if (!my_rank && prof_start) PROF(prof, prof_idx++);

	if (j < batch_count) { 
           if (use_async) { 
               if (use_kernel_ops) {
	           if (use_graphs) { 
	   	    post_work_async_graphs (size, j, kernel_size);
	   	} else { 
	   	    post_work_async_kernels (size, j, kernel_size);
	   	}
               } else { 
                   post_work_async (size, j, kernel_size);
	       }
           } else { 
               post_work_sync (size, j, kernel_size);
	   }
	}

        if (!my_rank && prof_start)  {
            PROF(prof, prof_idx++);
            prof_update(prof);
            prof_idx = 0;
        }

	j++;
    }

    MPI_Barrier(comm);

    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    time_stop = MPI_Wtime();
    latency = (((time_stop - time_start)*1e6 + prepost_latency)/(iter_count*2));

    CUDA_CHECK(cudaDeviceSynchronize());

    return latency;
}

int main (int argc, char *argv[])
{
    int iter_count, max_size, size, dev_count, local_rank, dev_id = 0;
    int kernel_size = 20;
    int comm_comp_ratio = 1;
    int validate = 0, user_iter_count = 0;

    size = 1;
    max_size = MAX_SIZE;

    char *value = getenv("ENABLE_VALIDATION");
    if (value != NULL) {
	validate = atoi(value);
    }
 
    value = getenv("ITER_COUNT");
    if (value != NULL) {
	user_iter_count = atoi(value);
    }

    value = getenv("ENABLE_DEBUG_MSG");
    if (value != NULL) {
	enable_debug_prints = atoi(value);
    }

    value = getenv("KERNEL_TIME");
    if (value != NULL) {
	kernel_size = atoi(value);
    }

    value = getenv("COMM_COMP_RATIO");
    if (value != NULL) {
        comm_comp_ratio = atoi(value);
    }

    value = getenv("CALC_SIZE");
    if (value != NULL) {
        calc_size = atoi(value);
    }

    use_calc_size = 0;
    value = getenv("USE_CALC_SIZE");
    if (value != NULL) {
        use_calc_size = atoi(value);
    }

    value = getenv("STEPS_PER_BATCH");
    if (value != NULL) {
        steps_per_batch = atoi(value);
    }

    value = getenv("BATCHES_INFLIGHT");
    if (value != NULL) {
        batches_inflight = atoi(value);
    }

    value = getenv("SIZE");
    if (value != NULL && atoi(value)) {
        size = atoi(value);
    }

    value = getenv("NUM_STREAMS");
    if (value != NULL && atoi(value)) {
        num_streams = atoi(value);
    }

    value = getenv("MP_ENABLE_UD");
    if (value != NULL) {
        enable_ud = atoi(value);
    }

    if (enable_ud) {
	if (max_size > 4096) { 
	    max_size = 4096;
        }
    }

    while(1) {
        int c;
        c = getopt(argc, argv, "d:h");
        if (c == -1)
            break;

        switch(c) {
        case 'd':
            gpu_id = strtol(optarg, NULL, 0);
            break;
	case 'h':
            printf("syntax: %s [-d <gpu_id]\n", argv[0]);
	    break;
        default:
            printf("ERROR: invalid option\n");
            exit(EXIT_FAILURE);
        }
    }

    char tags[] = "wait_recv|wait_send|post_recv|post_work";

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (comm_size != 2) { 
	fprintf(stderr, "this test requires exactly two processes \n");
        exit(-1);
    }

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        fprintf(stderr, "no CUDA devices found \n");
        exit(-1);
    }

    if (getenv("MV2_COMM_WORLD_LOCAL_RANK") != NULL) {
        local_rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
    } else if (getenv("OMPI_COMM_WORLD_LOCAL_RANK") != NULL) {
        local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
    } else {
        local_rank = 0;
    }

    if (gpu_id >= 0) {
        dev_id = gpu_id;
    } else if (getenv("USE_GPU")) {
        dev_id = atoi(getenv("USE_GPU"));
    } else {
        dev_id = local_rank%dev_count;
    }
    if (dev_id >= dev_count) {
        fprintf(stderr, "invalid dev_id=%d\n", dev_id);
        exit(-1);
    }

    fprintf(stdout, "[%d] local_rank: %d dev_count: %d using GPU device: %d\n", my_rank, local_rank, dev_count, dev_id);

    CUDA_CHECK(cudaSetDevice(dev_id));
    CUDA_CHECK(cudaFree(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0, cudaMemcpyHostToDevice));
    gpu_num_sm = prop.multiProcessorCount;

    fprintf(stdout, "[%d] GPU %d: %s PCIe %d:%d:%d\n", my_rank, dev_id, prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    peer = !my_rank;
    MP_CHECK(mp_init (MPI_COMM_WORLD, &peer, 1, MP_INIT_DEFAULT, dev_id));

    iter_count = user_iter_count ? user_iter_count : ITER_COUNT_SMALL;
    if (!my_rank) { 
        fprintf(stdout, "steps_per_batch: %d num_streams: %d batches_inflight: %d \n", 
   		steps_per_batch, num_streams, batches_inflight);
        fprintf(stdout, "NOTE: printing half round-trip latency!!!\n");
    }

    /*allocating requests*/
    sreq = (mp_request_t *) malloc(steps_per_batch*batches_inflight*num_streams*sizeof(mp_request_t));
    rreq = (mp_request_t *) malloc(steps_per_batch*(batches_inflight + 1)*num_streams*sizeof(mp_request_t));
    CUDA_CHECK(cudaHostAlloc((void **)&sdesc, sizeof(mp::mlx5::send_desc_t)*steps_per_batch*batches_inflight*num_streams, 0));
    CUDA_CHECK(cudaHostAlloc((void **)&wdesc, sizeof(mp::mlx5::wait_desc_t)*steps_per_batch*(batches_inflight + 1)*num_streams, 0));
    memset((void *)sdesc, 0, sizeof(mp::mlx5::send_desc_t)*steps_per_batch*batches_inflight*num_streams);
    memset((void *)wdesc, 0, sizeof(mp::mlx5::wait_desc_t)*steps_per_batch*(batches_inflight + 1)*num_streams);
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&sdesc_d, sdesc, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&wdesc_d, wdesc, 0));

    streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*num_streams);
    for (int i=0; i<num_streams; i++) { 
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], 0));
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&main_stream, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&start_event, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&stop_event, 0));

    if (!my_rank) {
	if (use_calc_size) { 
		fprintf(stdout, "%10s \t %10s \t %10s \t %10s \t  %10s \t %10s \t %10s \t %10s \n", "Size", "CalcSize", "No-async", "No-async+Kernel", "Async", "Async+Kernel", "Async-SM-Ops", "Async-SM-Ops+Kernel");
	} else {
		fprintf(stdout, "%10s \t %10s \t  %10s \t %10s \t %10s \t  %10s \t %10s \t  %10s \n", "Size", "KernelTime", "No-async", "No-async+Kernel", "Async", "Async+Kernel", "Async-SM-Ops", "Async-SM-Ops+Kernel");
	}
    }

    if (size != 1) size = max_size = size;
    for (; size<=max_size; size*=2) 
    {
	double latency;

        if (size > 1024) {
            iter_count = user_iter_count ? user_iter_count : ITER_COUNT_LARGE;
        }

	buf_size = size;

        buf = malloc (buf_size);
        memset(buf, 0, buf_size); 

        CUDA_CHECK(cudaMalloc((void **)&sbuf_d, buf_size));
        CUDA_CHECK(cudaMemset(sbuf_d, 0, buf_size)); 

        CUDA_CHECK(cudaMalloc((void **)&rbuf_d, buf_size));
        CUDA_CHECK(cudaMemset(rbuf_d, 0, buf_size)); 
 
        MP_CHECK(mp_register(sbuf_d, buf_size, &sreg));
        MP_CHECK(mp_register(rbuf_d, buf_size, &rreg));

        if (!my_rank) fprintf(stdout, "%10d", size);

        /*warmup*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

	sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 1/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/);

        MPI_Barrier(MPI_COMM_WORLD);

	sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/, 1/*use_graphs*/);

        MPI_Barrier(MPI_COMM_WORLD);

	/*calculate kenrel time based on latency*/
	latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, 0, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

	if (use_calc_size) 
	    kernel_size = calc_size; 
        else  
  	    kernel_size = (comm_comp_ratio > 0) ? comm_comp_ratio*(latency) : kernel_size;

        if (!my_rank) fprintf(stdout, "\t   %10d", kernel_size);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf)", latency, prepost_latency);

        cudaProfilerStart();
	if (!my_rank) { 
	    if (prof_init(&prof_normal, 10000, 10000, "10us", 100, 1, tags)) {
                fprintf(stderr, "error in prof_init init.\n");
                exit(-1);
            }
            prof_start = 1;
	}

	/*Normal*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, 0 /*no kernel*/, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        if (!my_rank) fprintf(stdout, "\t   %8.2lf ", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf)", latency, prepost_latency);

	/*Normal + Kernel*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        if (!my_rank) fprintf(stdout, "\t   %8.2lf ", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf)", latency, prepost_latency);

	/*Async*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, 0/*kernel_size*/, 1/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        if (!my_rank) fprintf(stdout, "\t   %8.2lf ", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf)", latency, prepost_latency);

	/*Async + kernel*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 1/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        if (!my_rank) fprintf(stdout, "\t   %8.2lf ", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf)", latency, prepost_latency);

	/*Async + Kernel Ops*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, 0 /*kernel_size*/, 1/*use_async*/, 1/*use_kernel_ops*/);

        MPI_Barrier(MPI_COMM_WORLD);

	if (!my_rank) fprintf(stdout, "\t   %8.2lf  ", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf) \n", latency, prepost_latency);

	/*Async + Kernel + Kernel Ops*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/);

        MPI_Barrier(MPI_COMM_WORLD);

	if (!my_rank) fprintf(stdout, "\t   %8.2lf  ", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf) \n", latency, prepost_latency);

#if 0
	/*Async + Kernel Ops + Graphs*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, 0 /*kernel_size*/, 1/*use_async*/, 1/*use_kernel_ops*/, 1/*use_graphs*/);

        MPI_Barrier(MPI_COMM_WORLD);

	if (!my_rank) fprintf(stdout, "\t   %8.2lf  \n", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf) \n", latency, prepost_latency);

	/*Async + Kernel + Kernel Ops + Graphs*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/, 1/*use_graphs*/);

        MPI_Barrier(MPI_COMM_WORLD);

	if (!my_rank) fprintf(stdout, "\t   %8.2lf  ", latency);
        //if (!my_rank) fprintf(stdout, "\t   %8.2lf (%8.2lf) \n", latency, prepost_latency);

#endif

	if (!my_rank) fprintf(stdout, " \n");

	prof_start = 0;
        cudaProfilerStop();

        if (!my_rank && validate) fprintf(stdout, "SendRecv test passed validation with message size: %d \n", size);

        if (!my_rank) {
	    //prof_dump(&prof_normal);
	    prof_dump(&prof_async);
	}

        mp_deregister(&sreg);
        mp_deregister(&rreg);

        CUDA_CHECK(cudaFree(sbuf_d));
        CUDA_CHECK(cudaFree(rbuf_d));
        free(buf);
    }

    for (int i=0; i<num_streams; i++) { 
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(main_stream));
    free(streams);
    free(sreq);
    free(rreq);

    mp_finalize ();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
