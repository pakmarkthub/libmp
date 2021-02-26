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

#define NULL_CHECK(ptr)                                 \
do {                                                    \
    if (ptr == NULL) {                                  \
        fprintf(stderr, "[%s:%d] memory allocation failed \n", \
         __FILE__, __LINE__);				\
        exit(-1);                                       \
    }                                                   \
} while (0)

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

#define MAX_SIZE 256*1024
#define ITER_COUNT_SMALL 1024
#define ITER_COUNT_LARGE 1024

int comm_size, my_rank, peer;
int steps_per_batch = 16, batches_inflight = 4;
int enable_async = 1;

int num_streams = 1;
volatile uint32_t tracking_event = 0;

//calc kernel params
int gpu_num_sm;
static const int over_sub_factor = 2;
int calc_size = 128*1024;
int use_calc_kernel = 0;
uint32_t windex_max;
uint32_t sindex_max;

__device__ int my_rank_d;
__device__ uint32_t windex_max_d;
__device__ uint32_t sindex_max_d;

FILE *cputimes,*streamtimes;
//per-stream state
typedef struct {
    float *in = NULL;
    float *out = NULL;
    void *buf_d;
    mp_request_t *sreq;
    mp_request_t *rreq;
    mp::mlx5::send_desc_t *sdesc;
    mp::mlx5::send_desc_t *sdesc_d;
    mp::mlx5::wait_desc_t *wdesc;
    mp::mlx5::wait_desc_t *wdesc_d;
    mp_reg_t reg; 
    uint32_t sindex;
    uint32_t windex;
    uint32_t *sindex_d;
    uint32_t *windex_d;
    cudaGraph_t subgraph;
    cudaGraph_t subgraph_comms;
    cudaStream_t stream;
} stream_state_t; 

typedef struct {
    int size;
    int batch_index;
} graph_preparation_arg_t;

//global state
stream_state_t *stream_state;
cudaStream_t main_stream;
size_t buf_size; 
int capture_graph = 0;
cudaGraphNode_t emptyNode; 
cudaGraph_t graph, graph_comms;
cudaGraph_t subgraph, subgraph_comms;
cudaGraphExec_t graphexec, graphexec_comms;

graph_preparation_arg_t *graph_pre_arg;

//timing 
cudaEvent_t iter_start_event, iter_stop_event;
cudaEvent_t timer_start_event, timer_stop_event;
volatile int *delay_flag;
volatile int *delay_flag_dptr;
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

__global__ void poll_kernel(long long int time)
{
    long long int start, stop;
    long long int usec;

    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64  %0, %globaltimer;" : "=l"(stop));
        assert(stop >= start);
        usec = ((stop-start)/1000);
        counter = usec;
    } while(usec < time);
}

int batch_to_rreq_idx (int batch_idx) { 
     return (batch_idx % (batches_inflight + 1))*steps_per_batch;
}

int batch_to_sreq_idx (int batch_idx) { 
     return (batch_idx % batches_inflight)*steps_per_batch;
}

void post_recv (int size, int batch_index)
{
    int req_idx = batch_to_rreq_idx (batch_index);
 
    for (int j=0; j<steps_per_batch; j++) {
	for (int k=0; k<num_streams; k++) {
            MP_CHECK(mp_irecv ((void *)stream_state[k].buf_d, 
				size, peer*num_streams + k, &stream_state[k].reg, 
				&stream_state[k].rreq[req_idx + j]));
	}
    }
}

void wait_send (int batch_index) 
{
    int req_idx = batch_to_sreq_idx (batch_index); 

    for (int j=0; j<steps_per_batch; j++) {
        for (int k=0; k<num_streams; k++) {
            MP_CHECK(mp_wait(&stream_state[k].sreq[req_idx + j]));
        }
    }
}

void wait_recv (int batch_index) 
{
    int req_idx = batch_to_rreq_idx (batch_index);

    for (int j=0; j<steps_per_batch; j++) {
        for (int k=0; k<num_streams; k++) {
            MP_CHECK(mp_wait(&stream_state[k].rreq[req_idx + j]));
        }
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
    unsigned int idx = *index;
    *index = (*index+1)%sindex_max_d;
    mp::device::mlx5::send(desc[idx]);
}

__global__ void wait_op_kernel_graph (mp::mlx5::wait_desc_t *desc, unsigned int *index)
{
    unsigned int idx = *index;
    *index = (*index+1)%windex_max_d;
    mp::device::mlx5::wait(desc[idx]);
    mp::device::mlx5::signal(desc[idx]);
}

void prepare_work_async_graphs (int size, int batch_index)
{
    int sreq_idx = batch_to_sreq_idx (batch_index);
    int rreq_idx = batch_to_rreq_idx (batch_index);

    for (int j=0; j<steps_per_batch; j++) {
        for(int k=0; k<num_streams; k++) {
            stream_state_t *curr_stream = (stream_state + k);   	
            int s_idx = sreq_idx + j;
            int r_idx = rreq_idx + j;

            if (!my_rank) {
                MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->wdesc[r_idx], &curr_stream->rreq[r_idx]));

                MP_CHECK(mp_send_prepare((void *)curr_stream->buf_d, size, peer*num_streams + k, 
                            &curr_stream->reg, &curr_stream->sreq[s_idx]));
                MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->sdesc[s_idx], 
                            &curr_stream->sreq[s_idx]));
            } else {
                MP_CHECK(mp_send_prepare((void *)curr_stream->buf_d, size, peer*num_streams + k, 
                            &curr_stream->reg, &curr_stream->sreq[s_idx]));
                MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->sdesc[s_idx], &curr_stream->sreq[s_idx]));

                MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->wdesc[r_idx], &curr_stream->rreq[r_idx]));
            }	
        }
    }
}


void graph_prepare_work (void *data)
{
    graph_preparation_arg_t *arg = (graph_preparation_arg_t *)data;
    prepare_work_async_graphs(arg->size, arg->batch_index);
}

void graph_post_work (void *data)
{
    graph_preparation_arg_t *arg = (graph_preparation_arg_t *)data;
    wait_recv(arg->batch_index);
    wait_send(arg->batch_index);
    ++arg->batch_index;
}

void capture_async_graph (int size, long long int kernel_size) 
{
    stream_state_t *curr_stream = stream_state;
    
    cudaStreamBeginCapture(curr_stream->stream, cudaStreamCaptureModeGlobal);

    CUDA_CHECK(cudaLaunchHostFunc(curr_stream->stream, graph_prepare_work, graph_pre_arg));

    for (int j=0; j<steps_per_batch; j++) {
        if (!my_rank) {
            wait_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc_d, curr_stream->windex_d);
            CUDA_CHECK(cudaGetLastError());

            if (kernel_size > 0) {
                if (use_calc_kernel > 0)
                    gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                else
                    poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
            }

            CUDA_CHECK(cudaGetLastError());

            send_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc_d, curr_stream->sindex_d);
            CUDA_CHECK(cudaGetLastError());
        } else {
            send_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc_d, curr_stream->sindex_d);
            CUDA_CHECK(cudaGetLastError());

            wait_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc_d, curr_stream->windex_d);
            CUDA_CHECK(cudaGetLastError());

            if (kernel_size > 0) {
                if (use_calc_kernel > 0)
                    gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                else
                    poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaLaunchHostFunc(curr_stream->stream, graph_post_work, graph_pre_arg));

    cudaStreamEndCapture(curr_stream->stream, &graph);
    cudaGraphInstantiate(&graphexec, graph, NULL, NULL, 0);
 
    cudaStreamBeginCapture(curr_stream->stream, cudaStreamCaptureModeGlobal);
    CUDA_CHECK(cudaLaunchHostFunc(curr_stream->stream, graph_prepare_work, graph_pre_arg));
    for (int j=0; j<steps_per_batch; j++) {
        if (!my_rank) {
            wait_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc_d, curr_stream->windex_d);
            CUDA_CHECK(cudaGetLastError());

            send_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc_d, curr_stream->sindex_d);
            CUDA_CHECK(cudaGetLastError());
        } else {
            send_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc_d, curr_stream->sindex_d);
            CUDA_CHECK(cudaGetLastError());

            wait_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc_d, curr_stream->windex_d);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaLaunchHostFunc(curr_stream->stream, graph_post_work, graph_pre_arg));
    cudaStreamEndCapture(curr_stream->stream, &graph_comms);
    cudaGraphInstantiate(&graphexec_comms, graph_comms, NULL, NULL, 0);
}

void create_async_graph (size_t size, long long int kernel_size) 
{
    std::vector<cudaGraphNode_t> nodeDependencies, nodeDependencies2;
    cudaGraphNode_t sendNode, waitNode, kernelNode, preNode, postNode;
    cudaKernelNodeParams waitParams, sendParams, calcKernelParams, pollKernelParams;
    cudaHostNodeParams preParams, postParams;
    cudaGraphNode_t subgraphNode, subgraphNode_prev;

    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaGraphCreate(&graph_comms, 0));
    CUDA_CHECK(cudaGraphCreate(&subgraph, 0));
    CUDA_CHECK(cudaGraphCreate(&subgraph_comms, 0));

    preParams.fn = graph_prepare_work;
    preParams.userData = graph_pre_arg;

    postParams.fn = graph_post_work;
    postParams.userData = graph_pre_arg;

    for(int k=0; k<num_streams; k++) {
	stream_state_t *curr_stream = (stream_state + k); 

	waitParams.func = (void*)wait_op_kernel_graph;
        waitParams.gridDim = 1;
        waitParams.blockDim = 1;
        waitParams.sharedMemBytes = 0;
        void *waitArgs[2] = {(void*)&curr_stream->wdesc_d, (void *)&curr_stream->windex_d};
        waitParams.kernelParams = waitArgs;
        waitParams.extra = NULL;

        sendParams.func = (void*)send_op_kernel_graph;
        sendParams.gridDim = 1;
        sendParams.blockDim = 1;
        sendParams.sharedMemBytes = 0;
        void *sendArgs[2] = {(void*)&curr_stream->sdesc_d, (void *)&curr_stream->sindex_d};
        sendParams.kernelParams = sendArgs;
        sendParams.extra = NULL;

        const float value = 0.1F; 
        int n = kernel_size / sizeof(float);

        calcKernelParams.func = (void*)calc_kernel;
        calcKernelParams.gridDim = over_sub_factor * gpu_num_sm;
        calcKernelParams.blockDim = 32*2;
        calcKernelParams.sharedMemBytes = 0;
        void *calcKernelArgs[4] = {(void*)&n, (void *)&value, (void *)&curr_stream->in, (void *)&curr_stream->out};
        calcKernelParams.kernelParams = calcKernelArgs;
        calcKernelParams.extra = NULL;

        pollKernelParams.func = (void*)poll_kernel;
        pollKernelParams.gridDim = 1;
        pollKernelParams.blockDim = 1;
        pollKernelParams.sharedMemBytes = 0;
        void *pollKernelArgs[1] = {(void*)&kernel_size};
        pollKernelParams.kernelParams = pollKernelArgs;
        pollKernelParams.extra = NULL;

     	CUDA_CHECK(cudaGraphCreate(&curr_stream->subgraph, 0));
        CUDA_CHECK(cudaGraphCreate(&curr_stream->subgraph_comms, 0));

	//subgraph with comms + comp
        if (!my_rank) {
	   nodeDependencies.clear();
     	   CUDA_CHECK(cudaGraphAddKernelNode(&waitNode, curr_stream->subgraph, nodeDependencies.data(), 
	        		   nodeDependencies.size(), &waitParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(waitNode);
           if (use_calc_kernel > 0) {
               CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, curr_stream->subgraph, nodeDependencies.data(), 
	        		       nodeDependencies.size(), &calcKernelParams));
           } else { 
               CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, curr_stream->subgraph, nodeDependencies.data(), 
	        		       nodeDependencies.size(), &pollKernelParams));
           }

	   nodeDependencies.clear();
           nodeDependencies.push_back(kernelNode);
           CUDA_CHECK(cudaGraphAddKernelNode(&sendNode, curr_stream->subgraph, nodeDependencies.data(), 
			   	nodeDependencies.size(), &sendParams));
	} else {
	   nodeDependencies.clear();
           CUDA_CHECK(cudaGraphAddKernelNode(&sendNode, curr_stream->subgraph, nodeDependencies.data(), 
	     		   	nodeDependencies.size(), &sendParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(sendNode);
     	   CUDA_CHECK(cudaGraphAddKernelNode(&waitNode, curr_stream->subgraph, nodeDependencies.data(), 
	     		   nodeDependencies.size(), &waitParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(waitNode);
           if (use_calc_kernel > 0) {
               CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, curr_stream->subgraph, nodeDependencies.data(), 
	     		       nodeDependencies.size(), &calcKernelParams));
           } else { 
               CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, curr_stream->subgraph, nodeDependencies.data(), 
	     		       nodeDependencies.size(), &pollKernelParams));
           }
    	}

	//subgraph with comms
        if (!my_rank) {
	   nodeDependencies.clear();
     	   CUDA_CHECK(cudaGraphAddKernelNode(&waitNode, curr_stream->subgraph_comms, nodeDependencies.data(), 
	        		   nodeDependencies.size(), &waitParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(waitNode);
           CUDA_CHECK(cudaGraphAddKernelNode(&sendNode, curr_stream->subgraph_comms, nodeDependencies.data(), 
	     		   	nodeDependencies.size(), &sendParams));
	   
        } else {
	   nodeDependencies.clear();
           CUDA_CHECK(cudaGraphAddKernelNode(&sendNode, curr_stream->subgraph_comms, nodeDependencies.data(), 
	     		   	nodeDependencies.size(), &sendParams));

	   nodeDependencies.clear();
           nodeDependencies.push_back(sendNode);
     	   CUDA_CHECK(cudaGraphAddKernelNode(&waitNode, curr_stream->subgraph_comms, nodeDependencies.data(), 
	     		   nodeDependencies.size(), &waitParams));
           
    	}
    }

    //graph with compute and comms 
    CUDA_CHECK(cudaGraphAddEmptyNode(&emptyNode, subgraph, NULL, 0));
    nodeDependencies.clear();
    nodeDependencies2.clear();
    nodeDependencies.push_back(emptyNode);
    for(int k=0; k<num_streams; k++) {
       CUDA_CHECK(cudaGraphAddChildGraphNode(&subgraphNode, subgraph, nodeDependencies.data(), 
			       		nodeDependencies.size(), 
			       		stream_state[k].subgraph));
       nodeDependencies2.push_back(subgraphNode);
    }
    CUDA_CHECK(cudaGraphAddEmptyNode(&emptyNode, subgraph, nodeDependencies2.data(), nodeDependencies2.size()));

    //graph with comms
    CUDA_CHECK(cudaGraphAddEmptyNode(&emptyNode, subgraph_comms, NULL, 0));
    nodeDependencies.clear();
    nodeDependencies2.clear();
    nodeDependencies.push_back(emptyNode);
    for(int k=0; k<num_streams; k++) {
       CUDA_CHECK(cudaGraphAddChildGraphNode(&subgraphNode, subgraph_comms, nodeDependencies.data(), nodeDependencies.size(), 
			       		stream_state[k].subgraph_comms));
       nodeDependencies2.push_back(subgraphNode);
    }
    CUDA_CHECK(cudaGraphAddEmptyNode(&emptyNode, subgraph_comms, nodeDependencies2.data(), nodeDependencies2.size()));

    //create a graph for a batch of iterations
    //graph with compute and comms 
    CUDA_CHECK(cudaGraphAddHostNode(&preNode, graph, NULL, 0, &preParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(preNode);
    CUDA_CHECK(cudaGraphAddChildGraphNode(&subgraphNode_prev, graph, nodeDependencies.data(), nodeDependencies.size(), subgraph_comms)); 
    for (int k=1; k<steps_per_batch; k++) {
        nodeDependencies.clear();
        nodeDependencies.push_back(subgraphNode_prev);
        CUDA_CHECK(cudaGraphAddChildGraphNode (&subgraphNode, graph, nodeDependencies.data(), 
                    nodeDependencies.size(), subgraph));
        subgraphNode_prev = subgraphNode;
    }
    nodeDependencies.clear();
    nodeDependencies.push_back(subgraphNode_prev);
    CUDA_CHECK(cudaGraphAddHostNode(&postNode, graph, nodeDependencies.data(), nodeDependencies.size(), &postParams));
    CUDA_CHECK(cudaGraphInstantiate(&graphexec, graph, NULL, NULL, 0));

    //graph with comms 
    CUDA_CHECK(cudaGraphAddHostNode(&preNode, graph_comms, NULL, 0, &preParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(preNode);
    CUDA_CHECK(cudaGraphAddChildGraphNode(&subgraphNode_prev, graph_comms, nodeDependencies.data(), nodeDependencies.size(), subgraph_comms)); 
    for (int k=1; k<steps_per_batch; k++) {
        nodeDependencies.clear();
        nodeDependencies.push_back(subgraphNode_prev);
        CUDA_CHECK(cudaGraphAddChildGraphNode (&subgraphNode, graph_comms, nodeDependencies.data(), 
                    nodeDependencies.size(), subgraph_comms));
        subgraphNode_prev = subgraphNode;
    }
    nodeDependencies.clear();
    nodeDependencies.push_back(subgraphNode_prev);
    CUDA_CHECK(cudaGraphAddHostNode(&postNode, graph_comms, nodeDependencies.data(), nodeDependencies.size(), &postParams));
    CUDA_CHECK(cudaGraphInstantiate(&graphexec_comms, graph_comms, NULL, NULL, 0));
}

void destroy_async_graph () 
{
    CUDA_CHECK(cudaGraphExecDestroy(graphexec));
    CUDA_CHECK(cudaGraphExecDestroy(graphexec_comms));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphDestroy(graph_comms));
    if (!capture_graph) {
       CUDA_CHECK(cudaGraphDestroy(subgraph));
       CUDA_CHECK(cudaGraphDestroy(subgraph_comms));
       for(int k=0; k<num_streams; k++) {
           CUDA_CHECK(cudaGraphDestroy(stream_state[k].subgraph));
           CUDA_CHECK(cudaGraphDestroy(stream_state[k].subgraph_comms));
       }
    }
}

void trigger_work_async_kernels (int size, int batch_index, long long int kernel_size) 
{
    for (int j=0; j<steps_per_batch; j++) {
	CUDA_CHECK(cudaEventRecord(iter_start_event, main_stream));	
	for(int k=0; k<num_streams; k++) {
	    stream_state_t *curr_stream = (stream_state + k);   	

   	    CUDA_CHECK(cudaStreamWaitEvent(curr_stream->stream, iter_start_event, 0));	

	    if (!my_rank) {
	        wait_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc_d, curr_stream->windex_d);
                CUDA_CHECK(cudaGetLastError());

                if (kernel_size > 0) {
                    if (use_calc_kernel > 0)
                       gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                    else
                       poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
                }

	        send_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc_d, curr_stream->sindex_d);
                CUDA_CHECK(cudaGetLastError());
 	    } else {
	       send_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc_d, curr_stream->sindex_d);
               CUDA_CHECK(cudaGetLastError());

	       wait_op_kernel_graph<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc_d, curr_stream->windex_d);
               CUDA_CHECK(cudaGetLastError());

               if (kernel_size > 0) {
                   if (use_calc_kernel > 0)
                      gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                   else
                      poll_kernel <<<1, 1, 0, curr_stream->stream>>>(kernel_size);
               }
 	    }

	    CUDA_CHECK(cudaEventRecord(iter_stop_event, curr_stream->stream));
	    CUDA_CHECK(cudaStreamWaitEvent(main_stream, iter_stop_event, 0));
	}
    }
}

void post_work_async_graphs (int size, int batch_index, long long int kernel_size)
{
    //prepare_work_async_graphs (size, batch_index);

    if (kernel_size) 
        CUDA_CHECK(cudaGraphLaunch(graphexec, main_stream));
    else 
        CUDA_CHECK(cudaGraphLaunch(graphexec_comms, main_stream));
}

void post_work_async_kernels (int size, int batch_index, long long int kernel_size) 
{
    for (int j=0; j<steps_per_batch; j++) {
	CUDA_CHECK(cudaEventRecord(iter_start_event, main_stream));	  	

	for(int k=0; k<num_streams; k++) {
	    stream_state_t *curr_stream = (stream_state + k); 
            uint32_t r_idx = curr_stream->windex; 
            uint32_t s_idx = curr_stream->sindex; 
            curr_stream->windex = (curr_stream->windex+1)%windex_max;
            curr_stream->sindex = (curr_stream->sindex+1)%sindex_max;

      	    CUDA_CHECK(cudaStreamWaitEvent(curr_stream->stream, iter_start_event, 0));	

	    if (!my_rank) {
                MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->wdesc[r_idx], &curr_stream->rreq[r_idx]));
	        wait_op_kernel<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc[r_idx]);
                CUDA_CHECK(cudaGetLastError());
	        
                if (kernel_size > 0) {
                    if (use_calc_kernel > 0)
                       gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                    else
                       poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
                }

                MP_CHECK(mp_send_prepare((void *)stream_state[k].buf_d, size, peer*num_streams + k, 
	        			&curr_stream->reg, &curr_stream->sreq[s_idx]));
                MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->sdesc[s_idx], &curr_stream->sreq[s_idx]));
	        send_op_kernel<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc[s_idx]);
                CUDA_CHECK(cudaGetLastError());
 	    } else {
                MP_CHECK(mp_send_prepare((void *)stream_state[k].buf_d, size, peer*num_streams + k, 
	         		       &curr_stream->reg, &curr_stream->sreq[s_idx]));
                MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->sdesc[s_idx], &curr_stream->sreq[s_idx]));
	        send_op_kernel<<<1,1,0,curr_stream->stream>>>(curr_stream->sdesc[s_idx]);
                CUDA_CHECK(cudaGetLastError());

	        MP_CHECK(mp::mlx5::get_descriptors(&curr_stream->wdesc[r_idx], &curr_stream->rreq[r_idx]));
	        wait_op_kernel<<<1,1,0,curr_stream->stream>>>(curr_stream->wdesc[r_idx]);
                CUDA_CHECK(cudaGetLastError());

                if (kernel_size > 0) {
                    if (use_calc_kernel > 0)
                       gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                    else
                       poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
                }
 	    }

	    CUDA_CHECK(cudaEventRecord(iter_stop_event, curr_stream->stream));
	    CUDA_CHECK(cudaStreamWaitEvent(main_stream, iter_stop_event, 0));
	}
    }
}

void post_work_async (int size, int batch_index, long long int kernel_size) 
{
    int sreq_idx = batch_to_sreq_idx (batch_index);
    int rreq_idx = batch_to_rreq_idx (batch_index);
   
    for (int j=0; j<steps_per_batch; j++) {
	CUDA_CHECK(cudaEventRecord(iter_start_event, main_stream));
	int s_idx = sreq_idx + j;
	int r_idx = rreq_idx + j;

	for(int k=0; k<num_streams; k++) {
	    stream_state_t *curr_stream = (stream_state + k); 
  	    CUDA_CHECK(cudaStreamWaitEvent(curr_stream->stream, iter_start_event, 0));

	    if (!my_rank) { 
   	        MP_CHECK(mp_wait_on_stream(&curr_stream->rreq[r_idx], curr_stream->stream));

                if (kernel_size > 0) {
                    if (use_calc_kernel > 0)
                       gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                    else
                       poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
                }

                MP_CHECK(mp_isend_on_stream ((void *)curr_stream->buf_d, size, peer*num_streams + k, 
	        			&curr_stream->reg, &curr_stream->sreq[s_idx], 
	        			curr_stream->stream));
   	    } else {
                MP_CHECK(mp_isend_on_stream ((void *)curr_stream->buf_d, size, peer*num_streams + k, 
	        			&curr_stream->reg, &curr_stream->sreq[s_idx], 
	        			curr_stream->stream));

                MP_CHECK(mp_wait_on_stream(&curr_stream->rreq[r_idx], curr_stream->stream));

                if (kernel_size > 0) {
                    if (use_calc_kernel > 0)
                       gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                    else
                       poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
                }
	    }

	    CUDA_CHECK(cudaEventRecord(iter_stop_event, curr_stream->stream));
	    CUDA_CHECK(cudaStreamWaitEvent(main_stream, iter_stop_event, 0));
	}
   }
}

void post_work_sync (int size, int batch_index, long long int kernel_size) 
{
    int rreq_idx = batch_to_rreq_idx (batch_index);
    int sreq_idx = batch_to_sreq_idx (batch_index);

    for (int j=0; j<steps_per_batch; j++) {
	if (!my_rank) {
	    for(int k=0; k<num_streams; k++) {
	        stream_state_t *curr_stream = (stream_state + k); 

		MP_CHECK(mp_wait(&curr_stream->rreq[rreq_idx + j]));

                if (kernel_size > 0) {
                    if (use_calc_kernel > 0)
                       gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                    else
                       poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
                }
	    }

	    for(int k=0; k<num_streams; k++) { 
	        stream_state_t *curr_stream = (stream_state + k); 

		CUDA_CHECK(cudaStreamSynchronize(curr_stream->stream));

                MP_CHECK(mp_isend ((void *)curr_stream->buf_d, size, peer*num_streams + k, 
					&curr_stream->reg, &curr_stream->sreq[sreq_idx + j]));
	    }
       } else {
	    for(int k=0; k<num_streams; k++) { 
	        stream_state_t *curr_stream = (stream_state + k); 
                MP_CHECK(mp_isend ((void *)curr_stream->buf_d, size, peer*num_streams + k, 
					&curr_stream->reg, &curr_stream->sreq[sreq_idx + j]));
	    }

	    for(int k=0; k<num_streams; k++) { 
	        stream_state_t *curr_stream = (stream_state + k); 
                MP_CHECK(mp_wait(&curr_stream->rreq[rreq_idx + j]));

                if (kernel_size > 0) {
                    if (use_calc_kernel > 0)
                       gpu_launch_calc_kernel(kernel_size, curr_stream->stream);
                    else
                       poll_kernel <<<1, 1, 0, curr_stream->stream>>> (kernel_size);
                }
	    }

	    for(int k=0; k<num_streams; k++) { 
                CUDA_CHECK(cudaStreamSynchronize(stream_state[k].stream));
	    }
        }
    }
}

double prepost_latency;

double sr_exchange (MPI_Comm comm, int size, int iter_count, int print_times, long long int kernel_size, 
            int use_async, int use_kernel_ops = 0, int use_graphs = 0)
{
    double time_start, time_stop;
    float cputime_elapsed, streamtime_elapsed;
    int batch_count, wait_send_batch = 0, wait_recv_batch = 0;
    int j;

    if (iter_count%steps_per_batch != 0) { 
	fprintf(stderr, "iter_count must be a multiple of steps_per_batch: %d \n", steps_per_batch);
	exit(-1);
    }
    batch_count = iter_count/steps_per_batch;
    tracking_event = 0;

    for (int i=0; i<num_streams; i++) { 
        stream_state_t *curr_stream = (stream_state + i); 
        CUDA_CHECK(cudaMemset(curr_stream->buf_d, 0, size));
        curr_stream->sindex = curr_stream->windex = 0;
        CUDA_CHECK(cudaMemset((void *)curr_stream->sindex_d, 
            		    0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset((void *)curr_stream->windex_d, 
        		    0, sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    post_recv (size, 0);

    MPI_Barrier(MPI_COMM_WORLD);

    *delay_flag = 0;
    if (use_async) { 
        CU_CHECK(cuStreamWaitValue32(main_stream, (CUdeviceptr)delay_flag_dptr, 1, CU_STREAM_WAIT_VALUE_EQ));
        CUDA_CHECK(cudaEventRecord(timer_start_event, main_stream));
    }
    time_start = MPI_Wtime();

    if (use_graphs) {
        graph_pre_arg->size = size;
        graph_pre_arg->batch_index = 0;
    }

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

    *((volatile int *)delay_flag) = 1;

    wait_send_batch = wait_recv_batch = 0;
    while (wait_send_batch < batch_count) { 
        if (use_async) {
            if (!use_graphs)
                wait_recv (wait_recv_batch);
            wait_recv_batch++;
        }

        if (!use_graphs)
            wait_send (wait_send_batch);
        else
            CUDA_CHECK(cudaStreamSynchronize(main_stream));
            
        wait_send_batch++;

        if (j < (batch_count-1)) {
            post_recv (size, j+1);
        }

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

            j++;
        }
    }

    MPI_Barrier(comm);

    if (use_async) { 
        CUDA_CHECK(cudaEventRecord(timer_stop_event, main_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    if (use_async) { 
        CUDA_CHECK(cudaEventElapsedTime(&streamtime_elapsed, timer_start_event, timer_stop_event));
        streamtime_elapsed = ((streamtime_elapsed*1e3)/(iter_count*2)); 
    }
    time_stop = MPI_Wtime();
    cputime_elapsed = (time_stop - time_start);
    cputime_elapsed = ((cputime_elapsed*1e6)/((double)iter_count*2)); 

    if (print_times && !my_rank) {
        //if not using async, stream times do not make sense setting it from cputime
        if (!use_async) streamtime_elapsed = cputime_elapsed;
        fprintf(streamtimes, "%8.2lf \t", streamtime_elapsed);
        fprintf(cputimes, "%8.2lf \t", cputime_elapsed);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    return (double)cputime_elapsed;
}

int main (int argc, char *argv[])
{
    int iter_count, max_size, size, dev_count, local_rank, dev_id = 0;
    long long int kernel_size = 20;
    int comm_comp_ratio = 1;
    int user_iter_count = 0;
    size = 1;
    max_size = MAX_SIZE;
    char line[200];
    char *value;

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

    use_calc_kernel = 0;
    value = getenv("USE_CALC_KERNEL");
    if (value != NULL) {
        use_calc_kernel = atoi(value);
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

    value = getenv("USE_GRAPH_CAPTURE");
    if (value != NULL && atoi(value)) {
        capture_graph = atoi(value);
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

    dev_id = local_rank;
    fprintf(stdout, "[%d] local_rank: %d dev_count: %d using GPU device: %d\n", my_rank, local_rank, dev_count, dev_id);

    CUDA_CHECK(cudaSetDevice(dev_id));
    CUDA_CHECK(cudaFree(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0, cudaMemcpyHostToDevice));
    gpu_num_sm = prop.multiProcessorCount;

    CUDA_CHECK(cudaMemcpyToSymbol(my_rank_d, (void *)&my_rank, sizeof(int), 0, cudaMemcpyHostToDevice));

    fprintf(stdout, "[%d] GPU %d: %s PCIe %d:%d:%d\n", my_rank, dev_id, prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    peer = !my_rank;
    MP_CHECK(mp_init_multistream (MPI_COMM_WORLD, &peer, 1, MP_INIT_DEFAULT, dev_id, num_streams));

    iter_count = user_iter_count ? user_iter_count : ITER_COUNT_SMALL;
    if (!my_rank) { 
        cputimes = fopen("cputimes.txt", "w+");
        streamtimes = fopen("streamtimes.txt", "w+");
        if (!cputimes || !streamtimes)
        {
            printf("Could not open log file");
            exit(-1);
        }
        fprintf(stdout, "steps_per_batch: %d num_streams: %d batches_inflight: %d \n", 
                steps_per_batch, num_streams, batches_inflight);
        fprintf(stdout, "NOTE: printing half round-trip latency!!!\n");
    }

    stream_state = (stream_state_t *)malloc(sizeof(stream_state_t)*num_streams);
    NULL_CHECK(stream_state);

    graph_pre_arg = (graph_preparation_arg_t *)calloc(1, sizeof(graph_preparation_arg_t));
    NULL_CHECK(graph_pre_arg);

    sindex_max = batches_inflight*steps_per_batch;
    windex_max = (batches_inflight + 1)*steps_per_batch;
    CUDA_CHECK(cudaMemcpyToSymbol(sindex_max_d, (void *)&sindex_max, sizeof(int), 
                0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(windex_max_d, (void *)&windex_max, sizeof(int), 
                0, cudaMemcpyHostToDevice));

    /*allocating requests*/
    for (int i=0; i<num_streams; i++) { 
        stream_state_t *curr = (stream_state + i); 
        int inflight_send_ops = steps_per_batch*batches_inflight;
        int inflight_recv_ops = steps_per_batch*(batches_inflight+1);
        curr->sreq = (mp_request_t *) malloc(inflight_send_ops*sizeof(mp_request_t));
        NULL_CHECK(curr->sreq);
        curr->rreq = (mp_request_t *) malloc(inflight_recv_ops*sizeof(mp_request_t));
        NULL_CHECK(curr->rreq);
        CUDA_CHECK(cudaHostAlloc((void **)&curr->sdesc, inflight_send_ops*sizeof(mp::mlx5::send_desc_t), 0));
        CUDA_CHECK(cudaHostAlloc((void **)&curr->wdesc, inflight_recv_ops*sizeof(mp::mlx5::wait_desc_t), 0));
        memset((void *)curr->sdesc, 0, inflight_send_ops*sizeof(mp::mlx5::send_desc_t));
        memset((void *)curr->wdesc, 0, inflight_recv_ops*sizeof(mp::mlx5::wait_desc_t));
        CUDA_CHECK(cudaHostGetDevicePointer((void **)&curr->sdesc_d, curr->sdesc, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **)&curr->wdesc_d, curr->wdesc, 0));

        CUDA_CHECK(cudaMalloc((void **)&curr->in, kernel_size));
        CUDA_CHECK(cudaMalloc((void **)&curr->out, kernel_size));
        CUDA_CHECK(cudaMalloc((void **)&curr->sindex_d, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc((void **)&curr->windex_d, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset((void *)curr->in, 1, kernel_size));
        CUDA_CHECK(cudaMemset((void *)curr->out, 1, kernel_size));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_state[i].stream, cudaStreamNonBlocking));
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaHostAlloc(&delay_flag, sizeof(int), 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&delay_flag_dptr, (void *)delay_flag, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&iter_start_event, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&iter_stop_event, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&timer_start_event, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&timer_stop_event, 0));

    if (!my_rank) {
        sprintf(line, "%10s \t %10s \t %10s \t %10s \t  %10s \t %10s \t %10s \t %10s \t %10s \t %10s  \n", 
                "MessageSize", "CompSize/Time", "CPU", "CPU+Comp", "MP", "MP+Comp", "MP-SM", 
                "MP-SM+Comp", "MP-Graph", "MP-Graph+Comp");
        fprintf(cputimes, "%s", line);
        fprintf(streamtimes, "%s", line);
    }

    if (size != 1) size = max_size = size;
    for (; size<=max_size; size*=2) {
        double latency;

        if (!my_rank) { 
            fprintf(stdout, "run for size: %10d in progress \n", size);
            fflush(stdout);
        }

        if (size > 1024) {
            iter_count = user_iter_count ? user_iter_count : ITER_COUNT_LARGE;
        }

        buf_size = size;

        for (int i=0; i<num_streams; i++) { 
            CUDA_CHECK(cudaMalloc((void **)&stream_state[i].buf_d, buf_size));
            CUDA_CHECK(cudaMemset(stream_state[i].buf_d, 0, buf_size)); 
            MP_CHECK(mp_register(stream_state[i].buf_d, buf_size, &stream_state[i].reg));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);

        /*warmup base case*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 0/*print times*/, 0, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*calculate kenrel time based on base case latency*/
        latency = sr_exchange(MPI_COMM_WORLD, size, iter_count, 0/*print times*/, 0, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        if (use_calc_kernel) 
            kernel_size = calc_size; 
        else  
            kernel_size = (comm_comp_ratio > 0) ? comm_comp_ratio*(latency) : kernel_size;

        if (!my_rank) {
            fprintf(cputimes, "%10d \t %10lld \t", size, kernel_size);
            fprintf(streamtimes, "%10d \t %10lld \t", size, kernel_size);
        }

        /*create graph*/
        if (capture_graph) { 
            if (num_streams != 1) {
                fprintf(stderr, "graph capture is only supported with single stream \n");
                exit (-1);
            }
            capture_async_graph (size, kernel_size); 
        } else { 
            create_async_graph (size, kernel_size);
        }

        cudaProfilerStart();

        /*warmup for all variants*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 0/*print times*/, kernel_size, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        sr_exchange(MPI_COMM_WORLD, size, iter_count, 0/*print times*/, kernel_size, 1/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        sr_exchange(MPI_COMM_WORLD, size, iter_count, 0/*print times*/, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/);

        MPI_Barrier(MPI_COMM_WORLD);

        sr_exchange(MPI_COMM_WORLD, size, iter_count, 0/*print times*/, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/, 1/*use_graphs*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Timed runs*/
        /*Normal*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, 0 /*no kernel*/, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Normal + Kernel*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, kernel_size, 0/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Async*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, 0/*kernel_size*/, 1/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Async + kernel*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, kernel_size, 1/*use_async*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Async + Kernel Ops*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, 0 /*kernel_size*/, 1/*use_async*/, 1/*use_kernel_ops*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Async + Kernel + Kernel Ops*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Async + Kernel Ops + Graphs*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, 0 /*kernel_size*/, 1/*use_async*/, 1/*use_kernel_ops*/, 1/*use_graphs*/);

        MPI_Barrier(MPI_COMM_WORLD);

        /*Async + Kernel + Kernel Ops + Graphs*/
        sr_exchange(MPI_COMM_WORLD, size, iter_count, 1/*print times*/, kernel_size, 1/*use_async*/, 1/*use_kernel_ops*/, 1/*use_graphs*/);

        MPI_Barrier(MPI_COMM_WORLD);

        //cudaProfilerStop();

        if (!my_rank) {
            fprintf(streamtimes, "\n");
            fprintf(cputimes, "\n");
        }

        /*destroy graphs*/
        destroy_async_graph();

        for (int i=0; i<num_streams; i++) {
            mp_deregister(&stream_state[i].reg);
            CUDA_CHECK(cudaFree(stream_state[i].buf_d));
        }
    }

    if (!my_rank) {
        fprintf(cputimes, " \n");
        fprintf(streamtimes, " \n");

        rewind(cputimes);
        rewind(streamtimes);

        fprintf(stdout, "******************** CPU Timing ********************** \n");

        while(fgets(line, 200, cputimes) != NULL) 
            fprintf(stdout, "%s", line);
        fprintf(stdout, "\n");

        fprintf(stdout, "******************** STREAM Timing ********************** \n");

        while(fgets(line, 200, streamtimes) != NULL) 
            fprintf(stdout, "%s", line);
        fprintf(stdout, "\n");
    }

    for (int i=0; i<num_streams; i++) { 
        CUDA_CHECK(cudaStreamDestroy(stream_state[i].stream));
        free(stream_state[i].sreq);
        free(stream_state[i].rreq);
    }
    CUDA_CHECK(cudaStreamDestroy(main_stream));
    free(graph_pre_arg);
    free(stream_state);
    if (!my_rank) { 
        fclose(streamtimes);
        fclose(cputimes);
    }
    mp_finalize ();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
