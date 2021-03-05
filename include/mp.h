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

#pragma once

#include <sys/uio.h> // for struct iov
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define MP_SUCCESS 0
#define MP_FAILURE 1

#define MP_API_MAJOR_VERSION    2
#define MP_API_MINOR_VERSION    0
#define MP_API_VERSION          ((MP_API_MAJOR_VERSION << 16) | MP_API_MINOR_VERSION)

#define MP_API_VERSION_COMPATIBLE(v) \
    ( ((((v) & 0xffff0000U) >> 16) == MP_API_MAJOR_VERSION) &&   \
      ((((v) & 0x0000ffffU) >> 0 ) >= MP_API_MINOR_VERSION) )


#ifdef __cplusplus
extern "C" 
{ 
#endif

typedef enum mp_param {
    MP_PARAM_VERSION,
    MP_NUM_PARAMS
} mp_param_t;

int mp_query_param(mp_param_t param, int *value);

struct mp_reg; 
struct mp_request; 
typedef struct mp_reg* mp_reg_t;
typedef struct mp_request* mp_request_t; 
typedef struct mp_window* mp_window_t;

enum mp_init_flags {
    MP_INIT_DEFAULT = 0,
    MP_INIT_WQ_ON_GPU,
    MP_INIT_RX_CQ_ON_GPU,
    MP_INIT_TX_CQ_ON_GPU,
    MP_INIT_DBREC_ON_GPU
};

/**
 * \brief Initialize the MP library
 *
 * \param comm - MPI communicator to use to bootstrap connection establishing
 * \param peers - array of MPI ranks with which to establish a connection
 * \param count - size of peers array
 * \param flags - combination of mp_init_flags
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_init(MPI_Comm comm, int *peers, int count, int flags, int gpu_id);
int mp_init_multistream(MPI_Comm comm, int *peers, int count, int flags, int gpu_id, int streams_per_rank);
void mp_finalize();

int mp_register(void *addr, size_t length, mp_reg_t *reg_t);
int mp_deregister(mp_reg_t *reg);

/*
 * regular, CPU synchronous primitives
 *
 */

/**
 * \brief Issue an asynchronous receive
 * \param peer - MPI rank of peer
 */
int mp_irecv (void *buf, int size, int peer, mp_reg_t *mp_reg, mp_request_t *req);
int mp_isend (void *buf, int size, int peer, mp_reg_t *mp_reg, mp_request_t *req);

int mp_test(mp_request_t *req);
int mp_wait (mp_request_t *req);
int mp_wait_all (uint32_t count, mp_request_t *req);
int mp_progress_all (uint32_t count, mp_request_t *req);


/*
 * CUDA stream synchronous primitives
 */
int mp_send_on_stream  (void *buf, int size, int peer, mp_reg_t *mp_reg,
                        mp_request_t *req, cudaStream_t stream);
int mp_isend_on_stream (void *buf, int size, int peer, mp_reg_t *mp_reg,
                        mp_request_t *req, cudaStream_t stream);
//int mp_irecv_on_stream (void *buf, int size, int peer, mp_reg_t *mp_reg,
//                        mp_request_t *req, cudaStream_t stream);

/* vector sends/recvs
 * caveats: all blocks are within same registration
 */
int mp_isendv(struct iovec *v, int nblocks, int peer, mp_reg_t *mp_reg, mp_request_t *req);
int mp_irecvv(struct iovec *v, int nblocks, int peer, mp_reg_t *mp_reg, mp_request_t *req);

int mp_isendv_on_stream (struct iovec *v, int nblocks, int peer, mp_reg_t *mp_reg,
			 mp_request_t *req, cudaStream_t stream);

/*
 * GPU synchronous functions
 */
int mp_wait_on_stream (mp_request_t *req, cudaStream_t stream);
int mp_wait_all_on_stream (uint32_t count, mp_request_t *req, cudaStream_t stream);


/* Split API to allow for batching of operations issued to the GPU
 */

int mp_send_prepare (void *buf, int size, int peer, mp_reg_t *mp_reg,
            mp_request_t *req);
int mp_sendv_prepare (struct iovec *v, int nblocks, int peer, mp_reg_t *mp_reg,
                         mp_request_t *req);

int mp_send_post_on_stream (mp_request_t *req, cudaStream_t stream);
int mp_isend_post_on_stream (mp_request_t *req, cudaStream_t stream);
int mp_send_post_all_on_stream (uint32_t count, mp_request_t *req, cudaStream_t stream);
int mp_isend_post_all_on_stream (uint32_t count, mp_request_t *req, cudaStream_t stream);

/*
 * One-sided communication primitives
 */

/* window creation */
int mp_window_create(void *addr, size_t size, mp_window_t *window_t);
int mp_window_destroy(mp_window_t *window_t);

enum mp_put_flags {
    MP_PUT_INLINE  = 1<<0,
    MP_PUT_NOWAIT  = 1<<1, // don't generate a CQE, req cannot be waited for
};

int mp_iput (void *src, int size, mp_reg_t *src_reg, int peer, size_t displ, mp_window_t *dst_window_t, mp_request_t *req, int flags);
int mp_iget (void *dst, int size, mp_reg_t *dst_reg, int peer, size_t displ, mp_window_t *src_window_t, mp_request_t *req);

int mp_iput_on_stream (void *src, int size, mp_reg_t *src_reg, int peer, size_t displ, mp_window_t *dst_window_t, mp_request_t *req, int flags, cudaStream_t stream);

int mp_put_prepare (void *src, int size, mp_reg_t *src_reg, int peer, size_t displ, mp_window_t *dst_window_t, mp_request_t *req, int flags);

int mp_iput_post_on_stream (mp_request_t *req, cudaStream_t stream);

int mp_iput_post_all_on_stream (uint32_t count, mp_request_t *req, cudaStream_t stream);

/*
 * Memory related primitives
 */

enum mp_wait_flags {
    MP_WAIT_GEQ = 0,
    MP_WAIT_EQ,
    MP_WAIT_AND,
};

int mp_wait32(uint32_t *ptr, uint32_t value, int flags);
int mp_wait32_on_stream(uint32_t *ptr, uint32_t value, int flags, cudaStream_t stream);

static inline int mp_wait_dword_geq_on_stream(uint32_t *ptr, uint32_t value, cudaStream_t stream)
{
    return mp_wait32_on_stream(ptr, value, MP_WAIT_GEQ, stream);
}

static inline int mp_wait_dword_eq_on_stream(uint32_t *ptr, uint32_t value, cudaStream_t stream)
{
    return mp_wait32_on_stream(ptr, value, MP_WAIT_EQ, stream);
}

/*
 *
 */

typedef struct mp_desc_queue *mp_desc_queue_t;

int mp_desc_queue_alloc(mp_desc_queue_t *dq);
int mp_desc_queue_free(mp_desc_queue_t *dq);
int mp_desc_queue_add_send(mp_desc_queue_t *dq, mp_request_t *req);
int mp_desc_queue_add_wait_send(mp_desc_queue_t *dq, mp_request_t *req);
int mp_desc_queue_add_wait_recv(mp_desc_queue_t *dq, mp_request_t *req);
int mp_desc_queue_add_wait_value32(mp_desc_queue_t *dq, uint32_t *ptr, uint32_t value, int flags);
int mp_desc_queue_add_write_value32(mp_desc_queue_t *dq, uint32_t *ptr, uint32_t value);
int mp_desc_queue_post_on_stream(cudaStream_t stream, mp_desc_queue_t *dq, int flags);


/**
 * Graph and CUDA-kernel related primitives
 */

typedef struct mp_kernel_gs* mp_kernel_gs_t;
typedef uint32_t mp_gs_req_t;


/**
 * \brief Set up `gs` to for the specified `graph`.
 * \param graph - Graph to be used for this mp communication.
 * \param max_num_send - Maximum number of inflight send requests.
 * \param max_num_recv - Maximum number of inflight receive requests.
 * \param peer - Peer number to be associated with this mp communication.
 * \param gs - Return mp_kernel_gs_t object to be used with other mp_graph_* API.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_graph_setup(cudaGraph_t graph, uint32_t max_num_send, uint32_t max_num_recv, int peer, mp_kernel_gs_t *gs);

/**
 * \brief Create a graph intent for mp communication preparation. This function must be called before mp_graph_add_*_node.
 * \param gs - mp_kernel_gs_t object.
 * \param dependencies - Graph nodes that must be executed before launching the preparation.
 * \param dep_size - Number of elements in `dependencies`.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_graph_begin(mp_kernel_gs_t gs, cudaGraphNode_t *dependencies, size_t dep_size);

/**
 * \brief Create a graph intent for mp communication wrapup.
 *      mp_graph_add_*_node cannot be used after this function. This function must
 *      be called if `mp_graph_begin` is called.
 * \param gs - mp_kernel_gs_t object.
 * \param dependencies - Graph nodes that must be executed before the wrapup. 
 * \param dep_size - Number of elements in `dependencies`.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_graph_end(mp_kernel_gs_t gs, cudaGraphNode_t *dependencies, size_t dep_size);

/**
 * \brief Create and add an mp-isend graph node on the graph.
 * \param gs - mp_kernel_gs_t object.
 * \param dependencies - Dependencies for this mp-isend graph node. 
 * \param dep_size - Number of elements in `dependencies`.
 * \param snode - Return this mp-isend graph node.
 * \param sreq - Return the gs request to be used in mp_graph_add_wait_node.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_graph_add_isend_node(mp_kernel_gs_t gs, cudaGraphNode_t *dependencies, size_t dep_size, cudaGraphNode_t *snode, mp_gs_req_t *sreq);

/**
 * \brief Create and add an mp-irecv graph node on the graph.
 * \param gs - mp_kernel_gs_t object.
 * \param dependencies - Dependencies for this mp-irecv graph node. 
 * \param dep_size - Number of elements in `dependencies`.
 * \param rnode - Return this mp-irecv graph node.
 * \param rreq - Return the gs request to be used in mp_graph_add_wait_node.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_graph_add_irecv_node(mp_kernel_gs_t gs, cudaGraphNode_t *dependencies, size_t dep_size, cudaGraphNode_t *rnode, mp_gs_req_t *rreq);

/**
 * \brief Create and add an mp-wait graph node on the graph.
 * \param gs - mp_kernel_gs_t object.
 * \param req - gs request object to wait.
 * \param dependencies - Dependencies for this mp-wait graph node. 
 * \param dep_size - Number of elements in `dependencies`.
 * \param wnode - Return this mp-wait graph node.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_graph_add_wait_node(mp_kernel_gs_t gs, mp_gs_req_t req, cudaGraphNode_t *dependencies, size_t dep_size, cudaGraphNode_t *wnode);


/**
 * \brief Setup `gs` for the specified `stream`.
 * \param stream - cudaStream for this `gs`. The `stream` must be being captured.
 * \param max_num_send - Maximum number of inflight send requests.
 * \param max_num_recv - Maximum number of inflight receive requests.
 * \param peer - Peer number to be associated with this mp communication.
 * \param gs - Return mp_kernel_gs_t object to be used with other mp_kernstream_* API.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_kernstream_setup(cudaStream_t stream, uint32_t max_num_send, uint32_t max_num_recv, int peer, mp_kernel_gs_t *gs);

/**
 * \brief Add mp preparation phase to the stream.
 * \param gs - mp_kernel_gs_t object.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_kernstream_begin(mp_kernel_gs_t gs);

/**
 * \brief Add mp wrapup phase to the stream.
 * \param gs - mp_kernel_gs_t object.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_kernstream_end(mp_kernel_gs_t gs);

/**
 * \brief Add CUDA-kernel mp-isend to the stream.
 * \param gs - mp_kernel_gs_t object.
 * \param sreq - Return mp_gs_req_t for later use in mp_kernstream_wait.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_kernstream_isend(mp_kernel_gs_t gs, mp_gs_req_t *sreq);

/**
 * \brief Add CUDA-kernel mp-irecv to the stream.
 * \param gs - mp_kernel_gs_t object.
 * \param rreq - Return mp_gs_req_t for later use in mp_kernstream_wait.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_kernstream_irecv(mp_kernel_gs_t gs, mp_gs_req_t *rreq);

/**
 * \brief Add CUDA-kernel mp-wait to the stream.
 * \param gs - mp_kernel_gs_t object.
 * \param req - Wait on the specified `req`.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_kernstream_wait(mp_kernel_gs_t gs, mp_gs_req_t req);

/**
 * \brief Notify mp that the stream has been converted to `graph`. If this
 *      function returns successfully, `gs` is compatible with mp_graph_*.
 * \param gs - mp_kernel_gs_t object.
 * \param graph - cudaGraph object that the `stream` has been converted to.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_kernstream_graph_init(mp_kernel_gs_t gs, cudaGraph_t graph);

/**
 * \brief Clean up and free `gs`.
 * \param gs - mp_kernel_gs_t object.
 *
 * \return MP_SUCCESS, MP_FAILURE
 */
int mp_gs_cleanup(mp_kernel_gs_t gs);

#ifdef __cplusplus
}
#endif

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */

