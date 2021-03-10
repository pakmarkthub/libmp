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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mp.h"
#include "mp/device.cuh"
#include "mp_internal.h"

#include <gdsync/mlx5.h>

namespace mp {
    namespace mlx5 {

        int get_descriptors(isem32_t *psem, uint32_t *ptr, uint32_t value)
        {
            int ret = 0;
            struct gds_mlx5_dword_wait_info mlx5_i;
            int flags = GDS_MEMORY_HOST;
            int retcode = gds_mlx5_get_dword_wait_info(ptr, value, flags, &mlx5_i);
            if (retcode) {
                mp_err_msg("got error %d/%s \n", retcode, strerror(retcode));
                ret = MP_FAILURE;
                goto out;
            }
            mp_dbg_msg("wait_info ptr=%p value=%08x\n", mlx5_i.ptr, mlx5_i.value);
            psem->ptr   = mlx5_i.ptr;
            psem->value = mlx5_i.value;
        out:
            return ret;
        }

        int get_descriptors(send_desc_t *sinfo, mp_request_t *req_t)
        {
            int retcode;
            int ret = 0; 
            struct mp_request *req = *req_t;

            mp_dbg_msg("req=%p status=%d id=%d\n", req, req->status, req->id);

            // track req
            assert(req->status == MP_PREPARED);
            if (use_event_sync) {
                mp_err_msg("unsupported call in async event mode\n"); 
                ret = MP_FAILURE;
                goto out;
            }
            req->status = MP_PENDING_NOWAIT;
            req->stream = 0;

            // get_descriptors in sinfo
            struct gds_mlx5_send_info mlx5_i;
            retcode = gds_mlx5_get_send_info(1, &req->gds_send_info, &mlx5_i);
            if (retcode) {
                mp_err_msg("got error %d/%s \n", retcode, strerror(retcode));
                ret = MP_FAILURE;
                goto out;
            }
            sinfo->dbrec.ptr   = mlx5_i.dbrec_ptr;
            sinfo->dbrec.value = mlx5_i.dbrec_value;
            sinfo->db.ptr      = mlx5_i.db_ptr;
            sinfo->db.value    = mlx5_i.db_value;

        out:
            return ret;
        }

        int get_descriptors(wait_desc_t *winfo, mp_request_t *req_t)
        {
            int retcode;
            int ret = 0;
            struct mp_request *req = *req_t;
            client_t *client = &clients[client_index[req->peer]];

            mp_dbg_msg("req=%p status=%d id=%d\n", req, req->status, req->id);

            assert(req->status == MP_PENDING_NOWAIT || req->status == MP_COMPLETE);
	
            req->stream = 0;
            req->status = MP_PENDING;
            
            gds_mlx5_wait_info_t mlx5_i;
            retcode = gds_mlx5_get_wait_info(1, &req->gds_wait_info, &mlx5_i);
            if (retcode) {
                mp_err_msg("error %d\n", retcode);
                ret = MP_FAILURE;
                // BUG: leaking req ??
                goto out;
            }
            // BUG: need a switch() here
            winfo->sema_cond  = mlx5_i.cond;
            winfo->sema.ptr   = mlx5_i.cqe_ptr;
            winfo->sema.value = mlx5_i.cqe_value;
            winfo->flag.ptr   = mlx5_i.flag_ptr;
            winfo->flag.value = mlx5_i.flag_value;

        out:
            return ret;
        }

    }
}

int mp_gs_add_isend_node(mp_gs_t gs, void **buf, int *size, mp_reg_t *reg, cudaGraph_t graph, cudaGraphNode_t *dependencies, size_t dep_size, cudaGraphNode_t *snode, mp_gs_req_t *sreq)
{
    int ret = 0;

    cudaError_t cuda_result;

    cudaGraphNode_t node;
    cudaKernelNodeParams params;

    struct mp_gs_req _sreq;

    void *args[3];

    if (gs->sindex >= gs->max_num_send) {
        mp_dbg_msg("No more slot to hold this in-flight send.\n");
        ret = ENOMEM;
        goto out;
    }

    params.func = (void *)mp::device::mlx5::send_op_kernel;
    params.gridDim = 1;
    params.blockDim = 1;
    params.sharedMemBytes = 0;

    args[0] = (void *)gs->sdesc_d;
    args[1] = (void *)gs->sindex_d;
    args[2] = (void *)gs->max_num_send_d;

    params.kernelParams = args;
    params.extra = NULL;

    cuda_result = cudaGraphAddKernelNode(&node, graph, dependencies, dep_size, &params);
    if (cuda_result != cudaSuccess) {
        mp_dbg_msg("Error in cudaGraphAddHostNode: %s\n", cudaGetErrorName(cuda_result));
        ret = EINVAL;
        goto out;
    }

    gs->send_params[gs->sindex].buf = buf;
    gs->send_params[gs->sindex].size = size;
    gs->send_params[gs->sindex].reg = reg;

    gs->send_nodes[gs->sindex] = node;
    *snode = node;

    _sreq.type = MP_GS_REQ_TYPE_SEND;
    _sreq.index = gs->sindex;
    *sreq = (mp_gs_req_t)_sreq;

    ++gs->sindex;

out:
    return ret;
}

int mp_gs_add_irecv_node(mp_gs_t gs, void **buf, int *size, mp_reg_t *reg, cudaGraph_t graph, cudaGraphNode_t *dependencies, size_t dep_size, cudaGraphNode_t *rnode, mp_gs_req_t *rreq)
{
    int ret = 0;

    cudaError_t cuda_result;

    cudaGraphNode_t node;

    struct mp_gs_req _rreq;

    if (gs->rindex >= gs->max_num_recv) {
        mp_dbg_msg("No more slot to hold this in-flight recv.\n");
        ret = ENOMEM;
        goto out;
    }

    cuda_result = cudaGraphAddEmptyNode(&node, graph, dependencies, dep_size, &params);
    if (cuda_result != cudaSuccess) {
        mp_dbg_msg("Error in cudaGraphAddHostNode: %s\n", cudaGetErrorName(cuda_result));
        ret = EINVAL;
        goto out;
    }

    gs->recv_params[gs->rindex].buf = buf;
    gs->recv_params[gs->rindex].size = size;
    gs->recv_params[gs->rindex].reg = reg;

    gs->recv_nodes[gs->rindex] = node;
    *snode = node;

    _rreq.type = MP_GS_REQ_TYPE_RECV;
    _rreq.index = gs->rindex;
    *rreq = (mp_gs_req_t)_rreq;

    ++gs->rindex;

out:
    return ret;
}

int mp_gs_add_wait_node(mp_gs_t gs, mp_gs_req_t req, cudaGraph_t graph, cudaGraphNode_t *dependencies, size_t dep_size, cudaGraphNode_t *wnode)
{
    int ret = 0;

    cudaError_t cuda_result;

    cudaGraphNode_t node;
    cudaKernelNodeParams params;

    struct mp_gs_req _req = (struct mp_gs_req)req;

    void *args[3];

    if (gs->windex >= gs->max_num_wait) {
        mp_dbg_msg("No more slot to hold this in-flight wait.\n");
        ret = ENOMEM;
        goto out;
    }

    params.func = (void *)mp::device::mlx5::wait_op_kernel;
    params.gridDim = 1;
    params.blockDim = 1;
    params.sharedMemBytes = 0;

    args[0] = (void *)gs->wdesc_d;
    args[1] = (void *)gs->windex_d;
    args[2] = (void *)gs->max_num_wait_d;

    params.kernelParams = args;
    params.extra = NULL;

    if ((_req.type == MP_GS_REQ_TYPE_SEND && _req.index > gs->sindex) || (_req.type == MP_GS_REQ_TYPE_RECV && _req.index > gs->rindex)) {
        mp_dbg_msg("req not found.\n");
        ret = EINVAL;
        goto out;
    }

    cuda_result = cudaGraphAddKernelNode(&node, graph, dependencies, dep_size, &params);
    if (cuda_result != cudaSuccess) {
        mp_dbg_msg("Error in cudaGraphAddHostNode: %s\n", cudaGetErrorName(cuda_result));
        ret = EINVAL;
        goto out;
    }

    gs->wait_params[gs->windex].req = _req;

    gs->wait_nodes[gs->windex] = node;
    *wnode = node;

    ++gs->windex;

out:
    return ret;
}

