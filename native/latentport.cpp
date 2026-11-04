#include "latentport.h"
#include "layout.h"
#include "speculation.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

static_assert(sizeof(lp_field)==24 && sizeof(lp_info)==56 && sizeof(lp_receipt)==24 && sizeof(lp_trace)==96,
              "LatentPort ABI requires the documented 64-bit layouts");

namespace {
thread_local std::string error;
std::mutex registry_mutex;
std::unordered_set<void *> registry;
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct port_context;
}

// Only this class has access to the pinned cache internals. No serialized llama
// blob is mistaken for a typed, portable cross-model state.
struct latentport_access {
    static ggml_tensor * tensor(llama_kv_cache * kv, int layer, bool value) {
        require(!kv->v_trans && !kv->attn_rot_k && !kv->attn_rot_v,
                "LatentPort needs non-transposed, unrotated BF16 attention storage");
        auto it=kv->map_layer_ids.find(layer);
        require(it!=kv->map_layer_ids.end(), "attention layer missing");
        auto & l=kv->layers.at(it->second);
        return value?l.v_stream.at(0):l.k_stream.at(0);
    }
    static void install_clock(llama_kv_cache * kv,int clock) {
        require(kv->n_stream==1 && clock<=(int)kv->get_size(), "invalid attention clock");
        auto & cells=kv->v_cells.at(0);
        for(int i=0;i<clock;++i){cells.pos_set(i,i);cells.seq_add(i,0);cells.ext_set(i,{i,i,LLAMA_TOKEN_NULL});}
        kv->v_heads.at(0)=clock%kv->get_size();
    }
};

namespace {
struct port_context {
    std::array<llama_model_tensor_buft_override,2> gpu_weights{};
    llama_model * model=nullptr;
    llama_context * ctx=nullptr, * draft=nullptr;
    llama_memory_hybrid * mem=nullptr;
    ggml_context * aux=nullptr, * head_ctx=nullptr;
    ggml_backend_buffer_t aux_buf=nullptr;
    ggml_backend_sched_t head_sched=nullptr;
    ggml_cgraph * head_graph=nullptr;
    ggml_tensor * terminal=nullptr, * h_tap=nullptr, * head_in=nullptr, * head_out=nullptr;
    std::vector<ggml_tensor *> raw,c4,before;
    std::vector<float> logits,pending_h;
    std::vector<int32_t> forced_drafts;
    lp_info info{};
    lp_trace trace{};
    int clock=0,batch=0,tap_capacity=0,last_count=0;
    bool ready=false,pending=false,poisoned=false;
    ~port_context(){
        {std::lock_guard<std::mutex> lock(registry_mutex);registry.erase(this);}
        if(head_sched)ggml_backend_sched_free(head_sched);
        if(head_ctx)ggml_free(head_ctx);
        if(draft)llama_free(draft);
        if(ctx)llama_free(ctx);
        if(aux_buf)ggml_backend_buffer_free(aux_buf);
        if(aux)ggml_free(aux);
        if(model)llama_model_free(model);
    }
    llama_memory_recurrent * rs(){return mem->get_mem_recr();}
    llama_kv_cache * kv(){return mem->get_mem_attn();}
    void health(){require(!poisoned,"Native transaction failed; reinstall a public snapshot before reuse");}
    void init_aux(){
        aux=ggml_init({ggml_tensor_overhead()*(info.layers*3+8),nullptr,true});
        require(aux,"auxiliary graph metadata allocation failed");
        raw.resize(info.layers);c4.resize(info.layers);before.resize(info.layers);
        terminal=ggml_new_tensor_1d(aux,GGML_TYPE_F32,info.hidden);
        h_tap=ggml_new_tensor_2d(aux,GGML_TYPE_F32,info.hidden,batch);
        for(int i=0;i<info.layers;++i)if(model->hparams.is_recr(i)){
            raw[i]=ggml_new_tensor_2d(aux,GGML_TYPE_F32,info.conv_channels,tap_capacity);
            c4[i]=ggml_new_tensor_2d(aux,GGML_TYPE_F32,4,info.conv_channels);
            before[i]=ggml_new_tensor_2d(aux,GGML_TYPE_F32,4,info.conv_channels);
        }
        aux_buf=ggml_backend_alloc_ctx_tensors_from_buft(aux,ggml_backend_cuda_buffer_type(0));
        require(aux_buf,"CUDA auxiliary allocation failed");
        ggml_backend_buffer_clear(aux_buf,0);
        info.auxiliary_bytes=ggml_backend_buffer_get_size(aux_buf);
    }
    void init_head(){
        auto sched=ctx->get_sched();
        std::vector<ggml_backend_t> backends;
        for(int i=0;i<ggml_backend_sched_get_n_backends(sched);++i)backends.push_back(ggml_backend_sched_get_backend(sched,i));
        head_sched=ggml_backend_sched_new(backends.data(),nullptr,(int)backends.size(),64,false,true);
        head_ctx=ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(64,false),nullptr,true});
        require(head_ctx&&head_sched,"LM-head graph allocation failed");
        head_graph=ggml_new_graph_custom(head_ctx,64,false);
        head_in=ggml_new_tensor_2d(head_ctx,GGML_TYPE_F32,info.hidden,1);
        ggml_set_input(head_in);
        // Inputs otherwise default to pinned host memory, even when the
        // matrix multiply is offloaded. Translated H is already on CUDA.
        ggml_backend_sched_set_tensor_backend(head_sched,head_in,backends.front());
        auto norm=ggml_rms_norm(head_ctx,head_in,model->hparams.f_norm_rms_eps);
        norm=ggml_mul(head_ctx,norm,model->output_norm);
        head_out=ggml_mul_mat(head_ctx,model->output,norm);
        if(model->output_s)head_out=ggml_mul(head_ctx,head_out,model->output_s);
        ggml_set_output(head_out);
        ggml_build_forward_expand(head_graph,head_out);
        require(ggml_backend_sched_alloc_graph(head_sched,head_graph),"LM-head GPU graph allocation failed");
        require(!ggml_backend_buffer_is_host(head_in->buffer),"LM-head input unexpectedly on host");
    }
    void compute_head(){
        double start=now();
        lp_copy(head_in->data,terminal->data,info.hidden*sizeof(float));lp_device_sync();
        require(ggml_backend_sched_graph_compute(head_sched,head_graph)==GGML_STATUS_SUCCESS,"LM-head graph failed");
        ggml_backend_tensor_get(head_out,logits.data(),0,logits.size()*sizeof(float));ready=true;
        trace.last_head_s=now()-start;++trace.head_calls;
    }
    void save_before(){
        for(int i=0;i<info.layers;++i)if(c4[i])lp_copy(before[i]->data,c4[i]->data,ggml_nbytes(c4[i]));
    }
    void finish(int consumed,int accepted,int output_row,bool all_outputs=false){
        int raw_count=std::min(consumed,tap_capacity),start=consumed-raw_count;
        for(int i=0;i<info.layers;++i)if(c4[i])
            lp_conv_update((float*)c4[i]->data,(float*)before[i]->data,(float*)raw[i]->data,
                           info.conv_channels,raw_count,accepted,start);
        lp_copy(terminal->data,(float*)h_tap->data+(size_t)output_row*info.hidden,info.hidden*sizeof(float));
        lp_device_sync();
        // _ith indexes the original batch, not the packed output tensor.
        // Ordinary prefill requests only its final row; its packed H row is 0.
        const int batch_row=all_outputs?output_row:-1;
        const float * l=llama_get_logits_ith(ctx,batch_row);
        require(l,"missing committed target logits");std::copy(l,l+info.vocabulary,logits.begin());
        const float * h=llama_get_embeddings_nextn_ith(ctx,batch_row);
        require(h,"missing native MTP hidden representation");std::copy(h,h+info.hidden,pending_h.begin());
        pending=true;ready=true;clock+=accepted;
    }
    void rollback(int keep,int processed){
        if(keep==processed)return;
        const int new_clock=clock+keep;
        require(llama_memory_seq_rm(llama_get_memory(ctx),0,new_clock,-1),"target recurrent rollback rejected");
        auto r=rs();int tail=r->cells[0].tail;
        require(tail>=0 && r->rs_idx[0]==(uint32_t)(processed-keep),"recurrent rollback slot mismatch");
        // Materialize pending rollback before publishing a snapshot. Stock
        // llama.cpp normally delays this copy until the following decode.
        int offset=r->rs_idx[0]*r->size+tail;
        for(int i=0;i<info.layers;++i)if(c4[i])for(auto t:{r->r_l[i],r->s_l[i]})
            lp_copy((char*)t->data+tail*t->nb[1],(char*)t->data+offset*t->nb[1],t->nb[1]);
        r->set_rs_idx(0,0);r->cells[tail].src=tail;lp_device_sync();
    }
};
port_context & get(void * p){require(p,"null native context");return *(port_context*)p;}

// No implicit BOS, no hidden prompt, no retokenization. Positions are absolute.
void decode(llama_context * ctx,const int32_t * ids,int n,int pos,int hidden,
            const float * hs=nullptr,bool all=false){
    auto b=llama_batch_init(n,0,1);
    b.n_tokens=n;
    for(int i=0;i<n;++i){b.token[i]=ids[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=all||i==n-1;}
    // MTP takes both token IDs and shifted hidden representations.
    b.embd=const_cast<float*>(hs);
    int rc=llama_decode(ctx,b);
    b.embd=nullptr;llama_batch_free(b);
    require(rc==0,"llama_decode failed (context capacity, CUDA allocation or graph execution)");
    llama_synchronize(ctx);
}
int choose(const float * values,int n){return (int)(std::max_element(values,values+n)-values);}
bool eos_token(int t,const int32_t * eos,int n){return std::find(eos,eos+n,t)!=eos+n;}
uint64_t field_bytes(port_context & p,int kind,int clock){
    switch(kind){case 0:return p.info.hidden*2ULL;case 1:case 2:return (uint64_t)p.info.kv_heads*clock*p.info.head_dim*2;
    case 3:return p.info.conv_channels*4ULL*2;case 4:return p.info.recurrent_heads*128ULL*128*4;default:throw std::runtime_error("unknown field kind");}
}
void validate(port_context & p,const lp_field * f,int count,int clock){
    require(count==p.info.layers*2+1 && clock>0 && clock<=p.info.context_size,"incomplete state or invalid clock");
    std::unordered_set<int> seen;
    for(int j=0;j<count;++j){auto & x=f[j];require(x.data,"null state data");
        require(x.kind>=0&&x.kind<=4,"unknown component");
        if(x.kind==0)require(x.layer==-1,"H must have no layer");
        else{
            require(x.layer>=0&&x.layer<p.info.layers,"layer outside model");
            require(p.model->hparams.is_recr(x.layer)==(x.kind==3||x.kind==4),"component kind differs from layer type");
        }
        uint64_t expected=field_bytes(p,x.kind,clock);
        require(x.bytes==expected||((x.kind==0||x.kind==3)&&x.bytes==expected*2),"wrong field byte length");
        require(seen.insert((x.layer+1)*5+x.kind).second,"duplicate field");
    }
}
void reset(port_context & p){
    llama_memory_clear(llama_get_memory(p.ctx),true);
    if(p.draft)llama_memory_clear(llama_get_memory(p.draft),true);
    ggml_backend_buffer_clear(p.aux_buf,0);p.clock=0;p.pending=false;p.ready=false;p.poisoned=false;p.forced_drafts.clear();
}
void commit(port_context & p,int token){
    p.health();require(token>=0&&token<p.info.vocabulary&&p.clock<p.info.context_size,"invalid committed token or full context");
    p.save_before();decode(p.ctx,&token,1,p.clock,p.info.hidden);p.finish(1,1,0);++p.trace.committed_tokens;
}
}

void lp_graph_tap(void * owner,ggml_context * ctx,ggml_cgraph * graph,ggml_tensor * value,int kind,int layer){
    {std::lock_guard<std::mutex> lock(registry_mutex);if(!registry.count(owner))return;}
    auto & p=get(owner);
    if(kind==0){
        require(value->ne[0]==p.info.hidden&&value->ne[1]<=p.batch,"H graph shape changed");
        auto dst=ggml_view_2d(ctx,p.h_tap,p.info.hidden,value->ne[1],p.h_tap->nb[1],0);
        ggml_build_forward_expand(graph,ggml_cpy(ctx,value,dst));
    }else{
        require(value->ne[0]==p.info.conv_channels&&value->ne[2]==1,"QKV graph shape changed");
        const int n=(int)std::min<int64_t>(value->ne[1],p.tap_capacity);
        auto src=ggml_view_2d(ctx,value,p.info.conv_channels,n,value->nb[1],(value->ne[1]-n)*value->nb[1]);
        auto dst=ggml_view_2d(ctx,p.raw.at(layer),p.info.conv_channels,n,p.raw[layer]->nb[1],0);
        ggml_build_forward_expand(graph,ggml_cpy(ctx,src,dst));
    }
}

#define LP_TRY try { error.clear();
#define LP_CATCH } catch(const std::exception & e){error=e.what();return -1;}
extern "C" {
int lp_abi(){return 2;}
const char * lp_error(){return error.c_str();}
void * lp_open(const char * path,int side,int context,int batch,int threads,int drafts){
    try{
        error.clear();require(side==0||side==1,"invalid model side");
        require(context>=32&&batch>=16&&batch<=1024&&drafts>=0&&drafts<=8,"invalid native options");
        static std::once_flag once;std::call_once(once,[]{llama_backend_init();});
        auto p=std::make_unique<port_context>();
        auto mp=llama_model_default_params();mp.n_gpu_layers=999;mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.main_gpu=0;mp.load_mtp=drafts>0;
        p->gpu_weights={{{".*",ggml_backend_cuda_buffer_type(0)},{nullptr,nullptr}}};
        mp.tensor_buft_overrides=p->gpu_weights.data();
        p->model=llama_model_load_from_file(path,mp);require(p->model,"could not load GGUF model");
        for(const auto & entry:p->model->tensors_by_name)
            require(entry.second->buffer&&!ggml_backend_buffer_is_host(entry.second->buffer),"GGUF weight is not GPU resident");
        const auto & h=p->model->hparams;
        require(p->model->arch==LLM_ARCH_QWEN35,"GGUF is not the supported dense Qwen hybrid architecture");
        require(h.n_layer()==(side?64:32)&&h.n_embd==(side?5120:2560)&&h.ssm_d_conv==4&&h.ssm_d_state==128,
                "GGUF geometry differs from selected bridge");
        require(h.n_head_kv()==4&&h.n_embd_head_k()==256&&h.ssm_dt_rank==(side?48:32)&&h.ssm_n_group==16,"GGUF head geometry mismatch");
        for(int i=0;i<(int)h.n_layer();++i)require(h.is_recr(i)==((i+1)%4!=0),"GGUF layer pattern differs");
        if(drafts)require(h.n_layer_nextn>0,"MTP requested but GGUF contains no trained MTP weights");
        p->info={2,(int)h.n_layer(),(int)h.n_embd,4,256,(int)(h.ssm_d_inner+2*h.ssm_n_group*h.ssm_d_state),
                 (int)h.ssm_dt_rank,(int)h.n_layer_nextn,drafts,context,llama_vocab_n_tokens(llama_model_get_vocab(p->model)),0};
        p->batch=batch;p->tap_capacity=std::max(4,drafts+1);p->logits.resize(p->info.vocabulary);p->pending_h.resize(h.n_embd);
        p->init_aux();
        {std::lock_guard<std::mutex> lock(registry_mutex);registry.insert(p.get());}
        auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_batch=batch;cp.n_ubatch=batch;cp.n_seq_max=1;
        cp.n_rs_seq=drafts;cp.n_threads=threads;cp.n_threads_batch=threads;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.type_k=GGML_TYPE_BF16;cp.type_v=GGML_TYPE_BF16;
        cp.offload_kqv=true;cp.op_offload=true;cp.cb_eval_user_data=p.get();
        p->ctx=llama_init_from_model(p->model,cp);require(p->ctx,"could not initialize full-GPU target context");
        p->mem=dynamic_cast<llama_memory_hybrid*>(llama_get_memory(p->ctx));require(p->mem,"hybrid cache missing");
        llama_set_embeddings_nextn(p->ctx,true,true);
        for(int i=0;i<p->info.layers;++i){
            auto tensor=h.is_recr(i)?p->rs()->s_l[i]:latentport_access::tensor(p->kv(),i,false);
            require(tensor && !ggml_backend_buffer_is_host(tensor->buffer),"cache is offloaded to host; full GPU execution required");
        }
        if(drafts){
            cp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;cp.n_rs_seq=0;cp.cb_eval_user_data=nullptr;
            p->draft=llama_init_from_model(p->model,cp);require(p->draft,"MTP context initialization failed");
            llama_set_embeddings_nextn(p->draft,true,true);
        }
        p->init_head();return p.release();
    }catch(const std::exception & e){error=e.what();return nullptr;}
}
void lp_close(void * ptr){delete (port_context*)ptr;}
int lp_get_info(void * ptr,lp_info * out){LP_TRY require(out,"null info");*out=get(ptr).info;return 0;LP_CATCH}
int lp_tokenize(void * ptr,const char * text,int bytes,int32_t * out,int capacity){LP_TRY
    auto & p=get(ptr);return llama_tokenize(llama_model_get_vocab(p.model),text,bytes,out,capacity,false,true);LP_CATCH}
int lp_reset(void * ptr){LP_TRY reset(get(ptr));return 0;LP_CATCH}
int lp_install(void * ptr,const lp_field * fields,int count,int clock){LP_TRY
    double started=now();
    auto & p=get(ptr);validate(p,fields,count,clock);reset(p);
    latentport_access::install_clock(p.kv(),clock);
    auto r=p.rs();r->cells[0].pos=clock-1;r->cells[0].src=0;r->cells[0].tail=0;r->cells[0].seq_id.insert(0);r->head=0;r->used=1;
    for(int j=0;j<count;++j){const auto & f=fields[j];int i=f.layer;
        if(f.kind==0){if(f.bytes==p.info.hidden*4ULL)lp_copy(p.terminal->data,f.data,f.bytes);else lp_bf16_to_float((float*)p.terminal->data,f.data,p.info.hidden);}
        if(f.kind==1||f.kind==2){auto t=latentport_access::tensor(p.kv(),i,f.kind==2);require(t->type==GGML_TYPE_BF16,"cache type changed");
            lp_convert_kv(t->data,f.data,clock,p.info.kv_heads,p.info.head_dim,true);}
        if(f.kind==3)lp_conv_import((float*)p.c4[i]->data,(float*)r->r_l[i]->data,f.data,p.info.conv_channels,f.bytes==p.info.conv_channels*16ULL);
        if(f.kind==4)lp_convert_r(r->s_l[i]->data,f.data,p.info.recurrent_heads,128,true);
    }
    lp_device_sync();p.clock=clock;p.trace.last_install_s=now()-started;++p.trace.installs;p.compute_head();return 0;LP_CATCH}
int lp_export(void * ptr,const lp_field * fields,int count){LP_TRY
    auto & p=get(ptr);p.health();validate(p,fields,count,p.clock);llama_synchronize(p.ctx);
    auto r=p.rs();int tail=r->cells[0].tail;require(tail>=0&&r->rs_idx[0]==0,"uncommitted recurrent state cannot be exported");
    for(int j=0;j<count;++j){const auto & f=fields[j];int i=f.layer;
        if(f.kind==0){if(f.bytes==p.info.hidden*4ULL)lp_copy(f.data,p.terminal->data,f.bytes);else lp_float_to_bf16(f.data,(float*)p.terminal->data,p.info.hidden);}
        if(f.kind==1||f.kind==2){auto t=latentport_access::tensor(p.kv(),i,f.kind==2);lp_convert_kv(f.data,t->data,p.clock,p.info.kv_heads,p.info.head_dim,false);}
        if(f.kind==3)lp_conv_export(f.data,(float*)p.c4[i]->data,p.info.conv_channels,f.bytes==p.info.conv_channels*16ULL);
        if(f.kind==4)lp_convert_r(f.data,(char*)r->s_l[i]->data+tail*r->s_l[i]->nb[1],p.info.recurrent_heads,128,false);
    }
    lp_device_sync();++p.trace.exports;return 0;LP_CATCH}
int lp_prefill(void * ptr,const int32_t * ids,int count){LP_TRY
    auto & p=get(ptr);p.health();require(count>0&&p.clock+count<=p.info.context_size,"prefill exceeds context");
    for(int j=0;j<count;++j)require(ids[j]>=0&&ids[j]<p.info.vocabulary,"token outside vocabulary");
    for(int start=0;start<count;start+=p.batch){int n=std::min(p.batch,count-start);p.save_before();decode(p.ctx,ids+start,n,p.clock,p.info.hidden);p.finish(n,n,0);}
    // Appending outside a decode transaction invalidates speculative suffix KV.
    if(p.draft)llama_memory_clear(llama_get_memory(p.draft),true);
    p.pending=false;p.trace.prefill_tokens+=count;return 0;LP_CATCH}
int lp_logits(void * ptr,float * out,int capacity){LP_TRY
    auto & p=get(ptr);p.health();require(p.ready&&capacity>=p.info.vocabulary,"no endpoint logits or small output");
    std::copy(p.logits.begin(),p.logits.end(),out);return 0;LP_CATCH}
int lp_head(void * ptr,const void * h,uint64_t bytes){LP_TRY
    auto & p=get(ptr);require(h&&(bytes==p.info.hidden*2ULL||bytes==p.info.hidden*4ULL),"invalid terminal H");
    if(bytes==p.info.hidden*4ULL)lp_copy(p.terminal->data,h,bytes);else lp_bf16_to_float((float*)p.terminal->data,h,p.info.hidden);
    lp_device_sync();p.compute_head();return 0;LP_CATCH}
int lp_commit(void * ptr,int32_t token){LP_TRY commit(get(ptr),token);return 0;LP_CATCH}
int lp_greedy(void * ptr,int limit,const int32_t * eos,int neos,int32_t * output,lp_receipt * result){LP_TRY
    auto & p=get(ptr);p.health();require(p.ready&&limit>0&&output&&result&&neos>0,"invalid generation call");
    int remaining=std::min(limit,p.info.context_size-p.clock);require(remaining>0,"context full");
    int first=choose(p.logits.data(),p.info.vocabulary);
    *result={0,0,0,0,p.clock,1};
    if(!p.draft||!p.pending||remaining==1||eos_token(first,eos,neos)){
        bool warm=p.draft&&!p.pending;commit(p,first);output[0]=first;*result={1,0,0,(int)warm,p.clock,1};return 0;
    }
    int start=p.clock,heads=p.info.mtp_heads;
    int count=std::min(p.info.max_drafts,remaining-1);if(heads>1)count=std::min(count,heads);
    std::vector<int32_t> ids{first};std::vector<float> chain=p.pending_h;
    auto dmem=llama_get_memory(p.draft);
    for(int step=0;step<count;++step){
        if(heads>1){
            require(llama_memory_seq_rm(dmem,0,start,-1),"draft trim failed");llama_set_nextn_layer_offset(p.draft,step);
            decode(p.draft,ids.data(),(int)ids.size(),start,p.info.hidden,chain.data(),true);
        }else{
            decode(p.draft,&ids.back(),1,start+step,p.info.hidden,chain.data()+(size_t)step*p.info.hidden,true);
        }
        int token=choose(llama_get_logits_ith(p.draft,-1),p.info.vocabulary);
        if(step<(int)p.forced_drafts.size())token=p.forced_drafts[step];
        const float * hidden=llama_get_embeddings_nextn_ith(p.draft,-1);
        require(hidden,"draft hidden output missing");chain.insert(chain.end(),hidden,hidden+p.info.hidden);ids.push_back(token);
        if(eos_token(token,eos,neos))break;
    }
    auto old_h=p.pending_h;
    p.save_before();decode(p.ctx,ids.data(),(int)ids.size(),start,p.info.hidden,nullptr,true);
    p.trace.verified_tokens+=ids.size();
    p.forced_drafts.clear();
    std::vector<int32_t> predictions(ids.size()-1);
    for(int i=0;i<(int)predictions.size();++i)predictions[i]=choose(llama_get_logits_ith(p.ctx,i),p.info.vocabulary);
    int accepted=lp_verified_prefix(ids.data(),predictions.data(),(int)ids.size(),eos,neos);
    // Correct each draft head's suffix with the actually committed target
    // hidden representations. No target processing of historical prompt.
    std::vector<float> shifted((size_t)accepted*p.info.hidden);
    std::copy(old_h.begin(),old_h.end(),shifted.begin());
    for(int i=1;i<accepted;++i){auto h=llama_get_embeddings_nextn_ith(p.ctx,i-1);std::copy(h,h+p.info.hidden,shifted.begin()+(size_t)i*p.info.hidden);}
    p.rollback(accepted,(int)ids.size());p.finish((int)ids.size(),accepted,accepted-1,true);
    p.trace.committed_tokens+=accepted;
    for(int head=0;head<heads;++head){
        require(llama_memory_seq_rm(dmem,0,start,-1),"draft correction rollback failed");
        llama_set_nextn_layer_offset(p.draft,head);decode(p.draft,ids.data(),accepted,start,p.info.hidden,shifted.data(),true);
    }
    llama_set_nextn_layer_offset(p.draft,0);
    std::copy(ids.begin(),ids.begin()+accepted,output);
    *result={accepted,(int)ids.size()-1,accepted-1,0,p.clock,1};return 0;
}catch(const std::exception & e){if(ptr)((port_context*)ptr)->poisoned=true;error=e.what();return -1;}}
int lp_clock(void * ptr){LP_TRY return get(ptr).clock;LP_CATCH}
int lp_sync(void * ptr){LP_TRY auto & p=get(ptr);llama_synchronize(p.ctx);lp_device_sync();return 0;LP_CATCH}
int lp_get_trace(void * ptr,lp_trace * out){LP_TRY
    auto & p=get(ptr);require(out,"missing trace output");*out=p.trace;
    for(const auto & entry:llama_get_memory_breakdown(p.ctx)){
        out->model_bytes+=entry.second.model;out->context_bytes+=entry.second.context;out->compute_bytes+=entry.second.compute;
    }
    if(p.draft)for(const auto & entry:llama_get_memory_breakdown(p.draft)){
        // Draft context shares the target model's weights.
        out->context_bytes+=entry.second.context;out->compute_bytes+=entry.second.compute;
    }
    out->auxiliary_bytes=p.info.auxiliary_bytes;
    for(int i=0;i<ggml_backend_sched_get_n_backends(p.head_sched);++i)
        out->compute_bytes+=ggml_backend_sched_get_buffer_size(p.head_sched,ggml_backend_sched_get_backend(p.head_sched,i));
    return 0;LP_CATCH}
int lp_test_force_drafts(void * ptr,const int32_t * ids,int count){LP_TRY
    auto & p=get(ptr);require(count>=0&&count<=p.info.max_drafts,"invalid forced proposal count");
    for(int i=0;i<count;++i)require(ids[i]>=0&&ids[i]<p.info.vocabulary,"forced token outside vocabulary");
    p.forced_drafts.assign(ids,ids+count);return 0;LP_CATCH}
}
