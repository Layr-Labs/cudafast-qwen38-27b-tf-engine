// Full-model-sized recurrence storage separates memory instructions from arithmetic and cache-capacity artifacts.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "qwen35.cuh"
using namespace tf_q35;
__device__ __forceinline__ bool past_windows(const int* p,int s){return p&&s>=p[3];}
__device__ __forceinline__ bool recorded(const int* p){return p&&p[4]!=0;}
__device__ __forceinline__ void release_next(){
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void await_prior(){
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
#include "qwen35_slots.cuh"
#include "width_siblings.cuh"
#include "width_siblings_nested.cuh"
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line%d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>T* upload(const std::vector<T>& v){auto p=alloc<T>(v.size());CU(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<int S,int MODE>
__global__ void __launch_bounds__(128) scan(Geo geo,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
 const float* g,const __nv_bfloat16* beta,const Seg* segs,const SlotStep* steps,int layer,__nv_bfloat16* y,const int* p){
    if constexpr(MODE==0) q35_slots<S,2>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
    else if constexpr(MODE==1) q35_siblings<S,2>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
    else if constexpr(MODE==2) {
        if constexpr(S==3) q35_slots<3,4,true>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
        else q35_slots<S,2>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
    } else if constexpr(MODE==3) q35_siblings<S,1>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
    else if constexpr(MODE==4) {
        if constexpr(S==3) q35_slots_nested<3,4,true>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
        else q35_slots<S,2>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
    } else q35_siblings<S,2,2,false>(geo,q,k,v,g,beta,segs,steps,layer,y,p);
}
struct Inputs{
    Geo geo{5120,16,48,24,4,48};
    __nv_bfloat16 *q,*k,*v,*beta,*y;
    float *g,*state,*initial,*before;
    unsigned char *log,*previous;Seg* segs;SlotStep* steps;int* params;
    size_t state_count,log_bytes,y_count,key_stride,value_stride,gate_stride;
};
struct Trial{const char* name;cudaGraphExec_t graph;cudaEvent_t begin,end;};
template<int S,int MODE>void add(std::vector<Trial>& trials,Inputs& a,cudaStream_t stream,const char* name){
    Trial t{name};CU(cudaEventCreate(&t.begin));CU(cudaEventCreate(&t.end));cudaGraph_t graph;
    CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));CU(cudaEventRecordWithFlags(t.begin,stream,cudaEventRecordExternal));
    for(int layer=0;layer<a.geo.nl;++layer)scan<S,MODE><<<dim3(1,MODE==3?32:((MODE==2 || MODE==4) && S==3?8:16),48),dim3(32,4),0,stream>>>(a.geo,a.q+layer*a.key_stride,a.k+layer*a.key_stride,a.v+layer*a.value_stride,a.g+layer*a.gate_stride,a.beta+layer*a.gate_stride,a.segs,a.steps,layer,a.y+layer*a.value_stride,a.params);
    CU(cudaEventRecordWithFlags(t.end,stream,cudaEventRecordExternal));CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&t.graph,graph,0));CU(cudaGraphDestroy(graph));trials.push_back(t);
}
template<int S>void measure(int rows,int pn,bool recording,int layers,int pattern){
    Inputs a{};a.geo.nl=layers;const size_t recurrent=size_t(48)*128*128;
    a.state_count=recurrent*layers;a.key_stride=size_t(rows)*16*128;a.value_stride=size_t(rows)*48*128;a.gate_stride=size_t(rows)*48;
    const size_t log_stride=48*4+48*128*4+16*128*2+3*(2*16+48)*128*2;
    a.log_bytes=size_t(rows)*layers*log_stride;a.y_count=a.value_stride*layers;
    unsigned seed=337;auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return(int(seed&65535)-32768)/32768.f;};
    auto bf_values=[&](size_t n,float scale){std::vector<__nv_bfloat16> v(n);for(auto& x:v)x=__float2bfloat16_rn(random()*scale);return v;};
    a.q=upload(bf_values(a.key_stride*layers,.08f));a.k=upload(bf_values(a.key_stride*layers,.08f));a.v=upload(bf_values(a.y_count,.5f));
    a.beta=upload(std::vector<__nv_bfloat16>(a.gate_stride*layers,__float2bfloat16_rn(.25f)));a.g=upload(std::vector<float>(a.gate_stride*layers,.96f));
    std::vector<float> initial(a.state_count);for(auto& x:initial)x=random()*.01f;a.initial=upload(initial);a.state=alloc<float>(a.state_count);a.before=alloc<float>(a.state_count);
    a.log=alloc<unsigned char>(a.log_bytes);a.y=alloc<__nv_bfloat16>(a.y_count);
    std::vector<unsigned char> previous(size_t(2)*layers*log_stride,0);
    for(int row=0;row<2;++row)for(int layer=0;layer<layers;++layer){auto g=reinterpret_cast<float*>(previous.data()+(size_t(row)*layers+layer)*log_stride);auto delta=g+48;auto key=reinterpret_cast<__nv_bfloat16*>(delta+48*128);
        for(int i=0;i<48;++i)g[i]=.97f;for(int i=0;i<48*128;++i)delta[i]=random()*.01f;for(int i=0;i<16*128;++i)key[i]=__float2bfloat16_rn(random()*.08f);}
    a.previous=upload(previous);
    Seg seg{};seg.rows=rows;seg.pn=pn;seg.plog=reinterpret_cast<unsigned long long>(a.previous);seg.log=reinterpret_cast<unsigned long long>(a.log);seg.rec=reinterpret_cast<unsigned long long>(a.state);seg.before=reinterpret_cast<unsigned long long>(a.before);
    a.segs=upload(std::vector<Seg>{seg});a.params=upload(std::vector<int>{1,rows,rows,1,recording?1:0});
    std::vector<SlotStep> schedule(rows);
    constexpr int live_slots=S<3?S:3;
    for(int r=0;r<rows;++r){
        if(r<live_slots) schedule[r]={unsigned(r),r-1,r};
        else if(pattern==0) schedule[r]={unsigned(r),live_slots-1,live_slots-1};
        else if(pattern==1) schedule[r]={unsigned(r),0,-1};
        else if(r%8==3) schedule[r]={unsigned(r),live_slots-1,live_slots-1};
        else schedule[r]={unsigned(r),live_slots-1,-1};
    }a.steps=upload(schedule);
    cudaStream_t stream;CU(cudaStreamCreate(&stream));std::vector<Trial> trials;
    add<S,0>(trials,a,stream,"parent");add<S,1>(trials,a,stream,"sibling2");add<S,2>(trials,a,stream,"nativebest");add<S,3>(trials,a,stream,"sibling1");add<S,4>(trials,a,stream,"nestedbest");add<S,5>(trials,a,stream,"hazard2");
    auto reset=[&](){CU(cudaMemcpyAsync(a.state,a.initial,a.state_count*4,cudaMemcpyDeviceToDevice,stream));CU(cudaMemsetAsync(a.before,0x7f,a.state_count*4,stream));CU(cudaMemsetAsync(a.log,0x7f,a.log_bytes,stream));CU(cudaMemsetAsync(a.y,0x7f,a.y_count*2,stream));};
    const size_t bytes=a.state_count*8+a.log_bytes+a.y_count*2;std::vector<unsigned char> want(bytes),got(bytes);
    auto collect=[&](std::vector<unsigned char>& out){size_t at=0;auto get=[&](const void* p,size_t n){CU(cudaMemcpy(out.data()+at,p,n,cudaMemcpyDeviceToHost));at+=n;};get(a.state,a.state_count*4);get(a.before,a.state_count*4);get(a.log,a.log_bytes);get(a.y,a.y_count*2);};
    reset();CU(cudaGraphLaunch(trials[0].graph,stream));CU(cudaStreamSynchronize(stream));collect(want);
    for(int rep=0;rep<15;++rep)for(size_t j=0;j<trials.size();++j){auto& t=trials[(j+rep)%trials.size()];reset();CU(cudaGraphLaunch(t.graph,stream));CU(cudaStreamSynchronize(stream));collect(got);
        if(got!=want){size_t at=0;while(at<got.size()&&got[at]==want[at])++at;fprintf(stderr,"slotIO S%d rows%d pn%d rec%d layers%d %s differs offset%zu\n",S,rows,pn,recording,layers,t.name,at);exit(1);}
        if(rep>=2){printf("{\"pattern\":%d,\"sample\":true}\n",pattern);float ms;CU(cudaEventElapsedTime(&ms,t.begin,t.end));printf("{\"slots\":%d,\"rows\":%d,\"pn\":%d,\"record\":%d,\"layers\":%d,\"case\":\"%s\",\"rep\":%d,\"ms\":%.8f,\"mismatches\":0}\n",S,rows,pn,recording,layers,t.name,rep-2,ms);}}
    fflush(stdout);
    for(auto& t:trials){CU(cudaGraphExecDestroy(t.graph));CU(cudaEventDestroy(t.begin));CU(cudaEventDestroy(t.end));}
    for(void* p:std::vector<void*>{a.q,a.k,a.v,a.beta,a.y,a.g,a.state,a.initial,a.before,a.log,a.previous,a.segs,a.params,a.steps})CU(cudaFree(p));CU(cudaStreamDestroy(stream));
}
int main(){for(int pattern:{0,1,2})for(int pn:{0,2})for(int rows:{16,32}){
 measure<3>(rows,pn,true,48,pattern);measure<4>(rows,pn,true,48,pattern);
}return 0;}
