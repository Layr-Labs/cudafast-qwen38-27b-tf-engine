// Original per-head MMA chains, packed into different query-row blocks.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "qwen35.cu"
#include "width_attention_pruned.cuh"
using namespace tf_q35;
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line%d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>T* upload(const std::vector<T>& v){auto p=alloc<T>(v.size());CU(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<int MT,bool SINGLE>__global__ void __launch_bounds__(64*MT) grouped(Geo geo,const __nv_bfloat16* q,const Seg* segs,const int* rs,const Ancestor* anc,
 const int* p,int total,float* pm,float* pl,float* po){
 extern __shared__ __align__(16) uint4 tiles[];
 for(int c=blockIdx.y;c<MAXCH;c+=gridDim.y){
  if(c!=(int)blockIdx.y)__syncthreads();
  const int rb=16*MT/(geo.qh/geo.kvh),m0=blockIdx.x*rb;
  if(m0>=total || past_rows(p,m0))return;
  const Seg sg=segs[rs[m0]];
  const int keys=sg.len+sg.rows,ck=chunk_keys(sg.len+1),nch=min(MAXCH,(keys+ck-1)/ck);
  if(c>=nch){
   for(int i=threadIdx.x;i<rb*(geo.qh/geo.kvh);i+=blockDim.x){
    const int m=m0+i/(geo.qh/geo.kvh),head=i%(geo.qh/geo.kvh);
    if(m<total && !past_rows(p,m)){const size_t at=(size_t(m)*geo.qh+blockIdx.z*(geo.qh/geo.kvh)+head)*MAXCH+c;pm[at]=-INFINITY;pl[at]=0;}
   }
   continue;
  }
  if constexpr(SINGLE)tf_q35_tc::attn_pruned<MT,tf_q35_tc::Bf16Tiles1<64*MT>,true>({tiles,nullptr,nullptr,nullptr},geo,q,segs,rs,anc,0,p,.0625f,total,pm,pl,po,c);
  else tf_q35_tc::attn_rows<MT,tf_q35_tc::Bf16Tiles<64*MT>,true>({tiles,nullptr,nullptr,nullptr},geo,q,segs,rs,anc,0,p,.0625f,total,pm,pl,po,c);
 }
}
__global__ void __launch_bounds__(128) original(Geo geo,const __nv_bfloat16* q,const Seg* segs,const int* rs,const Ancestor* anc,
 const int* p,float* pm,float* pl,float* po){
 extern __shared__ __align__(16) uint4 tiles[];
 const int m=blockIdx.x;const Seg sg=segs[rs[m]];const int keys=sg.len+rows_seen(anc[m]),ck=chunk_keys(keys),nch=(keys+ck-1)/ck;
 for(int c=blockIdx.y;c<MAXCH;c+=gridDim.y){
  if(c>=nch){if(threadIdx.x<6){const size_t at=(size_t(m)*24+blockIdx.z*6+threadIdx.x)*MAXCH+c;pm[at]=-INFINITY;pl[at]=0;}continue;}
  __syncthreads();tf_q35_tc::attn_one<tf_q35_tc::Bf16Tiles1<128>>({tiles,nullptr,nullptr,nullptr},geo,q,segs,rs,anc,0,p,.0625f,pm,pl,po,c);
 }
}
struct Trial{int mode;cudaGraphExec_t graph;cudaEvent_t begin,end;};
void measure(int rows,int length,int pattern){
 const int capacity=32;
 Geo geo{5120,16,48,24,4,48};unsigned seed=641;
 auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return (int(seed&65535)-32768)/32768.f;};
 auto values=[&](size_t n){std::vector<__nv_bfloat16> v(n);for(auto& x:v)x=__float2bfloat16_rn(random());return v;};
 auto q=upload(values(size_t(rows)*24*HD));const int cap=length+rows+7;
 auto kv=upload(values(size_t(2)*4*cap*HD));Seg seg{};seg.cap=cap;seg.len=length;seg.rows=rows;seg.kv=reinterpret_cast<unsigned long long>(kv);
 auto segs=upload(std::vector<Seg>{seg});std::vector<int> rowsegments(capacity,0x7fffffff);std::fill(rowsegments.begin(),rowsegments.begin()+rows,0);auto rs=upload(rowsegments);auto p=upload(std::vector<int>{1,rows,rows,1,1});
 std::vector<Ancestor> ancestors(rows);for(int i=0;i<rows;++i){
  if(pattern==0)ancestors[i]={0u,unsigned(i),1};
  else if(pattern==1)ancestors[i]={1ull|(1ull<<i),0,0};
  else {unsigned long long bits=1ull<<i;for(int j=i;j>0;j=(j-1)/3)bits|=1ull<<((j-1)/3);ancestors[i]={bits,0,0};}
 }auto anc=upload(ancestors);
 const size_t slots=size_t(capacity)*24*MAXCH;auto pm=alloc<float>(slots),pl=alloc<float>(slots),po=alloc<float>(slots*HD);
 cudaStream_t stream;CU(cudaStreamCreate(&stream));std::vector<Trial> trials;
 for(int mode=0;mode<7;++mode){Trial t{mode};CU(cudaEventCreate(&t.begin));CU(cudaEventCreate(&t.end));cudaGraph_t graph;
  if(mode==4)CU(cudaFuncSetAttribute(grouped<2,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
  if(mode==5)CU(cudaFuncSetAttribute(grouped<4,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
  CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));CU(cudaEventRecordWithFlags(t.begin,stream,cudaEventRecordExternal));
  // Sixteen calls match the model's attention layer count without saturating the cache with result copies.
  for(int layer=0;layer<16;++layer){
   if(mode==0)tf_q35_attn_partial<<<dim3(capacity,8,4),128,32768,stream>>>(geo,q,segs,rs,anc,0,p,.0625f,pm,pl,po);
   if(mode==1)grouped<1,true><<<dim3((capacity+1)/2,8,4),64,32768,stream>>>(geo,q,segs,rs,anc,p,capacity,pm,pl,po);
   if(mode==2)tf_q35_attn_wide<<<dim3((capacity+4)/5,8,4),128,32768,stream>>>(geo,q,segs,rs,anc,0,p,.0625f,capacity,pm,pl,po);
   if(mode==3)grouped<4,true><<<dim3((capacity+9)/10,8,4),256,32768,stream>>>(geo,q,segs,rs,anc,p,capacity,pm,pl,po);
   if(mode==4)grouped<2,false><<<dim3((capacity+4)/5,8,4),128,65536,stream>>>(geo,q,segs,rs,anc,p,capacity,pm,pl,po);
   if(mode==6)grouped<2,true><<<dim3((capacity+4)/5,8,4),128,32768,stream>>>(geo,q,segs,rs,anc,p,capacity,pm,pl,po);
   if(mode==5)grouped<4,false><<<dim3((capacity+9)/10,8,4),256,65536,stream>>>(geo,q,segs,rs,anc,p,capacity,pm,pl,po);
  }
  CU(cudaEventRecordWithFlags(t.end,stream,cudaEventRecordExternal));CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&t.graph,graph,0));CU(cudaGraphDestroy(graph));trials.push_back(t);
 }
 const size_t bytes=slots*(HD+2)*4;std::vector<unsigned char>want(bytes),got(bytes);
 auto reset=[&](){CU(cudaMemsetAsync(pm,0x7f,slots*4,stream));CU(cudaMemsetAsync(pl,0x7f,slots*4,stream));CU(cudaMemsetAsync(po,0x7f,slots*HD*4,stream));};
 auto collect=[&](std::vector<unsigned char>& v){CU(cudaMemcpy(v.data(),pm,slots*4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(v.data()+slots*4,pl,slots*4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(v.data()+slots*8,po,slots*HD*4,cudaMemcpyDeviceToHost));};
 reset();CU(cudaGraphLaunch(trials[0].graph,stream));CU(cudaStreamSynchronize(stream));collect(want);
 for(int rep=0;rep<9;++rep)for(int j=0;j<7;++j){auto& t=trials[(rep+j)%7];reset();CU(cudaGraphLaunch(t.graph,stream));CU(cudaStreamSynchronize(stream));collect(got);
  if(got!=want){size_t at=0;while(at<bytes&&got[at]==want[at])++at;fprintf(stderr,"attention differs rows%d len%d pattern%d mode%d byte%zu want%u got%u\n",rows,length,pattern,t.mode,at,want[at],got[at]);exit(1);}
  if(rep>=2){float ms;CU(cudaEventElapsedTime(&ms,t.begin,t.end));printf("{\"rows\":%d,\"length\":%d,\"pattern\":%d,\"mode\":%d,\"rep\":%d,\"ms\":%.8f,\"exact\":true}\n",rows,length,pattern,t.mode,rep-2,ms);}
 }
 fflush(stdout);for(auto t:trials){CU(cudaGraphExecDestroy(t.graph));CU(cudaEventDestroy(t.begin));CU(cudaEventDestroy(t.end));}CU(cudaStreamDestroy(stream));
 for(auto ptr:{(void*)q,(void*)kv,(void*)segs,(void*)rs,(void*)p,(void*)anc,(void*)pm,(void*)pl,(void*)po})CU(cudaFree(ptr));
}
int main(){for(int rows:{17,24,32})for(int length:{127,496,527,1024})for(int pattern:{0,1,2})measure(rows,length,pattern);}
