// A larger CTA groups independent original warps; every convolution, normalization and sum keeps its order.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "qwen35.cu"
#include "width_glue.cuh"
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line%d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>T* upload(const std::vector<T>& v){auto p=alloc<T>(v.size());CU(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct Trial{int mode;cudaGraphExec_t graph;cudaEvent_t begin,end;};
void measure(int rows,int pattern){
 const int capacity=32,layers=48;Geo geo{5120,16,48,24,4,48};
 const size_t kc=size_t(capacity)*16*128,vc=size_t(capacity)*48*128,gc=size_t(capacity)*48,cc=size_t(capacity)*80*128;
 const size_t logstride=90304,logbytes=size_t(capacity)*layers*logstride;
 unsigned seed=996;auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return (int(seed&65535)-32768)/32768.f;};
 auto values=[&](size_t n,float scale){std::vector<__nv_bfloat16> v(n);for(auto& x:v)x=__float2bfloat16_rn(random()*scale);return v;};
 auto qkv=upload(values(cc*layers,.1f)),cw=upload(values(size_t(80)*128*4*layers,.2f));
 auto conv=upload(values(size_t(3)*80*128*layers,.1f)),ain=upload(values(gc*layers,.5f)),bin=upload(values(gc*layers,.5f));
 auto dt=upload(values(48*layers,.1f)),nw=upload(values(128*layers,.2f)),y=upload(values(vc*layers,.1f)),z=upload(values(vc*layers,.2f));
 std::vector<float> halog(48*layers);for(auto& x:halog)x=random()*.1f;auto alog=upload(halog);
 auto q=alloc<__nv_bfloat16>(kc*layers),k=alloc<__nv_bfloat16>(kc*layers),v=alloc<__nv_bfloat16>(vc*layers),beta=alloc<__nv_bfloat16>(gc*layers),mix=alloc<__nv_bfloat16>(vc*layers);
 auto g=alloc<float>(gc*layers),xs=alloc<float>(vc/64*layers);auto log=alloc<unsigned char>(logbytes);
 Seg seg{};seg.conv=reinterpret_cast<unsigned long long>(conv);seg.log=reinterpret_cast<unsigned long long>(log);seg.rows=rows;auto segs=upload(std::vector<Seg>{seg});
 std::vector<int> hrs(capacity,0x7fffffff),hpar(capacity,-1);std::fill(hrs.begin(),hrs.begin()+rows,0);
 for(int r=1;r<rows;++r)hpar[r]=pattern==0?r-1:(pattern==1?0:(r-1)/2);
 auto rs=upload(hrs),par=upload(hpar),p=upload(std::vector<int>{1,rows,rows,1,1});
 cudaStream_t stream;CU(cudaStreamCreate(&stream));std::vector<Trial> trials;
#define PRE(fn,grid,block) fn<<<grid,block,0,stream>>>(geo,qkv+layer*cc,segs,rs,par,layer,cw+layer*80*128*4,ain+layer*gc,bin+layer*gc,alog+layer*48,dt+layer*48,q+layer*kc,k+layer*kc,v+layer*vc,g+layer*gc,beta+layer*gc,p)
#define POST(fn,grid,block) fn<<<grid,block,0,stream>>>(geo,y+layer*vc,z+layer*vc,nw+layer*128,1e-6f,mix+layer*vc,p,xs+layer*vc/64)
 for(int mode=0;mode<9;++mode){Trial t{mode};CU(cudaEventCreate(&t.begin));CU(cudaEventCreate(&t.end));cudaGraph_t graph;
  CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));CU(cudaEventRecordWithFlags(t.begin,stream,cudaEventRecordExternal));
  for(int layer=0;layer<layers;++layer){
   if(mode==1||mode==3)PRE(glue_pre_4r,dim3(1,80,capacity/4),dim3(32,4));
   else if(mode==4||mode==5)PRE(glue_pre_8r,dim3(1,80,capacity/8),dim3(32,8));
   else if(mode==7)PRE(glue_pre_4h,dim3(1,80/4,capacity),dim3(32,4));
   else if(mode==8)PRE(glue_pre_8h,dim3(1,80/8,capacity),dim3(32,8));
   else PRE(tf_q35_gdn_pre,dim3(1,80,capacity),dim3(32));
   if(mode==2||mode==3)POST(glue_post_4r,dim3(1,48,capacity/4),dim3(32,4));
   else if(mode==4||mode==6)POST(glue_post_8r,dim3(1,48,capacity/8),dim3(32,8));
   else if(mode==7)POST(glue_post_4h,dim3(1,48/4,capacity),dim3(32,4));
   else if(mode==8)POST(glue_post_8h,dim3(1,48/8,capacity),dim3(32,8));
   else POST(tf_q35_gdn_post,dim3(1,48,capacity),dim3(32));
  }
  CU(cudaEventRecordWithFlags(t.end,stream,cudaEventRecordExternal));CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&t.graph,graph,0));CU(cudaGraphDestroy(graph));trials.push_back(t);
 }
 struct Output{void* ptr;size_t bytes;};std::vector<Output> outputs{{q,kc*layers*2},{k,kc*layers*2},{v,vc*layers*2},{g,gc*layers*4},{beta,gc*layers*2},{mix,vc*layers*2},{xs,vc/64*layers*4},{log,logbytes}};
 size_t bytes=0;for(auto a:outputs)bytes+=a.bytes;std::vector<unsigned char>want(bytes),got(bytes);
 auto reset=[&](){for(auto a:outputs)CU(cudaMemsetAsync(a.ptr,0x7f,a.bytes,stream));};
 auto collect=[&](std::vector<unsigned char>& b){size_t at=0;for(auto a:outputs){CU(cudaMemcpy(b.data()+at,a.ptr,a.bytes,cudaMemcpyDeviceToHost));at+=a.bytes;}};
 reset();CU(cudaGraphLaunch(trials[0].graph,stream));CU(cudaStreamSynchronize(stream));collect(want);
 for(int rep=0;rep<9;++rep)for(int j=0;j<9;++j){auto& t=trials[(rep+j)%9];reset();CU(cudaGraphLaunch(t.graph,stream));CU(cudaStreamSynchronize(stream));collect(got);
  if(got!=want){size_t at=0;while(at<bytes&&got[at]==want[at])++at;fprintf(stderr,"glue differs rows%d pattern%d mode%d byte%zu\n",rows,pattern,t.mode,at);exit(1);}
  if(rep>=2){float ms;CU(cudaEventElapsedTime(&ms,t.begin,t.end));printf("{\"rows\":%d,\"pattern\":%d,\"mode\":%d,\"rep\":%d,\"ms\":%.8f,\"exact\":true}\n",rows,pattern,t.mode,rep-2,ms);}
 }fflush(stdout);
 for(auto t:trials){CU(cudaGraphExecDestroy(t.graph));CU(cudaEventDestroy(t.begin));CU(cudaEventDestroy(t.end));}CU(cudaStreamDestroy(stream));
 for(auto a:outputs)CU(cudaFree(a.ptr));
 for(auto ptr:{(void*)qkv,(void*)cw,(void*)conv,(void*)ain,(void*)bin,(void*)dt,(void*)nw,(void*)y,(void*)z,(void*)alog,(void*)segs,(void*)rs,(void*)par,(void*)p})CU(cudaFree(ptr));
}
int main(){for(int rows:{17,24,32})for(int pattern:{0,1,2})measure(rows,pattern);}
