// Fixed full-alphabet prefix code for escape masks. No prompts, activations or tokens.
// Trained on the public pinned-weight sample; all 256 masks are representable.
#pragma once
namespace tf_bp54 {
__device__ __constant__ uint32_t encode_table[256] = {
196609u,327701u,327709u,458879u,327685u,458791u,458799u,524476u,327693u,393264u,393220u,524426u,393272u,524434u,524538u,655982u,327707u,458863u,458871u,524386u,458767u,524514u,524412u,656286u,393224u,524380u,524460u,655798u,524370u,655606u,655886u,722596u,327683u,458759u,458855u,524482u,458815u,524474u,524390u,656238u,458847u,524458u,524316u,656334u,524518u,655630u,656222u,722532u,458831u,524442u,524306u,655422u,524290u,655934u,655454u,786942u,524322u,655406u,656062u,721764u,655998u,721828u,788734u,858942u,327699u,458807u,393256u,524374u,458783u,524346u,524348u,656366u,458775u,524422u,524314u,655678u,524444u,655966u,655486u,722788u,393216u,524378u,524330u,656182u,524490u,656246u,656054u,721380u,524294u,656374u,656094u,721124u,656302u,721636u,790398u,859190u,458823u,524498u,524508u,655598u,524454u,655470u,655670u,790014u,524524u,656118u,655542u,787454u,655582u,786814u,788478u,853366u,524486u,656078u,655710u,721508u,656206u,787966u,722212u,857462u,655990u,787198u,721956u,856438u,722340u,856894u,853822u,990974u,327691u,393236u,393268u,524466u,393240u,524332u,524450u,656350u,393232u,524506u,524418u,655862u,524402u,655502u,655838u,787326u,458839u,524362u,524394u,656174u,524522u,655518u,656014u,788862u,524540u,655646u,655854u,789758u,655902u,788350u,722724u,854582u,393248u,524410u,524298u,656142u,524354u,656310u,655726u,721444u,524310u,656318u,655662u,721892u,655694u,722660u,721252u,853558u,524438u,656046u,656158u,721060u,655822u,720950u,789886u,852798u,655478u,722148u,788990u,858486u,786686u,856630u,857142u,1007358u,393228u,524338u,524326u,655374u,524530u,655950u,655390u,790526u,524502u,655550u,655534u,720932u,655806u,787710u,722276u,857918u,524358u,656030u,656270u,788222u,655566u,787838u,789246u,855414u,655758u,722404u,789502u,853046u,722468u,859702u,858678u,920630u,524396u,655438u,655790u,722084u,655774u,721316u,722852u,852342u,656110u,789374u,721700u,857654u,721188u,852534u,933630u,928822u,655918u,722020u,722916u,859966u,720996u,855606u,859510u,1003262u,721572u,854846u,854390u,995070u,855870u,1109758u,986878u,1076990u
};
}
extern "C" __global__ void tf_bp54_init(uint16_t* decode,uint8_t* mapping) {
 const unsigned t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t<16) mapping[t]=t<8?t+5:(t<13?t-8:t);
 if(t<65536) {
  for(unsigned symbol=0;symbol<256;++symbol) {
   unsigned e=tf_bp54::encode_table[symbol],n=e>>16;
   if((t&((1u<<n)-1))==(e&65535)){decode[t]=symbol|(n<<8);break;}
  }
 }
}
// First pass returns aligned byte size; second pass writes exactly those bytes.
extern "C" __global__ void tf_bp54_pack(const uint32_t* words,uint32_t* sizes,const uint32_t* offsets,
 uint8_t* blob,int panels) {
 const unsigned lane=threadIdx.x&31,panel=blockIdx.x;
 if(panel>=unsigned(panels))return;
 unsigned masks[16],low3[16],length=0;
#pragma unroll
 for(unsigned j=0;j<16;++j) {
  unsigned w=words[panel*512+lane*16+j],mask=0,low=0;
#pragma unroll
  for(unsigned k=0;k<8;++k) {
   const unsigned q=(w>>(k*4))&15;
   const unsigned code=q<5?q+8:q<13?q-5:q;
   mask|=(code>>3)<<k;low|=(code&7)<<(3*k);
  }
  masks[j]=mask;low3[j]=low;length+=tf_bp54::encode_table[mask]>>16;
 }
 length=(length+7)/8;
 unsigned scan=length;
#pragma unroll
 for(unsigned d=1;d<32;d<<=1) {unsigned v=__shfl_up_sync(0xffffffffu,scan,d);if(lane>=d)scan+=v;}
 const unsigned total=__shfl_sync(0xffffffffu,scan,31);
 if(!blob){if(lane==0)sizes[panel]=(1556+total+3)&~3u;return;}
 uint8_t* p=blob+offsets[panel];
 // Race-free directory bytes: each lane writes at most one byte, gathering
 // at most three adjacent five-bit length fields before any divergence.
 const unsigned start=(lane*8)/5,shift=(lane*8)%5;
 unsigned f0=__shfl_sync(0xffffffffu,length-1,start&31);
 unsigned f1=__shfl_sync(0xffffffffu,length-1,(start+1)&31);
 unsigned f2=__shfl_sync(0xffffffffu,length-1,(start+2)&31);
 if(lane<20)p[1536+lane]=uint8_t((f0|(f1<<5)|(f2<<10))>>shift);
 uint8_t* stream=p+1556+scan-length;
 unsigned buffer=0,bits=0,cursor=0;
#pragma unroll
 for(unsigned j=0;j<16;++j) {
  const unsigned word=lane*16+j,low=low3[j];
  p[word*3]=uint8_t(low);p[word*3+1]=uint8_t(low>>8);p[word*3+2]=uint8_t(low>>16);
  const unsigned entry=tf_bp54::encode_table[masks[j]],n=entry>>16;
  buffer|=(entry&65535)<<bits;bits+=n;
  while(bits>=8){stream[cursor++]=uint8_t(buffer);buffer>>=8;bits-=8;}
 }
 if(bits)stream[cursor]=uint8_t(buffer);
}
