#pragma once
#include <stdint.h>
// Input[0] was selected from the already committed target endpoint. Each
// following draft must equal the target prediction from the preceding row.
inline int lp_verified_prefix(const int32_t * ids,const int32_t * predictions,int n,
                              const int32_t * eos,int eos_count){
    int keep=1;
    for(int i=1;i<n;++i){
        if(ids[i]!=predictions[i-1])break;
        ++keep;
        bool stop=false;for(int j=0;j<eos_count;++j)stop|=ids[i]==eos[j];
        if(stop)break;
    }
    return keep;
}
