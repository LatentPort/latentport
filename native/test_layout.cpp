#include "layout.h"
#include "speculation.h"
#include <vector>
#include <iostream>
#include <stdexcept>
static void require(bool value){if(!value)throw std::runtime_error("layout assertion failed");}
int main(){
    // Asymmetric marker tensors detect accidental H/T or K/V transposition.
    const int h=4,t=7,d=5;std::vector<int> native(h*t*d),back(h*t*d);
    for(int i=0;i<h*t*d;++i)native[lp_kv_index(i,t,h,d)]=i;
    for(int a=0;a<h;++a)for(int b=0;b<t;++b)for(int c=0;c<d;++c)
        require(native[(b*h+a)*d+c]==(a*t+b)*d+c);
    for(int i=0;i<h*d*d;++i)require(lp_r_index(lp_r_index(i,d),d)==i);
    require(lp_r_index(1,5)==5);
    for(int ratio:{2,3}){
        int keys=16,heads=keys*ratio,dim=7;
        std::vector<int> r(heads*dim*dim,-1);
        for(int i=0;i<(int)r.size();++i)r[lp_recurrent_index(i,keys,heads,dim)]=i;
        for(int group=0;group<keys;++group)for(int member=0;member<ratio;++member)
            for(int k=0;k<dim;++k)for(int v=0;v<dim;++v)
                require(r[((member*keys+group)*dim+v)*dim+k]==((group*ratio+member)*dim+k)*dim+v);
        int base=2*keys*dim;
        require(lp_c_channel(base+dim,keys,heads,dim)==base+keys*dim);
        require(lp_c_channel(base-1,keys,heads,dim)==base-1);
    }
    // Rejected projected tokens must not enter the published four-slot state.
    for(int accepted=1;accepted<=9;++accepted)for(int lane=0;lane<4;++lane)
        require(lp_conv_time(accepted,lane)==accepted+lane);
    int32_t ids[]={10,20,30,40},targets[]={20,30,40},stop[]={99};
    require(lp_verified_prefix(ids,targets,4,stop,1)==4);
    targets[0]=21;require(lp_verified_prefix(ids,targets,4,stop,1)==1);
    targets[0]=20;targets[1]=31;require(lp_verified_prefix(ids,targets,4,stop,1)==2);
    targets[1]=30;stop[0]=30;require(lp_verified_prefix(ids,targets,4,stop,1)==3);
    std::cout<<"KV/head permutation, recurrent transpose, convolution and speculative boundaries passed\n";
}
