import numpy as np
import pyopencl as cl
import png
import random
import functools as ft

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


program="""

__kernel void floatFlatToInt(__global uint2 *res_g,__global const  float *inp_g){
      int gid = get_global_id(0);
      float curr= 1000.0*inp_g[gid];
      res_g[gid][0] = curr;
      curr= 1000.0*inp_g[gid+8];
      res_g[gid][1] = curr;
    }
"""










mf = cl.mem_flags
prg = cl.Program(ctx, program).build()

#np python side
res_np = np.zeros(shape=(8,2),dtype=np.uint32)
inp_np =  np.random.rand(16).astype(np.float32)


#g is kernel side
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)
inp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp_np)




conv = prg.floatFlatToInt
conv(queue, inp_np.shape, None, res_g,inp_g)
cl.enqueue_copy(queue, res_np, res_g)

#random points created






#kd_tree=np.zeros(shape=(100,2),dtype=np.uint)
kd_tree=np.keys()
#map(lambda x:np.array([-1,-1]),kd_tree)
#print(kd_tree)
#even=True
#for x in res_np:
#    ind=0
    #if()
    #if(x[0]kd_tree)


















