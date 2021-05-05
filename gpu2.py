import numpy as np
import pyopencl as cl
import png
from datetime import datetime

size=12
side=2**(size)
total=side**2
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
program="""
__kernel void tsquare(__global char *res_g){
      int gid = get_global_id(0);
      ulong x=gid% side;
      ulong y=gid/ side;
      //s i z e size
      //s i d e side
      
      res_g[gid] = 255*popcount(popcount(popcount(popcount(
      (
            ( ( (x<<2)&~(1<<(size+1)) )  ^ ( (x<<1)&~(1<<size) )  ^  (x&~1) )
      &     ( ( (y<<2)&~(1<<(size+1)) )  ^ ( (y<<1)&~(1<<size) )  ^  (y&~1) )
      )
      ))));//(~(1|(1<<(size-1))))&
  }
"""
program=program.replace("side",str(side))
program=program.replace("size",str(size))
mf = cl.mem_flags
prg = cl.Program(ctx, program).build()
res_np = np.zeros(shape=total,dtype=np.uint8)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)
knl = prg.tsquare
knl(queue, res_np.shape, None, res_g)
cl.enqueue_copy(queue, res_np, res_g)
res_np=res_np.reshape(side,side)
print("done!")

file=open("which.txt","r")
num=str(int(file.read())+1)
file.close()

file=open("which.txt","w")
file.write(num)
file.close()

idee=num+str(datetime.now()).replace(":","--").replace(".","-")
file=open("Progdata"+idee+".txt","x")
file.write(program)
file.close()

png.from_array(res_np,info={"width": side,"height":side}, mode="L").save("testing"+idee+".png")
print("...and saved")
