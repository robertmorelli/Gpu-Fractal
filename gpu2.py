import numpy as np
import pyopencl as cl
import png
from datetime import datetime
import pygame


size=10
side=2**(size)
side3=3*side
total=side3*side
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
program="""
__kernel void chronch(__global char *res_g){
      int gid = get_global_id(0)/3;///3;
      //int rgb = get_global_id(0)%3;
      
      ulong x=gid% side;
      ulong y=gid/ side;
      //s i z e size
      //s i d e side

      res_g[gid*3+1] = 10*((popcount(popcount(
      (((x<<2)^(x))&((y<<2)^(y)))&~((3<<11)|3)
      ))));
      res_g[gid*3] = (res_g[gid*3+1]+10)*(10*(((popcount(
      (
      ((x<<size)^(x&side))&((y<<size)^(y&side))
      )
      ))))+10)
      *
      (10*(((popcount(
      (
      (((side-x)<<size)^((side-x)&side))&(((side-y)<<size)^((side-y)&side))
      )
      ))))+10)
      ;
      
      res_g[gid*3+2]=(255-10*popcount(
      (((x<<1)&~(1<<12))&~(x&~1))&((((y<<1)&~(1<<12))&~(y&~1)))
      ));

      
  }
"""
program=program.replace("side3",str(side3))
program=program.replace("side",str(side))
program=program.replace("size",str(size))
mf = cl.mem_flags
prg = cl.Program(ctx, program).build()
res_np = np.zeros(shape=total,dtype=np.uint8)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)
knl = prg.chronch
knl(queue, res_np.shape, None, res_g)
cl.enqueue_copy(queue, res_np, res_g)
#res_np=res_np.reshape(side*side,3)
res_np=res_np.reshape(side,side3)
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

png.from_array(res_np,info={"width": side,"height":side}, mode="RGB").save("testing"+idee+".png")
print("...png saved")

res_np=res_np.reshape(side,side,3)










#pygame.init()
#display = pygame.display.set_mode((side, side))
#surf = pygame.surfarray.make_surface(res_np)

running = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    display.blit(surf, (0, 0))
    pygame.display.update()
pygame.quit()











