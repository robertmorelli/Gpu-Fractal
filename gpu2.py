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

bezcurve=((20,80),(40,30),(70,90))

program="""
__kernel void chronch(__global char *res_g){
      //s i d e side
      //s i z e size

      uint gid = get_global_id(0);
      float r=(gid/(side.0*side.0*3.0));
      float band=(2*((gid%side)/(side.0))-1);

      float xf=(r*(400.0-200.0)*band)+200.0;
      float yf=(r*(300.0-800.0)*band)+800.0;


      float xl=(r*(700.0-400.0))+400.0;
      float yl=(r*(900.0-300.0))+300.0;

      uint x=(r*(xl-xf))+xf;
      uint y=(r*(yl-yf))+yf;

      res_g[3*(x+y*side)]=(((band+1)/2)*((band+1)/2)*232+20)*.8;
      res_g[3*(x+y*side)+1]=res_g[3*(x+y*side)];
      res_g[3*(x+y*side)+2]=(1-((band+1)/2))*(1-((band+1)/2))*225+30;
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











