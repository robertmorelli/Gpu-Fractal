import numpy as np
import pyopencl as cl
import png

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


program="""

__kernel void edge(__global char *res_g,__global char *image_g){
      int gid = get_global_id(0);
      if(gid>side&&gid<(side*side)-side){
      int x = (-1*image_g[gid-side-1]) + (0*image_g[gid-side])  + (1*image_g[gid-side+1])  +
              (-2*image_g[gid-1])      + (0*image_g[gid])       + (2*image_g[gid+1])       +
              (-1*image_g[gid+side-1]) + (0*image_g[gid+side])  + (1*image_g[gid+side+1])  ;
                 
      int y = (-1*image_g[gid-side-1]) + (-2*image_g[gid-side]) + (-1*image_g[gid-side+1]) +
              (0*image_g[gid-1])       + (0*image_g[gid])       + (0*image_g[gid+1])       +
              (1*image_g[gid+side-1])  + (2*image_g[gid+side])  + (1*image_g[gid+side+1])  ;

      res_g[gid] = x*x+y*y;
      }
      else{
res_g[gid]=0;
      }
      
    }
"""



program=program.replace("side","8192")





mf = cl.mem_flags
prg = cl.Program(ctx, program).build()



#pngdata=file
#np python side
res_np = np.zeros(shape=8192*8192,dtype=np.uint8)
#inp_np =  np.random.rand(16).astype(np.float32)
image_np =  np.array(list(png.Reader("tsquare.png").asDirect()[2]))
print(image_np)

#g is kernel side
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)
image_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_np)




conv = prg.edge
conv(queue, res_np.shape, None, res_g,image_g)
cl.enqueue_copy(queue, res_np, res_g)







res_np=res_np.reshape(8192,8192)
#print(res_np)
print("done!")
png.from_array(res_np,info={"width": 8192,"height":8192}, mode="L").save("tsquareedges.png")
print("...and saved")









