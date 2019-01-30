using ITensors,
      Test

include("2d_classical_ising.jl")

function ctmrg(T::ITensor,
               sh::Vector{Index},
               sv::Vector{Index};
               χ0::Int,χmax::Int,nsteps::Int)
  Nx,Ny = 2,2
  l = Index(χ0,"Link")
  lh = fill(Index(),(Nx,Ny))
  lv = fill(Index(),(Nx,Ny))
  for i = 1:Nx, j = 1:Ny
    lh[i,j] = addtags(l,"horiz,x$i,y$j")
    lv[i,j] = addtags(l,"vert,x$i,y$j")
  end

  Clu11 = randomITensor(lv[1,1],lh[1,1])
  Clu11 = 0.5*(Clu11+swaptags(Clu11,"vert","horiz"))

  Au11 = randomITensor(lh[1,1],lh[2,1],sv[1])
  Au11 = 0.5*(Au11+swaptags(Au11,"Link,x1","Link,x2"))
  Al11 = replacetags(swaptags(Au11,"horiz","vert"),"Link,x2,y1","Link,x1,y2")
  
  for i = 1:nsteps
    Clu11⁽¹⁾ = Clu11*Au11*Al11*T
    
    Ul12,Clu22 = eigen(Clu11⁽¹⁾,lv[1,2],sv[2];truncate=χmax,tags="Link,vert,x2,y2")
    Ul11 = replacetags(Ul12,"y2","y1")
    Ur21 = replacetags(swaptags(Ul12,"horiz","vert"),"x1,y2","x2,y1")

    # Update the links
    l = commonIndex(Ul12,Clu22)("Link")
    for i = 1:Nx, j = 1:Ny
      lh[i,j] = addtags(l,"horiz,x$i,y$j")
      lv[i,j] = addtags(l,"vert,x$i,y$j")
    end

    #TODO: get this from the eigen diagonals
    Clu11 = replacetags(Ul12*Al11*T*Clu11*Au11*Ur21,"x2,y2","x1,y1")
    Clu11 = Clu11*(1/norm(Clu11))

    Al11 = replacetags(Al11*Ul11*T*Ul12,"x2","x1")
    Al11 = Al11*(1/norm(Al11))
    Au11 = replacetags(swaptags(Al11,"horiz","vert"),"Link,x1,y2","Link,x2,y1")
  end
  return Clu11,Al11
end

@testset "ctmrg" begin
  # Make Ising model MPO
  β = 1.1*βc
  d = 2
  s = Index(d,"Site")
  Nx,Ny = 2,2
  sh = fill(Index(),Nx)
  sv = fill(Index(),Ny)
  for i = 1:Nx
    sh[i] = addtags(s,"horiz,x$i,y1")
  end
  for j = 1:Ny
    sv[j] = addtags(s,"vert,x1,y$j")
  end
  T = ising_mpo(sh,sv,β)

  Clu11,Al11 = ctmrg(T,sh,sv;χ0=2,χmax=30,nsteps=2000)

  # Normalize corner matrix
  Cru11 = replacetags(Clu11,"vert","vert,right")
  Crd11 = replacetags(Cru11,"horiz","horiz,down")
  Cld11 = replacetags(Clu11,"horiz","horiz,down")
  Clu11 = Clu11/scalar(Clu11*Cru11*Crd11*Cld11)^(1/4)

  # Normalize MPS tensor
  Cru11 = replacetags(Clu11,"vert","vert,right")
  Cld12 = replacetags(replacetags(Clu11,"y1","y2"),"horiz","horiz,down")
  Crd12 = replacetags(replacetags(Cru11,"y1","y2"),"horiz","horiz,down")
  Ar11 = replacetags(Al11,"vert","vert,right")
  Al11 = Al11/sqrt(scalar(Clu11*Cru11*Al11*Ar11*Cld12*Crd12))

  # Calculate partition function per site
  Au11 = replacetags(swaptags(Al11,"horiz","vert"),"Link,x1,y2","Link,x2,y1")
  Ad12 = replacetags(Au11,"y1","y2")
  Ar21 = replacetags(Al11,"x1","x2")
  Cru21 = replacetags(Clu11,"x1","x2")
  Crd22 = replacetags(Cru21,"y1","y2")
  Cld12 = replacetags(Clu11,"y1","y2")
  κ = scalar(Al11*Clu11*Au11*T*Cld12*Cru21*Ad12*Ar21*Crd22)
  @test κ≈exp(-β*ising_free_energy(β))
  
  # Calculate magnetization
  Tsz = ising_mpo(sh,sv,β;sz=true)
  m = scalar(Al11*Clu11*Au11*Tsz*Cld12*Cru21*Ad12*Ar21*Crd22)/κ
  @test abs(m)≈ising_magnetization(β)
end

