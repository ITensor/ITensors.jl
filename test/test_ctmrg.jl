using ITensors,
      Random,
      Test

include("2d_classical_ising.jl")

function ctmrg(T::ITensor,
               Clu::ITensor,
               Al::ITensor;
               χmax::Int,nsteps::Int)
  Clu = addtags(Clu,"orig","link")
  Al = addtags(Al,"orig","link")
  for i = 1:nsteps
    ## Get the grown corner transfer matrix (CTM)
    Au = replacetags(replacetags(replacetags(Al,"down","left","link"),"up","right","link"),"left","up","site")
    Clu⁽¹⁾ = Clu*Al*Au*T
    
    ## Diagonalize the grown CTM
    ld = findindex(Clu⁽¹⁾, "link,down")
    sd = findindex(Clu⁽¹⁾, "site,down")
    lr = findindex(Clu⁽¹⁾, "link,right")
    sr = findindex(Clu⁽¹⁾, "site,right")
    Ud,Cdr = eigen(Clu⁽¹⁾, (ld,sd), (lr,sr);
                   maxdim=χmax,
                   lefttags="link,down,renorm",
                   righttags="link,right,renorm")

    ## The renormalized CTM is the diagonal matrix of eigenvalues
    Clu = replacetags(replacetags(replacetags(Cdr,"renorm","orig"),"down","up"),"right","left")
    Clu = Clu/norm(Clu)

    ## Calculate the renormalized half row transfer matrix (HRTM)
    Al = Al*replacetags(Ud,"down","up")*T*Ud
    Al = replacetags(replacetags(Al,"renorm","orig"),"right","left","site")
    Al = Al/norm(Al)
  end
  Clu = removetags(Clu,"orig")
  Al = removetags(Al,"orig")
  return Clu,Al
end

@testset "ctmrg" begin
  # Make Ising model MPO
  β = 1.1*βc
  d = 2
  s = Index(d,"site")
  sl = addtags(s,"left")
  sr = addtags(s,"right")
  su = addtags(s,"up")
  sd = addtags(s,"down")

  T = ising_mpo((sl,sr),(su,sd),β)

  χ0 = 1
  l = Index(χ0,"link")
  ll = addtags(l,"left")
  lu = addtags(l,"up")
  ld = addtags(l,"down")

  # Initial CTM
  Clu = ITensor(lu,ll)
  Clu[1,1] = 1.0

  # Initial HRTM
  Al = ITensor(lu,ld,sl)
  Al[lu(1),ld(1),sl(1)] = 1.0
  Al[lu(1),ld(1),sl(2)] = 0.0

  Clu,Al = ctmrg(T,Clu,Al;χmax=30,nsteps=2000)

  # Normalize corner matrix
  trC⁴ = Clu*replacetags(Clu,"0","1","up")*
         replacetags(Clu,"0","1")*replacetags(Clu,"0","1","left")
  Clu = Clu/scalar(trC⁴)^(1/4)

  # Normalize MPS tensor
  trA² = Clu*replacetags(Clu,"0","1","up")*Al*
         replacetags(Al,"0","1","link")*
         replacetags(replacetags(Clu,"up,0","down,1"),"0","1","left")*
         replacetags(replacetags(Clu,"0","1","left"),"up","down")
  Al = Al/sqrt(scalar(trA²))

  ## Get environment tensors for a single site measurement
  Ar = replacetags(replacetags(Al,"left","right","site"),"0","1","link")
  Au = replacetags(replacetags(replacetags(Al,"left","up","site"),"down","left","link"),"up","right","link")
  Ad = replacetags(replacetags(Au,"up","down","site"),"0","1","link")
  Cld = replacetags(replacetags(Clu,"up","down"),"0","1","left")
  Cru = replacetags(replacetags(Clu,"left","right"),"0","1","up")
  Crd = replacetags(replacetags(Cru,"0","1","right"),"up","down")

  ## Calculate partition function per site
  κ = scalar(Clu*Al*Cld*Au*T*Ad*Cru*Ar*Crd)
  @test κ≈exp(-β*ising_free_energy(β))
  
  ## Calculate magnetization
  Tsz = ising_mpo((sl,sr),(su,sd),β;sz=true)
  m = scalar(Clu*Al*Cld*Au*Tsz*Ad*Cru*Ar*Crd)/κ
  @test abs(m)≈ising_magnetization(β)
end

