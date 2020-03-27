using ITensors,
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
    Au = replacetags(Al,"down,link","left,link")
    Au = replacetags(Au,"up,link","right,link")
    Au = replacetags(Au,"left,site","up,site")

    Clu⁽¹⁾ = Clu*Al*Au*T

    ## Diagonalize the grown CTM
    ld = firstind(Clu⁽¹⁾,"link,down")
    sd = firstind(Clu⁽¹⁾,"site,down")
    lr = firstind(Clu⁽¹⁾,"link,right")
    sr = firstind(Clu⁽¹⁾,"site,right")

    Ud,Cdr = eigen(Clu⁽¹⁾, (ld,sd), (lr,sr); ishermitian=true,
                                             maxdim=χmax,
                                             lefttags="link,down,renorm",
                                             righttags="link,right,renorm",
                                             truncate=true)

    ## The renormalized CTM is the diagonal matrix of eigenvalues
    Clu = replacetags(Cdr,"renorm","orig")
    Clu = replacetags(Clu,"down","up")
    Clu = replacetags(Clu,"right","left")
    Clu = Clu/norm(Clu)

    ## Calculate the renormalized half row transfer matrix (HRTM)
    Uu = replacetags(Ud,"down","up")

    Al = Al*Uu*T*Ud
    Al = replacetags(Al,"renorm","orig")
    Al = replacetags(Al,"right,site","left,site")
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
  trC⁴ = Clu*mapprime(Clu,0,1,"up")*
         mapprime(Clu,0,1)*mapprime(Clu,0,1,"left")
  Clu = Clu/scalar(trC⁴)^(1/4)

  # Normalize MPS tensor
  trA² = Clu*mapprime(Clu,0,1,"up")*Al*
         mapprime(Al,0,1,"link")*
         mapprime(replacetags(mapprime(Clu,0,1,"up"),"up","down"),0,1,"left")*
         replacetags(mapprime(Clu,0,1,"left"),"up","down")
  Al = Al/sqrt(scalar(trA²))

  ## Get environment tensors for a single site measurement
  Ar = mapprime(replacetags(Al,"left","right","site"),0,1,"link")
  Au = replacetags(replacetags(replacetags(Al,"left","up","site"),
                                              "down","left","link"),
                                              "up","right","link")
  Ad  = mapprime(replacetags(Au,"up","down","site"),0,1,"link")
  Cld = mapprime(replacetags(Clu,"up","down"),0,1,"left")
  Cru = mapprime(replacetags(Clu,"left","right"),0,1,"up")
  Crd = replacetags(mapprime(Cru,0,1,"right"),"up","down")

  ## Calculate partition function per site
  κ = scalar(Clu*Al*Cld*Au*T*Ad*Cru*Ar*Crd)
  @test κ≈exp(-β*ising_free_energy(β))
  
  ## Calculate magnetization
  Tsz = ising_mpo((sl,sr),(su,sd),β;sz=true)
  m = scalar(Clu*Al*Cld*Au*Tsz*Ad*Cru*Ar*Crd)/κ
  @test abs(m)≈ising_magnetization(β)
end

