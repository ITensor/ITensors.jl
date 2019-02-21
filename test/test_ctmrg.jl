using ITensors,
      Random,
      Test

Random.seed!(12345)

include("2d_classical_ising.jl")

function ctmrg(T::ITensor,
               Clu::ITensor,
               Al::ITensor;
               χmax::Int,nsteps::Int)
  Clu = tags(Clu,"link -> link,orig")
  Al = tags(Al,"link -> link,orig")
  for i = 1:nsteps
    ## Get the grown corner transfer matrix (CTM)
    Au = tags(Al,"link,down -> link,left",
                 "link,up -> link,right",
                 "site,left -> site,up")
    Clu⁽¹⁾ = Clu*Al*Au*T
    
    ## Diagonalize the grown CTM
    Ud,Cdr = eigen(Clu⁽¹⁾,"down","right";
                   truncate=χmax,
                   lefttags="link,down,renorm",
                   righttags="link,right,renorm")

    ## The renormalized CTM is the diagonal matrix of eigenvalues
    Clu = tags(Cdr,"renorm -> orig",
                   "down -> up",
                   "right -> left")
    Clu = Clu/norm(Clu)

    ## Calculate the renormalized half row transfer matrix (HRTM)
    Al = Al*tags(Ud,"down -> up")*T*Ud
    Al = tags(Al,"renorm -> orig",
                 "site,right -> site,left")
    Al = Al/norm(Al)
  end
  Clu = tags(Clu,"orig -> ")
  Al = tags(Al,"orig -> ")
  return Clu,Al
end

@testset "ctmrg" begin
  # Make Ising model MPO
  β = 1.1*βc
  d = 2
  s = Index(d,"site")
  sl = tags(s," -> left")
  sr = tags(s," -> right")
  su = tags(s," -> up")
  sd = tags(s," -> down")

  T = ising_mpo((sl,sr),(su,sd),β)

  χ0 = 1
  l = Index(χ0,"link")
  ll = tags(l," -> left")
  lu = tags(l," -> up")
  ld = tags(l," -> down")

  Clu = randomITensor(lu,ll)
  Al = randomITensor(lu,ld,sl)

  Clu = 0.5*(Clu+tags(Clu,"up <-> left"))
  Al = 0.5*(Al+tags(Al,"up <-> down"))

  Clu,Al = ctmrg(T,Clu,Al;χmax=30,nsteps=2000)

  # Normalize corner matrix
  trC⁴ = Clu*tags(Clu,"up,0 -> up,1")*
         tags(Clu,"0 -> 1")*tags(Clu,"left,0 -> left,1")
  Clu = Clu/scalar(trC⁴)^(1/4)

  # Normalize MPS tensor
  trA² = Clu*tags(Clu,"up,0 -> up,1")*Al*
         tags(Al,"link,0 -> link,1")*
         tags(Clu,"up,0 -> down,1","left,0 -> left,1")*
         tags(Clu,"left,0 -> left,1","up -> down")
  Al = Al/sqrt(scalar(trA²))

  ## Get environment tensors for a single site measurement
  Ar = tags(Al,"site,left -> site,right",
               "link,0 -> link,1")
  Au = tags(Al,"site,left -> site,up",
               "link,down -> link,left",
               "link,up -> link,right")
  Ad = tags(Au,"site,up -> site,down",
               "link,0 -> link,1")
  Cld = tags(Clu,"up -> down",
                 "left,0 -> left,1")
  Cru = tags(Clu,"left -> right",
                 "up,0 -> up,1")
  Crd = tags(Cru,"right,0 -> right,1",
                 "up -> down")

  ## Calculate partition function per site
  κ = scalar(Clu*Al*Cld*Au*T*Ad*Cru*Ar*Crd)
  @test κ≈exp(-β*ising_free_energy(β))
  
  ## Calculate magnetization
  Tsz = ising_mpo((sl,sr),(su,sd),β;sz=true)
  m = scalar(Clu*Al*Cld*Au*Tsz*Ad*Cru*Ar*Crd)/κ
  @test abs(m)≈ising_magnetization(β)
end

