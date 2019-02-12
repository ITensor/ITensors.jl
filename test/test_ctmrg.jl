using ITensors,
      Random,
      Test

Random.seed!(12345)

include("2d_classical_ising.jl")

function ctmrg(T::ITensor,
               Clu::ITensor,
               Al::ITensor;
               χmax::Int,nsteps::Int)
  tags(Clu,"link->link,orig")
  tags(Al,"link->link,orig")
  for i = 1:nsteps
    ## Get the grown corner transfer matrix (CTM)
    Clu⁽¹⁾ = contract(Clu,
                      Al,
                      Al,"link,down->link,left",
                         "link,up->link,right",
                         "site,left->site,up",
                      T)
    
    ## Diagonalize the grown CTM
    Ud,Cdr = eigen(Clu⁽¹⁾,"down","right";
                   truncate=χmax,
                   lefttags="link,down,renorm",
                   righttags="link,right,renorm")

    ## The renormalized CTM is the diagonal matrix of eigenvalues
    Clu = tags(Cdr,"renorm->orig",
                   "down->up",
                   "right->left")
    Clu = Clu/norm(Clu)

    ## Calculate the renormalized half row transfer matrix (HRTM)
    Al = contract(Al,
                  Ud,"down->up",
                  T,
                  Ud)
    Al = tags(Al,"renorm->orig",
                 "site,right->site,left")
    Al = Al/norm(Al)
  end
  tags(Clu,"orig->")
  tags(Al,"orig->")
  return Clu,Al
end

@testset "ctmrg" begin
  # Make Ising model MPO
  β = 1.1*βc
  d = 2
  s = Index(d,"site")
  sl = tags(s,"->left")
  sr = tags(s,"->right")
  su = tags(s,"->up")
  sd = tags(s,"->down")

  T = ising_mpo((sl,sr),(su,sd),β)

  χ0 = 1
  l = Index(χ0,"link")
  ll = tags(l,"->left")
  lu = tags(l,"->up")
  ld = tags(l,"->down")

  Clu = randomITensor(lu,ll)
  Al = randomITensor(lu,ld,sl)

  Clu = 0.5*(Clu+tags(Clu,"up<->left"))
  Al = 0.5*(Al+tags(Al,"up<->down"))

  Clu,Al = ctmrg(T,Clu,Al;χmax=30,nsteps=2000)

  # Normalize corner matrix
  trC⁴ = contract(Clu,
                  Clu,"up->up,prime",
                  Clu,"->prime",
                  Clu,"left->left,prime")
  Clu = Clu/scalar(trC⁴)^(1/4)

  # Normalize MPS tensor
  trA² = contract(Clu,
                  Clu,"up->up,prime",
                  Al,
                  Al,"link->link,prime",
                  Clu,"up->down,prime",
                      "left->left,prime",
                  Clu,"left->left,prime","up->down")
  Al = Al/sqrt(scalar(trA²))

  ## Get environment tensors for a single site measurement
  Ar = tags(Al,"site,left->site,right",
               "link->link,prime")
  Au = tags(Al,"site,left->site,up",
               "link,down->link,left",
               "link,up->link,right")
  Ad = tags(Au,"site,up->site,down",
               "link->link,prime")
  Cld = tags(Clu,"up->down",
                 "left->left,prime")
  Cru = tags(Clu,"left->right",
                 "up->up,prime")
  Crd = tags(Cru,"right->right,prime",
                 "up->down")

  ## Calculate partition function per site
  κ = scalar(contract(Clu,Al,Cld,Au,T,Ad,Cru,Ar,Crd))
  @test κ≈exp(-β*ising_free_energy(β))
  
  ## Calculate magnetization
  Tsz = ising_mpo((sl,sr),(su,sd),β;sz=true)
  m = scalar(contract(Clu,Al,Cld,Au,Tsz,Ad,Cru,Ar,Crd))/κ
  @test abs(m)≈ising_magnetization(β)
end

