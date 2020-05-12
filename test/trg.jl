using ITensors,
      Test
import Random

Random.seed!(12345)

include("2d_classical_ising.jl")

"""
trg(T::ITensor; χmax::Int, nsteps::Int) -> κ,T

Perform the TRG algorithm on the partition function composed of the ITensor T.
T is assumed to have Indices with tags "left", "right", "up", and "down".

The indices of T must obey: 

`firstind(T,"left") = tags(firstind(T,"right"),"right->left")`

`firstind(T,"up") = tags(firstind(T,"down"),"down->up")`

χmax is the maximum renormalized bond dimension.

nsteps are the number of renormalization steps performed.

The outputs are κ, the partition function per site, and the final renormalized
ITensor T (also with Indices with tags "left","right","up", and "down").
"""
function trg(T::ITensor, horiz_inds, vert_inds;
             χmax::Int, nsteps::Int)

  l = horiz_inds[1]
  r = horiz_inds[2]
  u = vert_inds[1]
  d = vert_inds[2]

  @assert hassameinds((l,r,u,d),T)

  T = addtags(T,"orig")
  l = addtags(l,"orig")
  r = addtags(r,"orig")
  u = addtags(u,"orig")
  d = addtags(d,"orig")

  # Keep track of the partition function per site
  κ = 1.0

  for n = 1:nsteps
    Fr,Fl = factorize(T, (l,u); ortho="none",
                                maxdim=χmax,
                                tags="renorm")
    Fd,Fu = factorize(T, (r,u); ortho="none",
                                maxdim=χmax,
                                tags="renorm")

    Fl = addtags(Fl,"left","renorm")
    Fr = addtags(Fr,"right","renorm")
    Fu = addtags(Fu,"up","renorm")
    Fd = addtags(Fd,"down","renorm")

    Fl = replacetags(replacetags(Fl,"down","dnleft","orig"),
                                    "right","upleft","orig")
    Fu = replacetags(replacetags(Fu,"left","upleft","orig"),
                                    "down","upright","orig")
    Fr = replacetags(replacetags(Fr,"up","upright","orig"),
                                    "left","dnright","orig")
    Fd = replacetags(replacetags(Fd,"right","dnright","orig"),
                                    "up","dnleft","orig")

    T = Fl*Fu*Fr*Fd
    T = replacetags(T,"renorm","orig")

    l = firstind(T,"left")
    r = firstind(T,"right")
    u = firstind(T,"up")
    d = firstind(T,"down")

    trT = abs((T*δ(l,r)*δ(u,d))[])
    T = T/trT
    κ *= trT^(1.0/2^n)
  end
  T = removetags(T,"orig")
  l = firstind(T,"left")
  r = firstind(T,"right")
  u = firstind(T,"up")
  d = firstind(T,"down")
  return κ,T,(l,r),(u,d)
end

@testset "trg" begin
  # Make Ising model MPO
  β = 1.1*βc
  d = 2
  s = Index(d)
  l = addtags(s,"left")
  r = addtags(s,"right")
  u = addtags(s,"up")
  d = addtags(s,"down")
  T = ising_mpo((l,r),(u,d),β)

  χmax = 20
  nsteps = 20
  κ,T,(l,r),(u,d) = trg(T,(l,r),(u,d);χmax=χmax,nsteps=nsteps)

  @test κ≈exp(-β*ising_free_energy(β)) atol=1e-4
end

nothing
