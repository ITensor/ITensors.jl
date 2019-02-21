using ITensors,
      Random,
      Test

Random.seed!(12345)

include("2d_classical_ising.jl")

function factorize(A::ITensor,Ltags::NTuple{NL,String},Rtags::NTuple{NR,String};maxm::Int,tags::String) where {NL,NR}
  Linds = NTuple{NL}((findtags(A,tags) for tags ∈ Ltags))
  Rinds = NTuple{NR}((findtags(A,tags) for tags ∈ Rtags))
  IndexSet(Linds...,Rinds...)!=inds(A) && error("Tags must match those contained by the ITensor")
  U,S,V = svd(A,Linds...;maxm=maxm,utags="$tags,u",vtags="$tags,v")
  u = commonindex(U,S)
  v = commonindex(S,V)
  for ss = 1:dim(u)
    S[ss,ss] = sqrt(S[ss,ss])
  end
  FU = removetags(U*S,"v")
  FV = removetags(S*V,"u")
  return FU,FV
end

function trace(A::ITensor,tags1::Tuple{String,String},tags::Tuple{String,String}...)
  return trace(trace(A,tags1),tags...)
end

function trace(A::ITensor,tags::Tuple{String,String})
  i1 = findtags(A,tags[1])
  i2 = findtags(A,tags[2])
  return A*δ(i1,i2)
end

"""
trg(T::ITensor; χmax::Int, nsteps::Int) -> κ,T

Perform the TRG algorithm on the partition function composed of the ITensor T.
T is assumed to have Indices with tags "left", "right", "up", and "down".

The indices of T must obey: 

`findindex(T,"left") = tags(findindex(T,"right"),"right->left")`

`findindex(T,"up") = tags(findindex(T,"down"),"down->up")`

χmax is the maximum renormalized bond dimension.

nsteps are the number of renormalization steps performed.

The outputs are κ, the partition function per site, and the final renormalized
ITensor T (also with Indices with tags "left","right","up", and "down").
"""
function trg(T::ITensor;
             χmax::Int, nsteps::Int)
  # Check tags are correct
  @assert findtags(T,"left") == tags(findtags(T,"right"),"right->left")
  @assert findtags(T,"up") == tags(findtags(T,"down"),"down->up")
  
  # Keep track of the partition function per site
  κ = 1.0

  T = tags(T," -> orig")
  for n = 1:nsteps
    Fr,Fl = factorize(T,("left","up"),("right","down");maxm=χmax,tags="renorm")
    Fd,Fu = factorize(T,("right","up"),("left","down");maxm=χmax,tags="renorm")

    Fl = tags(Fl,"renorm -> renorm,left")
    Fr = tags(Fr,"renorm -> renorm,right")
    Fu = tags(Fu,"renorm -> renorm,up")
    Fd = tags(Fd,"renorm -> renorm,down")

    T = tags(Fl,"orig,down -> downleft","orig,right -> upleft")*
        tags(Fu,"orig,left -> upleft","orig,down -> upright")* 
        tags(Fr,"orig,up -> upright","orig,left -> downright")*
        tags(Fd,"orig,right -> downright","orig,up -> downleft")
    T = tags(T,"renorm -> orig")

    trT = abs(scalar(trace(T,("left","right"),("up","down"))))
    T = T/trT
    κ *= trT^(1.0/2^n)
  end
  T = tags(T,"orig -> ")
  return κ,T
end

@testset "trg" begin
  # Make Ising model MPO
  β = 1.1*βc
  d = 2
  s = Index(d)
  l = tags(s," -> left")
  r = tags(s," -> right")
  u = tags(s," -> up")
  d = tags(s," -> down")
  T = ising_mpo((l,r),(u,d),β)

  χmax = 20
  nsteps = 20
  κ,T = trg(T;χmax=χmax,nsteps=nsteps)

  @test κ≈exp(-β*ising_free_energy(β)) atol=1e-4
end

