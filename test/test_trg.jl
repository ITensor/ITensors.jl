using ITensors,
      Test,
      Printf

include("2d_classical_ising.jl")

function factorize(A::ITensor,Linds::Index...;maxm::Int,tags::String)
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

function trg(T::ITensor,
             sh::Vector{Index},
             sv::Vector{Index};
             χmax::Int,nsteps::Int)
  #Keep track of the partition function per site
  κ = 1.0
  for n = 1:nsteps
    #@printf("Doing TRG Step %d\n",n)
    Fh1,Fh2 = factorize(T,sh[1],sv[1];maxm=χmax,tags="renorm")
    Fv1,Fv2 = factorize(T,sh[2],sv[1];maxm=χmax,tags="renorm")
    #TODO: replace this with sh[1] = findindex(Fh1,"renorm")("orig,h1")
    sh[1] = commonindex(Fh1,Fh2)("orig,h1")
    sh[2] = sh[1]("orig,h2")
    sv[1] = commonindex(Fv1,Fv2)("orig,v1")
    sv[2] = sv[1]("orig,v2")

    #TODO: replace this with Fh1 = addtags(Fh1,"h1","renorm"), etc.
    Fh1 = replacetags(Fh1,"renorm","renorm,h1")
    Fh2 = replacetags(Fh2,"renorm","renorm,h2")
    Fv1 = replacetags(Fv1,"renorm","renorm,v1")
    Fv2 = replacetags(Fv2,"renorm","renorm,v2")

    Fv1 = replacetags(replacetags(Fv1,"orig,h2","orig,h1"),"orig,v1","orig,v2")
    Fv2 = replacetags(replacetags(Fv2,"orig,h1","orig,h2"),"orig,v2","orig,v1")

    T = replacetags(Fv1*Fh1*Fv2*Fh2,"renorm","orig")

    trT = abs(scalar(T*δ(sh[1],sh[2])*δ(sv[1],sv[2])))
    T = T/trT
    κ *= trT^(1.0/2^n)
  end
  return κ
end

@testset "trg" begin
  # Make Ising model MPO
  β = 1.1*βc
  d = 2
  sh = fill(Index(d),2)
  sv = fill(Index(d),2)
  for i = 1:2
    sh[i] = sh[i]("orig,h$i")
    sv[i] = sv[i]("orig,v$i")
  end
  T = ising_mpo(sh,sv,β)

  χmax = 20
  nsteps = 20
  κ = trg(T,sh,sv;χmax=χmax,nsteps=nsteps)

  @test κ≈exp(-β*ising_free_energy(β)) atol=1e-4
end

