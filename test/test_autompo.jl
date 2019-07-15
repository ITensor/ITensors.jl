using ITensors,
      Test,
      Random

function setElt(iv::IndexVal)::ITensor
  T = ITensor(ind(iv))
  T[iv] = 1.0
  return T
end

function makeRandomMPS(sites::SiteSet,
                     chi::Int=4)::MPS
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(chi, "Link,l=$n") for n=1:N-1]
  for n=1:N
    s = sites[n]
    if n == 1
      v[n] = ITensor(l[n], s)
    elseif n == N
      v[n] = ITensor(l[n-1], s)
    else
      v[n] = ITensor(l[n-1], l[n], s)
    end
    randn!(v[n])
    normalize!(v[n])
  end
  return MPS(N,v,0,N+1)
end

function isingMPO(sites::SiteSet)::MPO
  H = MPO(sites)
  N = length(H)
  link = fill(Index(),N+1)
  for n=1:N+1
    link[n] = Index(3,"Link,Ising,l=$(n-1)")
  end
  for n=1:N
    s = sites[n]
    ll = link[n]
    rl = link[n+1]
    H[n] = ITensor(dag(ll),dag(s),s',rl)
    H[n] += setElt(ll[1])*setElt(rl[1])*op(sites,"Id",n)
    H[n] += setElt(ll[3])*setElt(rl[3])*op(sites,"Id",n)
    H[n] += setElt(ll[2])*setElt(rl[1])*op(sites,"Sz",n)
    H[n] += setElt(ll[3])*setElt(rl[2])*op(sites,"Sz",n)
  end
  LE = ITensor(link[1]); LE[3] = 1.0;
  RE = ITensor(dag(link[N+1])); RE[1] = 1.0;
  H[1] *= LE
  H[N] *= RE
  return H
end

function heisenbergMPO(sites::SiteSet,
                       h::Vector{Float64})::MPO
  H = MPO(sites)
  N = length(H)
  link = fill(Index(),N+1)
  for n=1:N+1
    link[n] = Index(5,"Link,Heis,l=$(n-1)")
  end
  for n=1:N
    s = sites[n]
    ll = link[n]
    rl = link[n+1]
    H[n] = ITensor(ll,s,s',rl)
    H[n] += setElt(ll[1])*setElt(rl[1])*op(sites,"Id",n)
    H[n] += setElt(ll[5])*setElt(rl[5])*op(sites,"Id",n)
    H[n] += setElt(ll[2])*setElt(rl[1])*op(sites,"S+",n)
    H[n] += setElt(ll[3])*setElt(rl[1])*op(sites,"S-",n)
    H[n] += setElt(ll[4])*setElt(rl[1])*op(sites,"Sz",n)
    H[n] += setElt(ll[5])*setElt(rl[2])*op(sites,"S-",n)*0.5
    H[n] += setElt(ll[5])*setElt(rl[3])*op(sites,"S+",n)*0.5
    H[n] += setElt(ll[5])*setElt(rl[4])*op(sites,"Sz",n)
    H[n] += setElt(ll[5])*setElt(rl[1])*op(sites,"Sz",n)*h[n]
  end
  H[1] *= setElt(link[1][5])
  H[N] *= setElt(link[N+1][1])
  return H
end

@testset "AutoMPO" begin

  N = 10

  @testset "Single creation op" begin
    sites = electronSites(N)
    ampo = AutoMPO(sites)
    add!(ampo,"Cdagup",3)
    W = toMPO(ampo)
    psi = makeRandomMPS(sites)
    cdu_psi = copy(psi)
    cdu_psi[3] = noprime(cdu_psi[3]*op(sites,"Cdagup",3))
    @test inner(psi,W,psi) ≈ inner(cdu_psi,psi)
  end

  @testset "Ising" begin
    sites = spinHalfSites(N)
    ampo = AutoMPO(sites)
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
    end
    Ha = toMPO(ampo)
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

  @testset "Ising-Different Order" begin
    sites = spinHalfSites(N)
    ampo = AutoMPO(sites)
    for j=1:N-1
      add!(ampo,"Sz",j+1,"Sz",j)
    end
    Ha = toMPO(ampo)
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

  @testset "Heisenberg" begin
    sites = spinHalfSites(N)
    ampo = AutoMPO(sites)
    h = rand(N) #random magnetic fields
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
      add!(ampo,0.5,"S+",j,"S-",j+1)
      add!(ampo,0.5,"S-",j,"S+",j+1)
      add!(ampo,h[j],"Sz",j)
    end
    add!(ampo,h[N],"Sz",N)

    Ha = toMPO(ampo)
    He = heisenbergMPO(sites,h)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

end
