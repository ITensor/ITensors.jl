using ITensors,
      Test

function makeRandMPS(sites::SiteSet,
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

@testset "AutoMPO" begin

  N = 4
  sites = SiteSet(N,2)

  @testset "Ising" begin
    ampo = AutoMPO(sites)
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
    end
    Ha = toMPO(ampo)
    He = isingMPO(sites)
    psi = makeRandMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

end
