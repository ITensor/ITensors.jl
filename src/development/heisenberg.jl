export setElt,
       Heisenberg

function setElt(iv::IndexVal)::ITensor
  T = ITensor(ind(iv))
  T[iv] = 1.0
  return T
end

function Heisenberg(sites::SiteSet)::MPO
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
  end
  H[1] *= setElt(link[1][5])
  H[N] *= setElt(link[N+1][1])
  return H
end

