using ITensors
  
function heisenberg(s::Vector{<:Index})
  N = length(s)

  l = [Index(QN(0)=>3,QN(-2)=>1,QN(2)=>1; tags="l=$(l-1)") for l in 1:N+1]

  h = ITensor[ITensor(dag(l[i]),l[i+1],dag(s[i]),s[i]') for i in 1:N]

  H = MPO(h)

  for n in 1:N
    H[n] += op(s[n],"Id")*setelt(l[n](1))*setelt(l[n+1](1))
    H[n] += op(s[n],"Id")*setelt(l[n](2))*setelt(l[n+1](2))

    H[n] += op(s[n],"Sz")*setelt(l[n](3))*setelt(l[n+1](1))
    H[n] += op(s[n],"Sz")*setelt(l[n](2))*setelt(l[n+1](3))

    H[n] += op(s[n],"S⁻")*setelt(l[n](4))*setelt(l[n+1](1))
    H[n] += 0.5*op(s[n],"S⁺")*setelt(l[n](2))*setelt(l[n+1](4))

    H[n] += op(s[n],"S⁺")*setelt(l[n](5))*setelt(l[n+1](1))
    H[n] += 0.5*op(s[n],"S⁻")*setelt(l[n](2))*setelt(l[n+1](5))
  end
  H[1] *= setelt(l[1](2))
  H[N] *= setelt(dag(l[N+1])(1));

  return H
end

