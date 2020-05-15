using ITensors,
      Random

function makeRandomMPS(sites;
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
  return MPS(v,0,N+1)
end

function makeRandomMPO(sites;
                       chi::Int=4)::MPO
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(chi, "Link,l=$n") for n=1:N-1]
  for n=1:N
    s = sites[n]
    if n == 1
      v[n] = ITensor(l[n],s,s')
    elseif n == N
      v[n] = ITensor(l[n-1],s,s')
    else
      v[n] = ITensor(l[n-1],s,s',l[n])
    end
    randn!(v[n])
    normalize!(v[n])
  end
  return MPO(v,0,N+1)
end

