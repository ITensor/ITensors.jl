
struct IndexSet
    inds::Vector{Index}
    IndexSet(N::Integer) = new(Vector{Index}(undef,N))
    IndexSet(inds::Vector{Index}) = new(inds)
    IndexSet(inds::Index...) = new([inds...])
end

getindex(is::IndexSet,n::Integer) = getindex(is.inds,n)
setindex!(is::IndexSet,i::Index,n::Integer) = setindex!(is.inds,i,n)
length(is::IndexSet) = length(is.inds)
rank(is::IndexSet) = length(is)
order(is::IndexSet) = length(is)
copy(is::IndexSet) = IndexSet(copy(is.inds))
dims(is::IndexSet) = Tuple(dim(i) for i ∈ is)
dim(is::IndexSet) = dim(is...)

dag(is::IndexSet) = IndexSet(dag.(is.inds))

# Allow iteration
size(is::IndexSet) = size(is.inds)
iterate(is::IndexSet,state=1) = iterate(is.inds,state)

push!(is::IndexSet,i::Index) = push!(is.inds,i)

function hasindex(is::IndexSet,i::Index)
  for j ∈ is
    i==j && return true
  end
  return false
end

function ==(Ais::IndexSet,Bis::IndexSet)
  order(Ais)!=order(Bis) && throw(ErrorException("IndexSets must have the same number of Indices to be equal"))
  for i ∈ Ais
    !hasindex(Bis,i) && return false
  end
  return true
end
!=(Ais::IndexSet,Bis::IndexSet) = !(Ais==Bis)

function issubset(Ais::IndexSet,Bis::IndexSet)
  for i ∈ Ais
    !hasindex(Bis,i) && return false
  end
  return true
end

"Output the IndexSet with Indices in Bis but not in Ais"
function difference(Bis::IndexSet,Ais::IndexSet)
  Cis = IndexSet()
  for j ∈ Bis
    !hasindex(Ais,j) && push!(Cis,j)
  end
  return Cis
end

"Output the IndexSet in the intersection of Ais and Bis"
function intersect(Ais::IndexSet,Bis::IndexSet)
  Cis = IndexSet()
  for i ∈ Ais
    hasindex(Bis,i) && push!(Cis,i)
  end
  return Cis
end

function commonindex(Ais::IndexSet,Bis::IndexSet)
  Cis = Ais∩Bis
  if order(Cis)>1 throw(ErrorException("IndexSets have more than one common Index"))
  elseif order(Cis)==1 return Cis[1]
  end
  return Index()
end

function prime(is::IndexSet,plinc::Integer=1)
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = prime(res[jj],plinc)
  end
  return res
end
function prime(is::IndexSet,i::Index,plinc::Integer=1)
  res = copy(is)
  for jj ∈ 1:length(res)
    if res[jj]==i
      res[jj] = prime(res[jj],plinc)
    end
  end
  return res
end
function primeexcept(is::IndexSet,i::Index,plinc::Integer=1)
  res = copy(is)
  for jj ∈ 1:length(res)
    if res[jj]!=i
      res[jj] = prime(res[jj],plinc)
    end
  end
  return res
end

function setprime(is::IndexSet,plev::Int)
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = setprime(res[jj],plev)
  end
  return res
end

noprime(is::IndexSet) = setprime(is,0)

function mapprime(is::IndexSet,plevold::Int,plevnew::Int,imatch::Index=Index())
  res = copy(is)
  for jj ∈ 1:length(res)
    if(imatch==Index() || noprime(imatch)==noprime(res[jj]))
      plev(res[jj])==plevold && (res[jj] = setprime(res[jj],plevnew))
    end
  end
  return res
end

function swapprime(is::IndexSet,plev1::Int,plev2::Int,imatch::Index=Index())
  res = copy(is)
  plevtemp = 7017049418157811712
  res = mapprime(res,plev1,plevtemp,imatch)
  res = mapprime(res,plev2,plev1,imatch)
  res = mapprime(res,plevtemp,plev2,imatch)
  return res
end

function addtags(is::IndexSet,tags::String)
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = addtags(res[jj],tags)
  end
  return res
end

function removetags(is::IndexSet,tags::String)
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = removetags(res[jj],tags)
  end
  return res
end

function replacetags(is::IndexSet,ts1::String,ts2::String)
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = replacetags(res[jj],ts1,ts2)
  end
  return res
end

function swaptags(is::IndexSet,ts1::String,ts2::String)
  res = copy(is)
  tstemp = "e43efds"
  res = replacetags(res,ts1,tstemp)
  res = replacetags(res,ts2,ts1)
  res = replacetags(res,tstemp,ts2)
  return res
end

function calculate_permutation(set1,set2)
  l1 = length(set1)
  l2 = length(set2)
  l1==l2 || error("Mismatched input sizes in calcPerm")
  p = zeros(Int,l1)
  for i1 = 1:l1
    for i2 = 1:l2
      if set1[i1]==set2[i2]
        p[i1] = i2
        break
      end
    end #i2
    p[i1]!=0 || error("Sets aren't permutations of each other")
  end #i1
  return p
end

function compute_contraction_labels(Ai::IndexSet,Bi::IndexSet)
  rA = order(Ai)
  rB = order(Bi)
  Aind = zeros(Int,rA)
  Bind = zeros(Int,rB)

  ncont = 0
  for i = 1:rA, j = 1:rB
    if Ai[i]==Bi[j]
      Aind[i] = Bind[j] = -(1+ncont)
      ncont += 1
    end
  end

  u = ncont
  for i = 1:rA
    if(Aind[i]==0) Aind[i] = (u+=1) end
  end
  for j = 1:rB
    if(Bind[j]==0) Bind[j] = (u+=1) end
  end

  return (Aind,Bind)
end

function contract_inds(Ais::IndexSet,Aind,
                       Bis::IndexSet,Bind)
  ncont = 0
  for i in Aind
    if(i < 0) ncont += 1 end 
  end
  nuniq = rank(Ais)+rank(Bis)-2*ncont
  Cind = zeros(Int,nuniq)
  Cis = fill(Index(),nuniq)
  u = 1
  for i = 1:rank(Ais)
    if(Aind[i] > 0) 
      Cind[u] = Aind[i]; 
      Cis[u] = Ais[i]; 
      u += 1 
    end
  end
  for i = 1:rank(Bis)
    if(Bind[i] > 0) 
      Cind[u] = Bind[i]; 
      Cis[u] = Bis[i]; 
      u += 1 
    end
  end
  return (IndexSet(Cis...),Cind)
end

function compute_strides(inds::IndexSet)
  r = order(inds)
  stride = zeros(Int, r)
  s = 1
  for i = 1:r
    stride[i] = s
    s *= dim(inds[i])
  end
  return stride
end

