
struct IndexSet
    inds::Vector{Index}
    IndexSet(inds::Vector{Index}) = new(inds)
end

# Empty constructor
IndexSet() = IndexSet(Index[])

# Construct of some size
IndexSet(N::Integer) = IndexSet(Vector{Index}(undef,N))

# Construct from various sets of indices
IndexSet(inds::Index...) = IndexSet(Index[inds...])
IndexSet(inds::NTuple{N,Index}) where {N} = IndexSet(inds...)

# Construct from various sets of IndexSets
IndexSet(inds::IndexSet) = inds
IndexSet(inds::IndexSet,i::Index) = IndexSet(inds...,i)
IndexSet(i::Index,inds::IndexSet) = IndexSet(i,inds...)
IndexSet(inds1::IndexSet,inds2::IndexSet) = IndexSet(inds1...,inds2...)
IndexSet(inds1::IndexSet,inds2::IndexSet,inds3::IndexSet) = IndexSet(inds1...,inds2...,inds3...)
# TODO: how do we make arbitrary N work?
IndexSet(inds::NTuple{N,IndexSet}) where {N} = IndexSet(inds...)

# Convert to an Index if there is only one
Index(is::IndexSet) = length(is)==1 ? is[1] : error("IndexSet has more than one Index")

getindex(is::IndexSet,n::Integer) = getindex(is.inds,n)
setindex!(is::IndexSet,i::Index,n::Integer) = setindex!(is.inds,i,n)
length(is::IndexSet) = length(is.inds)
order(is::IndexSet) = length(is)
copy(is::IndexSet) = IndexSet(copy(is.inds))
dims(is::IndexSet) = Tuple(dim(i) for i ∈ is)
dim(is::IndexSet) = dim(is...)

dag(is::IndexSet) = IndexSet(dag.(is.inds))

# Allow iteration
size(is::IndexSet) = size(is.inds)
iterate(is::IndexSet,state::Int=1) = iterate(is.inds,state)

push!(is::IndexSet,i::Index) = push!(is.inds,i)

# 
# Set operations
#

function in(i::Index,inds)
  is = IndexSet(inds)
  for j ∈ is
    i==j && return true
  end
  return false
end
hasindex(is,i::Index) = i ∈ is

function issubset(Ais::IndexSet,Binds)
  Bis = IndexSet(Binds)
  for i ∈ Ais
    i ∉ Bis && return false
  end
  return true
end
hasinds(Ais,Bis) = IndexSet(Ais) ⊆ Bis

function issetequal(Ais::IndexSet,Binds)
  Bis = IndexSet(Binds)
  return Ais ⊆ Bis && length(Ais) == length(Bis)
end
hassameinds(Ais,Bis) = issetequal(IndexSet(Ais),Bis)

"Output the IndexSet with Indices in Bis but not in Ais"
function setdiff(Bis::IndexSet,Ainds)
  Ais = IndexSet(Ainds)
  Cis = IndexSet()
  for j ∈ Bis
    j ∉ Ais && push!(Cis,j)
  end
  return Cis
end
uniqueinds(Bis,Ais) = setdiff(IndexSet(Bis),Ais)

"Output the IndexSet in the intersection of Ais and Bis"
function intersect(Ais::IndexSet,Binds)
  Bis = IndexSet(Binds)
  Cis = IndexSet()
  for i ∈ Ais
    i ∈ Bis && push!(Cis,i)
  end
  return Cis
end
commoninds(Ais,Bis) = IndexSet(Ais) ∩ Bis
commonindex(Ais,Bis) = Index(commoninds(Ais,Bis))

function findinds(inds, tags)
  is = IndexSet(inds)
  ts = TagSet(tags)
  found_inds = IndexSet()
  for i in is
    if hastags(i,ts)
      push!(found_inds,i)
    end
  end
  return found_inds
end
findindex(inds, tags) = Index(findinds(inds,tags))

#
# Tagging functions
#

function prime(is::IndexSet,plinc::Integer=1)
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = prime(res[jj],plinc)
  end
  return res
end
#TODO: implement a more generic version
#prime(is::IndexSet,plinc::Integer,match)
#   index_positions(is,match) -> pos::Vector{Int}
#   for p ∈ pos prime(is[p],plinc)
function prime(is::IndexSet,plinc::Integer,i::Index)
  res = copy(is)
  for jj ∈ 1:length(res)
    if res[jj]==i
      res[jj] = prime(res[jj],plinc)
    end
  end
  return res
end
prime(is::IndexSet,i::Index) = prime(is,1,i)
function prime(is::IndexSet,plinc::Integer,ts)
  res = copy(is)
  for jj ∈ 1:length(res)
    if TagSet(ts)∈tags(res[jj])
      res[jj] = prime(res[jj],plinc)
    end
  end
  return res
end
prime(is::IndexSet,ts) = prime(is,1,ts)
adjoint(is::IndexSet) = prime(is)

function setprime(is::IndexSet,plev::Int)
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = setprime(res[jj],plev)
  end
  return res
end

noprime(is::IndexSet) = setprime(is,0)

function mapprime(is::IndexSet,
                  plevold::Int,
                  plevnew::Int,
                  imatch::Index=Index())
  res = copy(is)
  for jj ∈ 1:length(res)
    if(imatch==Index() || noprime(imatch)==noprime(res[jj]))
      plev(res[jj])==plevold && (res[jj] = setprime(res[jj],plevnew))
    end
  end
  return res
end

function swapprime(is::IndexSet,
                   plev1::Int,
                   plev2::Int,
                   imatch::Index=Index())
  res = copy(is)
  plevtemp = 7017049418157811712
  res = mapprime(res,plev1,plevtemp,imatch)
  res = mapprime(res,plev2,plev1,imatch)
  res = mapprime(res,plevtemp,plev2,imatch)
  return res
end

function addtags(is::IndexSet,
                 ts::AbstractString,
                 tsmatch::String="")
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = addtags(res[jj],ts,tsmatch)
  end
  return res
end

function removetags(is::IndexSet,
                    ts::AbstractString,
                    tsmatch::String="")
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = removetags(res[jj],ts,tsmatch)
  end
  return res
end

function replacetags(is::IndexSet,
                     ts1::AbstractString,
                     ts2::AbstractString,
                     tsmatch::String="")
  res = copy(is)
  for jj ∈ 1:length(res)
    res[jj] = replacetags(res[jj],ts1,ts2,tsmatch)
  end
  return res
end

function swaptags(is::IndexSet,
                  ts1::AbstractString,
                  ts2::AbstractString,
                  tsmatch::String="")
  res = copy(is)
  tstemp = "e43efds"
  res = replacetags(res,ts1,tstemp,tsmatch)
  res = replacetags(res,ts2,ts1,tsmatch)
  res = replacetags(res,tstemp,ts2,tsmatch)
  return res
end

function calculate_permutation(set1, set2)
  l1 = length(set1)
  l2 = length(set2)
  l1==l2 || throw(DimensionMismatch("Mismatched input sizes in calcPerm: l1=$l1, l2=$l2"))
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

function contract_inds(Ais::IndexSet,
                       Aind,
                       Bis::IndexSet,
                       Bind)
  ncont = 0
  for i in Aind
    if(i < 0) ncont += 1 end 
  end
  nuniq = length(Ais)+length(Bis)-2*ncont
  Cind = zeros(Int,nuniq)
  Cis = fill(Index(),nuniq)
  u = 1
  for i ∈ 1:length(Ais)
    if(Aind[i] > 0) 
      Cind[u] = Aind[i]; 
      Cis[u] = Ais[i]; 
      u += 1 
    end
  end
  for i ∈ 1:length(Bis)
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

