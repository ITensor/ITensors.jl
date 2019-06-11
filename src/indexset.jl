
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
IndexSet(is1::IndexSet,is2::IndexSet) = IndexSet(is1...,is2...)
IndexSet(inds::NTuple{2,IndexSet}) = IndexSet(inds...)

# Convert to an Index if there is only one
Index(is::IndexSet) = length(is)==1 ? is[1] : error("IndexSet has more than one Index")

getindex(is::IndexSet,n::Integer) = getindex(is.inds,n)
setindex!(is::IndexSet,i::Index,n::Integer) = setindex!(is.inds,i,n)
length(is::IndexSet) = length(is.inds)
order(is::IndexSet) = length(is)
copy(is::IndexSet) = IndexSet(copy(is.inds))
dims(is::IndexSet) = Tuple(dim(i) for i ∈ is)
dim(is::IndexSet) = prod(dim.(is))

dag(is::IndexSet) = IndexSet(dag.(is.inds))

# Allow iteration
size(is::IndexSet) = size(is.inds)
iterate(is::IndexSet,state::Int=1) = iterate(is.inds,state)

push!(is::IndexSet,i::Index) = push!(is.inds,i)

# 
# Set operations
#

function hasindex(inds,i::Index)
  is = IndexSet(inds)
  for j ∈ is
    i==j && return true
  end
  return false
end

function hasinds(Binds,Ainds)
  Ais = IndexSet(Ainds)
  for i ∈ Ais
    !hasindex(Binds,i) && return false
  end
  return true
end
hasinds(Binds,Ainds::Index...) = hasinds(Binds,IndexSet(Ainds...))

function hassameinds(Ainds,Binds)
  Ais = IndexSet(Ainds)
  Bis = IndexSet(Binds)
  return hasinds(Ais,Bis) && length(Ais) == length(Bis)
end

"Output the IndexSet with Indices in Bis but not in Ais"
function uniqueinds(Binds,Ainds)
  Bis = IndexSet(Binds)
  Cis = IndexSet()
  for j ∈ Bis
    !hasindex(Ainds,j) && push!(Cis,j)
  end
  return Cis
end

"Output the IndexSet in the intersection of Ais and Bis"
function commoninds(Binds,Ainds)
  Ais = IndexSet(Ainds)
  Cis = IndexSet()
  for i ∈ Ais
    hasindex(Binds,i) && push!(Cis,i)
  end
  return Cis
end
commonindex(Ais,Bis) = Index(commoninds(Ais,Bis))

function findinds(inds,tags)
  is = IndexSet(inds)
  ts = TagSet(tags)
  found_inds = IndexSet()
  for i ∈ is
    if hastags(i,ts)
      push!(found_inds,i)
    end
  end
  return found_inds
end
findindex(inds, tags) = Index(findinds(inds,tags))

# From a tag set or index set, find the positions
# of the matching indices as a vector of integers
indexpositions(inds, match::Nothing) = collect(1:length(inds))
# Version for matching a tag set
function indexpositions(inds, match::T) where {T<:Union{AbstractString,TagSet}}
  tsmatch = TagSet(match)
  pos = Int[]
  for (j,I) ∈ enumerate(inds)
    hastags(I,tsmatch) && push!(pos,j)
  end
  return pos
end
# Version for matching a collection of indices
function indexpositions(inds, match)
  ismatch = IndexSet(match)
  pos = Int[]
  for (j,I) ∈ enumerate(inds)
    hasindex(ismatch,I) && push!(pos,j)
  end
  return pos
end

#
# Tagging functions
#

function prime!(is::IndexSet, plinc::Integer, match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = prime(is[jj],plinc)
  end
  return is
end
prime!(is::IndexSet,match=nothing) = prime(is,1,match)
prime(is::IndexSet, vargs...) = prime!(copy(is), vargs...)
# For is' notation
adjoint(is::IndexSet) = prime(is)

function setprime!(is::IndexSet, plev::Integer, match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = setprime(is[jj],plev)
  end
  return is
end
setprime(is::IndexSet, vargs...) = setprime!(copy(is), vargs...)

noprime!(is::IndexSet, match = nothing) = setprime!(is, 0, match)
noprime(is::IndexSet, vargs...) = noprime!(copy(is), vargs...)

function addtags!(is::IndexSet,
                  tags,
                  match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = addtags(is[jj],tags)
  end
  return is
end
addtags(is, vargs...) = addtags!(copy(is), vargs...)

function removetags!(is::IndexSet,
                     tags,
                     match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = removetags(is[jj],tags)
  end
  return is
end
removetags(is, vargs...) = removetags!(copy(is), vargs...)

function replacetags!(is::IndexSet,
                      tags_old, tags_new,
                      match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = replacetags(is[jj],tags_old,tags_new)
  end
  return is
end
replacetags(is, vargs...) = replacetags!(copy(is), vargs...)

function swaptags!(is::IndexSet,
                   tags1, tags2,
                   match = nothing)
  ts1 = TagSet(tags1)
  ts2 = TagSet(tags2)
  tstemp = TagSet("e43efds")
  plev(ts1) ≥ 0 && (tstemp = setprime(tstemp,431534))
  replacetags!(is, ts1, tstemp, match)
  replacetags!(is, ts2, ts1, match)
  replacetags!(is, tstemp, ts2, match)
  return is
end
swaptags(is, vargs...) = swaptags!(copy(is), vargs...)

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

