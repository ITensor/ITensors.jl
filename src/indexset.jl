export IndexSet,
       hasindex,
       hasinds,
       hassameinds,
       findindex,
       findinds,
       swaptags,
       swaptags!,
       swapprime,
       swapprime!,
       mapprime,
       mapprime!,
       commoninds,
       commonindex,
       uniqueinds,
       uniqueindex,
       dims,
       minDim,
       maxDim

struct IndexSet
    inds::Vector{Index}
    IndexSet(inds::Vector{Index}) = new(inds)
end

inds(is::IndexSet) = is.inds

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
Index(is::IndexSet) = length(is)==1 ? is[1] : error("Number of Index in IndexSet ≠ 1")

function Base.show(io::IO, is::IndexSet)
  for i in is.inds
    print(io,i)
    print(io," ")
  end
end

getindex(is::IndexSet,n::Integer) = getindex(is.inds,n)
setindex!(is::IndexSet,i::Index,n::Integer) = setindex!(is.inds,i,n)
lastindex(is :: IndexSet) = lastindex(is.inds)
length(is::IndexSet) = length(is.inds)
order(is::IndexSet) = length(is)
copy(is::IndexSet) = IndexSet(copy(is.inds))
dims(is::IndexSet) = Tuple(dim(i) for i ∈ is)
dim(is::IndexSet) = prod(dim.(is))
dim(is::IndexSet,pos::Integer) = dim(is[pos])

# TODO: what should size(::IndexSet) do?
#size(is::IndexSet) = size(is.inds)
#Base.size(is::IndexSet) = dims(is)
#Base.size(is::IndexSet,pos::Integer) = dim(is,pos)

# Optimize this (right own function that extracts dimensions
# with a function)
Base.strides(is::IndexSet) = Base.size_to_strides(1, dims(is)...)
Base.stride(is::IndexSet,k::Integer) = strides(is)[k]

dag(is::IndexSet) = IndexSet(dag.(is.inds))

# Allow iteration
iterate(is::IndexSet,state::Int=1) = iterate(is.inds,state)

push!(is::IndexSet,i::Index) = push!(is.inds,i)

"""
minDim(is::IndexSet)

Get the minimum dimension of the indices in the index set.

Returns 1 if the IndexSet is empty.
"""
function minDim(is::IndexSet)
  length(is) == 0 && (return 1)
  md = dim(is[1])
  for n ∈ 2:length(is)
    md = min(md,dim(is[n]))
  end
  return md
end

"""
maxDim(is::IndexSet)

Get the maximum dimension of the indices in the index set.

Returns 1 if the IndexSet is empty.
"""
function maxDim(is::IndexSet)
  length(is) == 0 && (return 1)
  md = dim(is[1])
  for n ∈ 2:length(is)
    md = max(md,dim(is[n]))
  end
  return md
end

# 
# Set operations
#

# inds has the index i
function hasindex(inds,i::Index)
  is = IndexSet(inds)
  for j ∈ is
    i==j && return true
  end
  return false
end

# Binds is subset of Ainds
function hasinds(Binds,Ainds)
  Ais = IndexSet(Ainds)
  for i ∈ Ais
    !hasindex(Binds,i) && return false
  end
  return true
end
hasinds(Binds,Ainds::Index...) = hasinds(Binds,IndexSet(Ainds...))

# Set equality (order independent)
function hassameinds(Ainds,Binds)
  Ais = IndexSet(Ainds)
  Bis = IndexSet(Binds)
  return hasinds(Ais,Bis) && length(Ais) == length(Bis)
end

"""
==(is1::IndexSet, is2::IndexSet)

IndexSet quality (order dependent)
"""
function ==(Ais::IndexSet,Bis::IndexSet)
  length(Ais) ≠ length(Bis) && return false
  for i ∈ 1:length(Ais)
    Ais[i] ≠ Bis[i] && return false
  end
  return true
end

# Helper function for uniqueinds
# Return true if the Index is not in any
# of the input sets of indices
function _is_unique_index(j::Index,inds::T) where {T<:Tuple}
  for I in inds
    hasindex(I,j) && return false
  end
  return true
end
# Version taking one ITensor or IndexSet
function _is_unique_index(j::Index,inds)
  hasindex(inds,j) && return false
  return true
end


"""
uniqueinds(Ais,Bis...)

Output the IndexSet with Indices in Ais but not in
the IndexSets Bis.
"""
function uniqueinds(Ainds,Binds)
  Ais = IndexSet(Ainds)
  Cis = IndexSet()
  for j ∈ Ais
    _is_unique_index(j,Binds) && push!(Cis,j)
  end
  return Cis
end

"""
uniqueindex(Ais,Bis)

Output the Index in Ais but not in the IndexSets Bis.
Otherwise, return a default constructed Index.

In the future, this may throw an error if more than 
one Index is found.
"""
function uniqueindex(Ainds,Binds)
  Ais = IndexSet(Ainds)
  for j ∈ Ais
    _is_unique_index(j,Binds) && return j
  end
  return Index()
end
# This version can check for repeats, but is a bit
# slower because of IndexSet allocation
#uniqueindex(Ais,Bis) = Index(uniqueinds(Ais,Bis)) 

setdiff(Ais::IndexSet,Bis) = uniqueinds(Ais,Bis)

"""
commoninds(Ais,Bis)

Output the IndexSet in the intersection of Ais and Bis
"""
function commoninds(Ainds,Binds)
  Ais = IndexSet(Ainds)
  Cis = IndexSet()
  for i ∈ Ais
    hasindex(Binds,i) && push!(Cis,i)
  end
  return Cis
end

"""
commonindex(Ais,Bis)

Output the Index common to Ais and Bis.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
function commonindex(Ainds,Binds)
  Ais = IndexSet(Ainds)
  for i ∈ Ais
    hasindex(Binds,i) && return i
  end
  return Index()
end
# This version checks if there are more than one indices
#commonindex(Ais,Bis) = Index(commoninds(Ais,Bis))

"""
findinds(inds,tags)

Output the IndexSet containing the subset of indices
of inds containing the tags in the input tagset.
"""
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
"""
findindex(inds,tags)

Output the Index containing the tags in the input tagset.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
function findindex(inds,tags)
  is = IndexSet(inds)
  ts = TagSet(tags)
  for i ∈ is
    if hastags(i,ts)
      return i
    end
  end
  return Index()
end
# This version checks if there are more than one indices
#findindex(inds, tags) = Index(findinds(inds,tags))

function findindex(is::IndexSet,
                   i::Index)::Int
  for (n,j) in enumerate(is)
    if i==j
      return n
    end
  end
  return 0
end

# From a tag set or index set, find the positions
# of the matching indices as a vector of integers
indexpositions(inds) = collect(1:length(inds))
indexpositions(inds, match::Nothing) = collect(1:length(inds))
#indexpositions(inds, match::Tuple{}) = collect(1:length(inds))
# Version for matching a tag set
function indexpositions(inds, match::T) where {T<:Union{AbstractString,
                                                        Tuple{<:AbstractString,<:Integer},
                                                        TagSet}}
  is = IndexSet(inds)
  tsmatch = TagSet(match)
  pos = Int[]
  for (j,I) ∈ enumerate(is)
    hastags(I,tsmatch) && push!(pos,j)
  end
  return pos
end
# Version for matching a collection of indices
function indexpositions(inds, match)
  is = IndexSet(inds)
  ismatch = IndexSet(match)
  pos = Int[]
  for (j,I) ∈ enumerate(is)
    hasindex(ismatch,I) && push!(pos,j)
  end
  return pos
end
# Version for matching a list of indices
indexpositions(inds, match_inds::Index...) = indexpositions(inds, IndexSet(match_inds...))

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
prime!(is::IndexSet,match=nothing) = prime!(is,1,match)
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

function swapprime!(is::IndexSet, 
                    pl1::Int,
                    pl2::Int,
                    vargs...) 
  pos = indexpositions(is,vargs...)
  for n in pos
    if plev(is[n])==pl1
      is[n] = setprime(is[n],pl2)
    end
  end
  return is
end

swapprime(is::IndexSet,pl1::Int,pl2::Int,vargs...) = swapprime!(copy(is),pl1,pl2,vargs...)

function mapprime!(is::IndexSet,
                   plold::Integer,
                   plnew::Integer,
                   match = nothing)
  pos = indexpositions(is,match)
  for n in pos
    if plev(is[n])==plold 
      is[n] = setprime(is[n],plnew)
    end
  end
  return is
end

function mapprime(is::IndexSet,
                  plold::Integer,
                  plnew::Integer,
                  match=nothing)
  return mapprime!(copy(is),plold,plnew,match)
end


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

function settags!(is::IndexSet,
                  ts,
                  match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = settags(is[jj],ts)
  end
  return is
end
settags(is, vargs...) = settags!(copy(is), vargs...)

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

