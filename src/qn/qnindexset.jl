
const QNIndexSet{N} = IndexSet{N, QNIndex, NTuple{N, QNIndex}}

const QNIndices{N} = Union{QNIndexSet{N},
                           NTuple{N, QNIndex}}

# Get a list of the non-zero blocks given a desired flux
# TODO: make a fillqns(inds::IndexSet) function that makes all indices
# in inds have the same qns. Then, use a faster comparison:
#   ==(flux(inds,block; assume_filled=true), qn; assume_filled=true)
function nzblocks(qn::QN, inds::IndexSet{N}) where {N}
  blocks = Block{N}[]
  for block in eachblock(inds)
    if flux(inds, block) == qn
      push!(blocks, block)
    end
  end
  return blocks
end

function nzdiagblocks(qn::QN,
                      inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachdiagblock(inds)
    if flux(inds,block) == qn
      push!(blocks,Tuple(block))
    end
  end
  return blocks
end

removeqns(is::QNIndexSet) = map(i -> removeqns(i), is)

anyfermionic(is::IndexSet) = any(isfermionic, is)

allfermionic(is::IndexSet) = all(isfermionic, is)

function Base.show(io::IO, is::QNIndexSet)
  print(io,"IndexSet{$(length(is))} ")
  for i in is
    print(io, i)
    println(io)
  end
end

