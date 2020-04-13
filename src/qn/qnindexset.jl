
const QNIndexSet{N} = IndexSet{N,QNIndex}

# Get a list of the non-zero blocks given a desired flux
# TODO: make a fillqns(inds::IndexSet) function that makes all indices
# in inds have the same qns. Then, use a faster comparison:
#   ==(flux(inds,block; assume_filled=true), qn; assume_filled=true)
function NDTensors.nzblocks(qn::QN,
                            inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachblock(inds)
    if flux(inds,block) == qn
      push!(blocks,Tuple(block))
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

