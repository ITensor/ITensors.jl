# Makes for generic code

@traitfn dim(ind::IndT) where {IndT; is_blocked{IndT}} = sum(ind)

#Base.ndims(ds::Type{<:BlockDims{N}}) where {N} = N
@traitfn Base.ndims(ds::IndsT) where {IndsT; is_blocked{IndsT}} = length(IndsT.parameters)

@traitfn Base.ndims(ds::Type{<:IndsT}) where {IndsT; is_blocked{IndsT}} = length(ds.parameters)


"""
dim(::BlockDims,::Integer)

Return the total extent of the specified dimensions.
"""
@traitfn function dim(ds::IndsT, i::Integer) where {IndsT; is_blocked{IndsT}}
  return sum(ds[i])
end

function dim(ds::IndsT, i::Integer) where{IndsT}
  return ds[i]
end

#dim(d::BlockDim) = sum(d)
#Base.ndims(ds::Type{<:BlockDims{N}}) where {N} = N