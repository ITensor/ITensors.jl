# Makes for generic code


#Base.ndims(ds::Type{<:BlockDims{N}}) where {N} = N
@traitfn Base.ndims(ds::IndsT) where {IndsT; is_blocked{IndsT}} = length(IndsT.parameters)

@traitfn Base.ndims(ds::Type{<:IndsT}) where {IndsT; is_blocked{IndsT}} = length(ds.parameters)
