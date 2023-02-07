# NDTensors.similar
function similar(storagetype::Type{<:Diag}, dims::Dims)
  return setdata(storagetype, similar(datatype(storagetype), mindim(dims)))
end

# TODO: Redesign UniformDiag to make it handled better
# by generic code.
function similartype(storagetype::Type{<:UniformDiag}, eltype::Type)
  # This will also set the `datatype`.
  return set_eltype(storagetype, eltype)
end

# Needed to get slice of DiagTensor like T[1:3,1:3]
function similar(
  T::DiagTensor{<:Number,N}, ::Type{ElR}, inds::Dims{N}
) where {ElR<:Number,N}
  return tensor(similar(storage(T), ElR, minimum(inds)), inds)
end

similar(storage::NonuniformDiag) = setdata(storage, similar(data(storage)))

similar(D::UniformDiag{ElT}) where {ElT} = Diag(zero(ElT))
similar(D::UniformDiag, inds) = similar(D)
similar(::Type{<:UniformDiag{ElT}}, inds) where {ElT} = Diag(zero(ElT))

similar(D::Diag, n::Int) = Diag(similar(data(D), n))

similar(D::Diag, ::Type{ElR}, n::Int) where {ElR} = Diag(similar(data(D), ElR, n))
