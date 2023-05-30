function set_datatype(storagetype::Type{<:BlockSparse}, datatype::Type{<:AbstractVector})
  return BlockSparse{eltype(datatype),datatype,ndims(storagetype)}
end

function promote_rule(
  ::Type{<:BlockSparse{ElT1,VecT1,N}}, ::Type{<:BlockSparse{ElT2,VecT2,N}}
) where {ElT1,ElT2,VecT1,VecT2,N}
  return BlockSparse{promote_type(ElT1, ElT2),promote_type(VecT1, VecT2),N}
end

function promote_rule(
  ::Type{<:BlockSparse{ElT1,VecT1,N1}}, ::Type{<:BlockSparse{ElT2,VecT2,N2}}
) where {ElT1,ElT2,VecT1,VecT2,N1,N2}
  return BlockSparse{promote_type(ElT1, ElT2),promote_type(VecT1, VecT2),NR} where {NR}
end

function promote_rule(
  ::Type{<:BlockSparse{ElT1,Vector{ElT1},N1}}, ::Type{ElT2}
) where {ElT1,ElT2<:Number,N1}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return BlockSparse{ElR,VecR,N1}
end

function convert(
  ::Type{<:BlockSparse{ElR,VecR,N}}, D::BlockSparse{ElD,VecD,N}
) where {ElR,VecR,N,ElD,VecD}
  return setdata(D, convert(VecR, data(D)))
end
