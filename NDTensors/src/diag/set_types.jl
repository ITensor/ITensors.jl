function set_eltype(storagetype::Type{<:UniformDiag}, eltype::Type)
  return Diag{eltype,eltype}
end

function set_eltype(storagetype::Type{<:NonuniformDiag}, eltype::Type{<:AbstractArray})
  return Diag{eltype,similartype(storagetype, eltype)}
end

function set_datatype(storagetype::Type{<:UniformDiag}, datatype::Type)
  return Diag{datatype,datatype}
end

function set_datatype(storagetype::Type{<:NonuniformDiag}, datatype::Type{<:AbstractArray})
  return Diag{eltype(datatype),datatype}
end
