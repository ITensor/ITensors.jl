function set_eltype(storagetype::Type{<:Diag}, eltype::Type)
  return set_datatype(storagetype, set_eltype(datatype(storagetype), eltype))
end

function set_datatype(storagetype::Type{<:Diag}, datatype::Type{<:AbstractVector})
  return Diag{eltype(datatype),datatype}
end

function set_eltype(storagetype::Type{<:UniformDiag}, eltype::Type)
  return Diag{eltype,eltype}
end

function set_datatype(storagetype::Type{<:NonuniformDiag}, datatype::Type)
  return Diag{eltype(datatype), datatype}
end
