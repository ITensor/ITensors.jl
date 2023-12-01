const DiagonalVector{T,Diag,Zero} = DiagonalArray{T,1,Diag,Zero}

function DiagonalVector(diag::AbstractVector)
  return DiagonalArray{<:Any,1}(diag)
end
