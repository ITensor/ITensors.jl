const DiagonalMatrix{T,Diag,Zero} = DiagonalArray{T,2,Diag,Zero}

function DiagonalMatrix(diag::AbstractVector)
  return DiagonalArray{<:Any,2}(diag)
end
