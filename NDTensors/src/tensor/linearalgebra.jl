function LinearAlgebra.qr(T::Tensor; kwargs...)
  return qr(T; kwargs...)
end

function LinearAlgebra.svd(T::Tensor; kwargs...)
  return svd(T; kwargs...)
end
