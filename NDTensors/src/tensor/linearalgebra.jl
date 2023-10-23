function LinearAlgebra.qr(T::Tensor; kwargs...)
  return qr(T; kwargs...)
end

function LinearAlgebra.eigen(T::Tensor; kwargs...)
  return eigen(T; kwargs...)
end

function LinearAlgebra.eigen(T::Hermitian{<:Real,<:Tensor}; kwargs...)
  return eigen(T; kwargs...)
end

function LinearAlgebra.eigen(T::Hermitian{<:Complex{<:Real},<:Tensor}; kwargs...)
  return eigen(T; kwargs...)
end

function LinearAlgebra.svd(T::Tensor; kwargs...)
  return svd(T; kwargs...)
end
