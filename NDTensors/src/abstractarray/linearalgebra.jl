# NDTensors.qr
qr(A::AbstractMatrix) = qr(leaf_parenttype(A), A)
qr(::Type{<:AbstractArray}, A::AbstractMatrix) = LinearAlgebra.qr(A)

# NDTensors.eigen
eigen(A::AbstractMatrix) = eigen(leaf_parenttype(A), A)
eigen(::Type{<:AbstractArray}, A::AbstractMatrix) = LinearAlgebra.eigen(A)

# NDTensors.svd
svd(A::AbstractMatrix) = svd(leaf_parenttype(A), A)
svd(::Type{<:AbstractArray}, A::AbstractMatrix) = LinearAlgebra.svd(A)
