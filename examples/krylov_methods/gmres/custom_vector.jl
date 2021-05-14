using ITensors
using KrylovKit
using LinearAlgebra

struct ITensorVector
  A::ITensor
end

# Overloads for ITensorVector
LinearAlgebra.norm(v::ITensorVector) = norm(v.A)
LinearAlgebra.dot(x::ITensorVector, y::ITensorVector) = dot(x.A, y.A)
Base.:*(a::Number, v::ITensorVector) = ITensorVector(a * v.A)
LinearAlgebra.axpy!(a::Number, x::ITensorVector, y::ITensorVector) = (axpy!(a, x.A, y.A); y)
Base.similar(v::ITensorVector) = ITensorVector(similar(v.A))
LinearAlgebra.mul!(x::ITensorVector, y::ITensorVector, a::Number) = (mul!(x.A, y.A, a); x)
LinearAlgebra.rmul!(v::ITensorVector, a::Number) = (rmul!(v.A, a); v)

# Overloads for ITensorMatrix
struct ITensorMatrix
  A::ITensor
end
(A::ITensorMatrix)(v::ITensorVector) = ITensorVector(noprime(A.A * v.A))

i = Index(4, "i")
A = ITensorMatrix(randomITensor(i', dag(i)))
b = ITensorVector(randomITensor(i))
x0 = ITensorVector(randomITensor(i))

x, _ = linsolve(A, b, x0)
err = A(x)
axpy!(-1, b, err)
@show norm(err)
