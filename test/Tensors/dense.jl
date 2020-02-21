using ITensors,
      Test

@testset "DenseTensor basic functionality" begin

A = Tensor(3,4)

for I in eachindex(A)
  @test A[I] == 0
end

randn!(A)

for I in eachindex(A)
  @test A[I] != 0
end

for I in eachindex(A)
  @test A[I] != 0
end

@test ndims(A) == 2
@test dims(A) == (3,4)
@test inds(A) == (3,4)

A[1,1] = 11

@test A[1,1] == 11

Aview = A[2:3,2:3]

@test dims(Aview) == (2,2)
@test A[2,2] == Aview[1,1]

B = Tensor(undef,3,4)
randn!(B)

C = A+B

for I in eachindex(C)
  @test C[I] == A[I] + B[I]
end

Ap = permutedims(A,(2,1))

for I in eachindex(A)
  @test A[I] == Ap[permute(I,(2,1))]
end

t = Tensor(ComplexF64,100,100)
randn!(t)
@test conj(data(store(t))) == data(store(conj(t)))
@test typeof(conj(t)) <: DenseTensor

end
