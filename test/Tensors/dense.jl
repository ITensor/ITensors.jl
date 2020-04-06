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

@test A * 2.0 == 2.0 * A

Asim = similar(data(A), 10)
@test eltype(Asim) == Float64
@test length(Asim) == 10

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


@test Dense(ComplexF64) == Dense{ComplexF64}()
@test Dense(ComplexF64) == complex(Dense(Float64))

D = Tensor(ComplexF64,(100,100))
@test eltype(D) == ComplexF64
@test ndims(D) == 2
@test dim(D) == 100^2

E = Tensor(ComplexF64,undef, (100,100))
@test eltype(E) == ComplexF64
@test ndims(E) == 2
@test dim(E) == 100^2

F = Tensor((100,100))
@test eltype(F) == Float64
@test ndims(F) == 2
@test dim(F) == 100^2

G = Tensor(undef, (100,100))
@test eltype(G) == Float64
@test ndims(G) == 2
@test dim(G) == 100^2

H = Tensor(ComplexF64,undef, 100,100)
@test eltype(H) == ComplexF64
@test ndims(H) == 2
@test dim(H) == 100^2

I_arr = rand(10,10,10)
I = Tensor(I_arr, (10,10,10))
@test eltype(I) == Float64
@test dim(I) == 1000
@test Array(I) == I_arr

i = Index(2,"i")
j = Index(2,"j")
k = Index(2,"k")
J = randomITensor(i, j)
K = randomITensor(j, k)
@test Array(tensor(J) * tensor(K)) ≈ Array(J*K, i, k) 
end
