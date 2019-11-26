using ITensors,
      Test

@testset "DenseTensor basic functionality" begin

indsA = (3,4)

A = DenseTensor(indsA)
randn!(A)

@test ndims(A) == 2
@test dims(A) == (3,4)
@test inds(A) == (3,4)

A[1,1] = 11

@test A[1,1] == 11

Aview = A[CartesianIndex(2,2):CartesianIndex(3,3)]

@test dims(Aview) == (2,2)
@test A[2,2] == Aview[1,1]

B = DenseTensor(indsA)
randn!(B)

C = A+B

@test C[1,2] == A[1,2]+B[1,2]

Ap = permutedims(A,(2,1))

@test A[3,4]==Ap[4,3]

end
