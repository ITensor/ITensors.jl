using ITensors,
      Test

@testset "BlockSparseTensor basic functionality" begin

# Indices
i1 = [2,3]
i2 = [4,5]

indsA = (i1,i2)

# Locations of non-zero blocks
locs = [(1,2),(2,1)]

A = BlockSparseTensor(locs,indsA)
randn!(A)

@test nnzblocks(A) == 2
@test nnz(A) == 2*5+3*4
@test inds(A) == indsA

A[1,5] = 15
A[2,5] = 25

@test A[1,1] == 0
@test A[1,5] == 15
@test A[2,5] == 25

D = dense(A)

@test D == A

A12 = blockview(A,(1,2))

@test dims(A12) == (2,5)
@test A12[1,1] == A[1,5]

B = BlockSparseTensor(locs,indsA)
randn!(B)

C = A+B

@test C[3,6] == A[3,6]+B[3,6]

Ap = permutedims(A,(2,1))

@test A[5,3]==Ap[3,5]

end

