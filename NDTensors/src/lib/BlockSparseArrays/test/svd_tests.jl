using Test
using NDTensors.BlockSparseArrays
using NDTensors.BlockSparseArrays: BlockSparseArray, svd, tsvd, notrunc, truncbelow, truncdim, BlockDiagonal
using BlockArrays
using LinearAlgebra: LinearAlgebra, Diagonal, svdvals

function test_svd(a, usv)
  U, S, V = usv

  @test U * Diagonal(S) * V' ≈ a
  @test U' * U ≈ LinearAlgebra.I
  @test V' * V ≈ LinearAlgebra.I
end

# regular matrix
# --------------
sizes = ((3, 3), (4, 3), (3, 4))
eltypes = (Float32, Float64, ComplexF64)
@testset "($m, $n) Matrix{$T}" for ((m, n), T) in Iterators.product(sizes, eltypes)
  a = rand(3, 3)
  usv = @inferred svd(a)
  test_svd(a, usv)

  # TODO: type unstable?
  usv2 = tsvd(a)
  test_svd(a, usv2)

  usv3 = tsvd(a; trunc=truncdim(2))
  @test length(usv3.S) == 2
  @test usv3.U' * usv3.U ≈ LinearAlgebra.I
  @test usv3.Vt * usv3.V ≈ LinearAlgebra.I

  s = usv3.S[end]
  usv4 = tsvd(a; trunc=truncbelow(s))
  @test length(usv4.S) == 2
  @test usv4.U' * usv4.U ≈ LinearAlgebra.I
  @test usv4.Vt * usv4.V ≈ LinearAlgebra.I
end

# block matrix
# ------------
blockszs = (([2, 2], [2, 2]), ([2, 2], [2, 3]), ([2, 2, 1], [2, 3]), ([2, 3], [2]))
@testset "($m, $n) BlockMatrix{$T}" for ((m, n), T) in Iterators.product(blockszs, eltypes)
  a = mortar([rand(T, i, j) for i in m, j in n])
  usv = svd(a)
  test_svd(a, usv)
  @test usv.U isa BlockedMatrix
  @test usv.Vt isa BlockedMatrix
  @test usv.S isa BlockedVector

  usv2 = tsvd(a)
  test_svd(a, usv2)
  @test usv.U isa BlockedMatrix
  @test usv.Vt isa BlockedMatrix
  @test usv.S isa BlockedVector

  usv3 = tsvd(a; trunc=truncdim(2))
  @test length(usv3.S) == 2
  @test usv3.U' * usv3.U ≈ LinearAlgebra.I
  @test usv3.Vt * usv3.V ≈ LinearAlgebra.I
  @test usv.U isa BlockedMatrix
  @test usv.Vt isa BlockedMatrix
  @test usv.S isa BlockedVector

  s = usv3.S[end]
  usv4 = tsvd(a; trunc=truncbelow(s))
  @test length(usv4.S) == 2
  @test usv4.U' * usv4.U ≈ LinearAlgebra.I
  @test usv4.Vt * usv4.V ≈ LinearAlgebra.I
  @test usv.U isa BlockedMatrix
  @test usv.Vt isa BlockedMatrix
  @test usv.S isa BlockedVector
end

# Block-Diagonal matrices
# -----------------------
@testset "($m, $n) BlockDiagonal{$T}" for ((m, n), T) in Iterators.product(blockszs, eltypes)
  a = BlockDiagonal([rand(T, i, j) for (i, j) in zip(m, n)])
  usv = svd(a)
  test_svd(a, usv)
  @test usv.U isa BlockDiagonal
  @test usv.Vt isa BlockDiagonal
  @test usv.S isa BlockVector

  usv2 = tsvd(a)
  test_svd(a, usv2)
  @test usv.U isa BlockDiagonal
  @test usv.Vt isa BlockDiagonal
  @test usv.S isa BlockVector

  usv3 = tsvd(a; trunc=truncdim(2))
  @test length(usv3.S) == 2
  @test usv3.U' * usv3.U ≈ LinearAlgebra.I
  @test usv3.Vt * usv3.V ≈ LinearAlgebra.I
  @test usv.U isa BlockDiagonal
  @test usv.Vt isa BlockDiagonal
  @test usv.S isa BlockVector

  @show s = usv3.S[end]
  usv4 = tsvd(a; trunc=truncbelow(s))
  @test length(usv4.S) == 2
  @test usv4.U' * usv4.U ≈ LinearAlgebra.I
  @test usv4.Vt * usv4.V ≈ LinearAlgebra.I
  @test usv.U isa BlockDiagonal
  @test usv.Vt isa BlockDiagonal
  @test usv.S isa BlockVector
end


a = mortar([rand(2, 2) for i in 1:2, j in 1:3])
usv = svd(a)
test_svd(a, usv)

a = mortar([rand(2, 2) for i in 1:3, j in 1:2])
usv = svd(a)
test_svd(a, usv)

# blocksparse 
# -----------
a = BlockSparseArray([Block(2, 1), Block(1, 2)], [rand(2, 2), rand(2, 2)], (blockedrange([2, 2]), blockedrange([2, 2])))
usv = svd(a)
test_svd(a, usv)


using NDTensors.BlockSparseArrays: block_stored_indices