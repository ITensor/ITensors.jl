using ITensors
using Test
import Random: seed!

seed!(12345)

@testset "ITensor contraction sequence optimization" begin
  d = 100
  i = Index(d, "i")
  A = randomITensor(i', dag(i))

  @test !ITensors.using_contraction_sequence_optimization()

  A2 = A' * A
  @test hassameinds(A2, (i'', i))

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  @test A' * A ≈ A2

  ITensors.disable_contraction_sequence_optimization()
  @test !ITensors.using_contraction_sequence_optimization()

  A3 = A'' * A' * A
  @test hassameinds(A3, (i''', i))
  @test contract([A'', A', A]) ≈ A3
  @test contract([A'', A', A]; sequence = "automatic") ≈ A3
  @test contract([A'', A', A]; sequence = "left_associative") ≈ A3
  @test contract([A'', A', A]; sequence = "right_associative") ≈ A3
  @test contract([A'', A', A]; sequence = [[1, 2], 3]) ≈ A3
  @test contract([A'', A', A]; sequence = [[2, 3], 1]) ≈ A3
  # A bad sequence
  @test contract([A'', A', A]; sequence = [[1, 3], 2]) ≈ A3

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  @test A'' * A' * A ≈ A3
  @test A * A'' * A' ≈ A3
  @test contract([A'', A', A]) ≈ A3
  @test contract([A, A'', A']) ≈ A3
  @test contract([A'', A', A]; sequence = "automatic") ≈ A3
  @test contract([A'', A', A]; sequence = "left_associative") ≈ A3
  @test contract([A'', A', A]; sequence = "right_associative") ≈ A3
  @test contract([A'', A', A]; sequence = [[1, 2], 3]) ≈ A3
  @test contract([A'', A', A]; sequence = [[2, 3], 1]) ≈ A3
  @test contract([A'', A', A]; sequence = [[1, 3], 2]) ≈ A3

  ITensors.disable_contraction_sequence_optimization()
  @test !ITensors.using_contraction_sequence_optimization()

  # This is not the only sequence
  @test ITensors.optimal_contraction_sequence([A, A'', A']) == Any[1, Any[2, 3]]

  time_without_opt = @elapsed A * A'' * A'

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  time_with_opt = @elapsed A * A'' * A'

  @test time_with_opt < time_without_opt

  ITensors.disable_contraction_sequence_optimization()
  @test !ITensors.using_contraction_sequence_optimization()

  A4 = A''' * A'' * A' * A
  @test hassameinds(A4, (i'''', i))
  @test contract([A''', A'', A', A]; sequence = [[[1, 2], 3], 4]) ≈ A4

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  @test A'' * A * A''' * A' ≈ A4

  ITensors.disable_contraction_sequence_optimization()
  @test !ITensors.using_contraction_sequence_optimization()

  # This is not the only sequence
  @test ITensors.optimal_contraction_sequence([A, A'', A', A''']) == Any[Any[1, 3], Any[2, 4]]

  time_without_opt = @elapsed A * A'' * A' * A'''

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  time_with_opt = @elapsed A * A'' * A' * A'''

  @test time_with_opt < time_without_opt

  ITensors.disable_contraction_sequence_optimization()
  @test !ITensors.using_contraction_sequence_optimization()
end

