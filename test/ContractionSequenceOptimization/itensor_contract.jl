using ITensors
using Test
import Random: seed!

seed!(12345)

using ITensors.ContractionSequenceOptimization: optimal_contraction_sequence, deepmap

@testset "ITensor contraction sequence optimization" begin
  d = 100
  i = Index(d, "i")
  A = randomITensor(i', dag(i))

  # Low level functions
  @test dim([1, 2], [4, 5, 6]) == 4 * 5
  @test dim(Int[], [4, 5, 6]) == 1

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
  @test contract([A'', A', A]; sequence="automatic") ≈ A3
  @test contract([A'', A', A]; sequence="left_associative") ≈ A3
  @test contract([A'', A', A]; sequence="right_associative") ≈ A3
  @test contract([A'', A', A]; sequence=[[1, 2], 3]) ≈ A3
  @test contract([A'', A', A]; sequence=[[2, 3], 1]) ≈ A3
  # A bad sequence
  @test contract([A'', A', A]; sequence=[[1, 3], 2]) ≈ A3

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  @test A'' * A' * A ≈ A3
  @test A * A'' * A' ≈ A3
  @test contract([A'', A', A]) ≈ A3
  @test contract([A, A'', A']) ≈ A3
  @test contract([A'', A', A]; sequence="automatic") ≈ A3
  @test contract([A'', A', A]; sequence="left_associative") ≈ A3
  @test contract([A'', A', A]; sequence="right_associative") ≈ A3
  @test contract([A'', A', A]; sequence=[[1, 2], 3]) ≈ A3
  @test contract([A'', A', A]; sequence=[[2, 3], 1]) ≈ A3
  @test contract([A'', A', A]; sequence=[[1, 3], 2]) ≈ A3

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
  @test contract([A''', A'', A', A]; sequence=[[[1, 2], 3], 4]) ≈ A4

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  @test A'' * A * A''' * A' ≈ A4

  ITensors.disable_contraction_sequence_optimization()
  @test !ITensors.using_contraction_sequence_optimization()

  # This is not the only sequence
  @test ITensors.optimal_contraction_sequence([A, A'', A', A''']) ==
    Any[Any[1, 3], Any[2, 4]]

  time_without_opt = @elapsed A * A'' * A' * A'''

  ITensors.enable_contraction_sequence_optimization()
  @test ITensors.using_contraction_sequence_optimization()

  time_with_opt = @elapsed A * A'' * A' * A'''

  @test time_with_opt < time_without_opt

  ITensors.disable_contraction_sequence_optimization()
  @test !ITensors.using_contraction_sequence_optimization()
end

@testset "contract sequence optimization interfaces" begin
  # Network and dimensions need to be large enough
  # so that tensor allocations dominate over network
  # analysis for testing the number of allocations below.
  d0 = 2
  δd = 1000
  ntensors = 6
  ElType = Float64
  d = [d0 + (n - 1) * δd for n in 1:ntensors]
  t = ["$n" for n in 1:ntensors]
  is = Index.(d, t)

  As = [randomITensor(ElType, is[n], is[mod1(n + 1, ntensors)]) for n in 1:ntensors]

  # Warmup
  contract(As)
  allocations_left_associative = @allocated contract(As)

  allocations_left_associative_pairwise = 0
  tmp = As[1]
  for n in 2:length(As)
    tmp * As[n]
    allocations_left_associative_pairwise += @allocated tmp = tmp * As[n]
  end
  @test allocations_left_associative ≈ allocations_left_associative_pairwise rtol = 0.01

  sequence = foldr((x, y) -> [x, y], 1:ntensors)
  @test sequence == optimal_contraction_sequence(As)
  As_network = foldr((x, y) -> [x, y], As)
  @test deepmap(n -> As[n], sequence) == As_network

  # Warmup
  contract(As; sequence=sequence)
  contract(As; sequence="right_associative")
  contract(As; sequence="automatic")
  contract(As_network)

  # Measure allocations of different interfaces
  allocations_right_associative_1 = @allocated contract(As; sequence=sequence)
  allocations_right_associative_2 = @allocated contract(As; sequence="right_associative")
  allocations_right_associative_3 = @allocated contract(As; sequence="automatic")
  allocations_right_associative_4 = @allocated contract(As_network)

  allocations_right_associative_pairwise = 0
  tmp = As[end]
  for n in reverse(1:(length(As) - 1))
    tmp * As[n]
    allocations_right_associative_pairwise += @allocated tmp = tmp * As[n]
  end
  @test allocations_right_associative_pairwise ≈ allocations_right_associative_1 rtol = 0.1
  @test allocations_right_associative_pairwise ≈ allocations_right_associative_2 rtol = 0.1
  @test allocations_right_associative_pairwise ≈ allocations_right_associative_3 rtol = 0.2
  @test allocations_right_associative_pairwise ≈ allocations_right_associative_4 rtol = 0.1

  @test allocations_right_associative_1 < allocations_left_associative
  @test allocations_right_associative_2 < allocations_left_associative
  @test allocations_right_associative_3 < allocations_left_associative
  @test allocations_right_associative_4 < allocations_left_associative
end
