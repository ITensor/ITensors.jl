using ITensors, Test

@testset "sweepnext function" begin
  @testset "one site" begin
    N = 6
    count = 1
    output = [
      (1, 1),
      (2, 1),
      (3, 1),
      (4, 1),
      (5, 1),
      (6, 1),
      (6, 2),
      (5, 2),
      (4, 2),
      (3, 2),
      (2, 2),
      (1, 2),
    ]
    for (b, ha) in sweepnext(N; ncenter=1)
      @test (b, ha) == output[count]
      count += 1
    end
    @test count == 2 * N + 1
  end

  @testset "two site" begin
    N = 6
    count = 1
    output = [
      (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2)
    ]
    for (b, ha) in sweepnext(N)
      @test (b, ha) == output[count]
      count += 1
    end
    @test count == 2 * (N - 1) + 1
  end

  @testset "three site" begin
    N = 6
    count = 1
    output = [(1, 1), (2, 1), (3, 1), (4, 1), (4, 2), (3, 2), (2, 2), (1, 2)]
    for (b, ha) in sweepnext(N; ncenter=3)
      @test (b, ha) == output[count]
      count += 1
    end
    @test count == 2 * (N - 2) + 1
  end
end

nothing
