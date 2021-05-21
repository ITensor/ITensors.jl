using ITensors
using Test

@testset "ITensors.minimal_swap_range" begin
  @test ITensors.minimal_swap_range([1, 2], [1, 2]) == 1:2
  @test ITensors.minimal_swap_range([1, 2], [3, 4]) == 1:2
  @test ITensors.minimal_swap_range([1, 3], [5, 6]) == 2:3
  @test ITensors.minimal_swap_range([1, 4], [5, 6]) == 3:4
  @test ITensors.minimal_swap_range([1, 5], [5, 6]) == 4:5
  @test ITensors.minimal_swap_range([1, 6], [5, 6]) == 5:6
  @test ITensors.minimal_swap_range([1, 6], [4, 5, 6]) == 4:5
  @test ITensors.minimal_swap_range([1, 3, 5], [7, 8]) == 3:5
  @test ITensors.minimal_swap_range([1, 5], [2, 3]) == 2:3
  @test ITensors.minimal_swap_range([1, 5], [2, 4]) == 2:3
  @test ITensors.minimal_swap_range([1, 3, 5], [2, 4]) == 2:4
  @test ITensors.minimal_swap_range([1, 3, 5], [5, 8]) == 3:5
  @test ITensors.minimal_swap_range([3, 4], [1, 2]) == 3:4
  @test ITensors.minimal_swap_range([4, 6], [1, 2]) == 4:5
  @test ITensors.minimal_swap_range([4, 6], [1, 3]) == 4:5
  @test ITensors.minimal_swap_range([4, 6], [1, 4]) == 4:5
  @test ITensors.minimal_swap_range([2, 6], [1, 6]) == 2:3
  @test ITensors.minimal_swap_range([2, 6], [1, 7]) == 2:3
  @test ITensors.minimal_swap_range([2, 6], [1, 5]) == 3:4
  @test ITensors.minimal_swap_range([2, 5, 6], [1, 5]) == 3:5
  @test ITensors.minimal_swap_range([1, 5], [2, 6]) == 3:4
  @test ITensors.minimal_swap_range([4, 8], [3, 8]) == 4:5
  @test ITensors.minimal_swap_range([4, 6, 8], [3, 5]) == 5:7
  @test ITensors.minimal_swap_range([4, 6, 8], [3, 4]) == 4:6
  @test ITensors.minimal_swap_range([4, 6, 8], [3, 8]) == 4:6
end

