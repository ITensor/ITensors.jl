using ITensors
using Test

@testset "Test debug checks on IndexSet construction" begin
  i = Index(2, "i")

  @test !ITensors.use_debug_checks()
  # Test that no error is thrown in constructor
  IndexSet(i, i)
  IndexSet(i, i')

  # Turn on debug checks
  ITensors.enable_debug_checks()
  @test ITensors.use_debug_checks()
  @test_throws ErrorException IndexSet(i, i)
  # Test that no error is thrown in constructor
  IndexSet(i, i')

  # Turn off debug checks
  ITensors.disable_debug_checks()
  @test !ITensors.use_debug_checks()
  # Test that no error is thrown in constructor
  IndexSet(i, i)
  IndexSet(i, i')
end

