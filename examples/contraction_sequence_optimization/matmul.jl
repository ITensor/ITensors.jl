using ITensors
using ProfileView

# Examples of optimizing simple chained matrix multiplications,
# useful for getting an idea for the overhead.
function main(d=100)
  i = Index(d, "i")
  A = randomITensor(i', dag(i))

  #
  # 2 tensors
  #

  println("\n2 tensors")

  # Simple pair contraction
  @btime $A' * $A

  #
  # 3 tensors
  #

  println("\n3 tensors")

  println("Pick an optimal sequence explicitly")

  ITensors.disable_contraction_sequence_optimization()
  @show ITensors.using_contraction_sequence_optimization()

  @btime $A'' * $A' * $A
  # Alternative syntax
  #@btime $A'' * ($A' * $A)
  #@btime contract([$A'', $A', $A]; sequence = "right_associative")
  #@btime contract([$A'', $A', $A]; sequence = "left_associative")
  #@btime contract([$A'', $A', $A]; sequence = $([[1, 2], 3]))
  #@btime contract([$A'', $A', $A]; sequence = $([[2, 3], 1]))

  println("Pick a bad sequence explicitly")

  @btime $A'' * $A * $A'
  # Alternative syntax
  #@btime contract([$A'', $A', $A]; sequence = $([[1, 3], 2]))

  println("Let it optimize")

  ITensors.enable_contraction_sequence_optimization()
  @show ITensors.using_contraction_sequence_optimization()

  # Already starting from an optimal sequence, there
  # is still overhead to checking if there isn't a better sequence
  println("Starting from optimal sequence")
  @btime $A'' * $A' * $A
  # Find a better sequence
  println("Starting from non-optimal sequence")
  @btime $A'' * $A * $A'

  f(A, N) =
    for _ in 1:N
      A'' * A' * A
    end
  @profview f(A, 1e6)

  ITensors.disable_contraction_sequence_optimization()

  #
  # 4 tensors
  #

  println("\n4 tensors")

  println("Pick an optimal sequence explicitly")

  ITensors.disable_contraction_sequence_optimization()
  @show ITensors.using_contraction_sequence_optimization()

  @btime $A''' * $A'' * $A' * $A
  # Alternative syntax
  #@btime $A''' * ($A'' * ($A' * $A))
  #@btime contract([$A''', $A'', $A', $A]; sequence = $([[[1, 2], 3], 4]))

  println("Pick a bad sequence explicitly")

  @btime $A'' * $A * $A''' * $A'

  println("Let it optimize")

  ITensors.enable_contraction_sequence_optimization()
  @show ITensors.using_contraction_sequence_optimization()

  println("Starting from optimal sequence")
  @btime $A''' * $A'' * $A' * $A
  println("Starting from non-optimal sequence")
  @btime $A'' * $A * $A''' * $A'

  #
  # 5 tensors
  #

  println("\n5 tensors")

  println("Pick an optimal sequence explicitly")

  ITensors.disable_contraction_sequence_optimization()
  @show ITensors.using_contraction_sequence_optimization()

  @btime $A'''' * $A''' * $A'' * $A' * $A

  println("Pick a bad sequence explicitly")

  @btime $A'' * $A * $A''' * $A' * $A''''

  println("Let it optimize")

  ITensors.enable_contraction_sequence_optimization()
  @show ITensors.using_contraction_sequence_optimization()

  println("Starting from optimal sequence")
  @btime $A'''' * $A''' * $A'' * $A' * $A
  println("Starting from non-optimal sequence")
  @btime $A'' * $A'''' * $A * $A''' * $A'

  return ITensors.disable_contraction_sequence_optimization()
end
