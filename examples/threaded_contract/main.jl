using BenchmarkTools
using ITensors
using ITensors.Strided
using LinearAlgebra

function main(; d = 2, order = 4)
  BLAS.set_num_threads(1)
  Strided.set_num_threads(1)

  println("#################################################")
  println("# order = ", order)
  println("# d = ", d)
  println("#################################################")
  println()

  i = Index(QN(0, 2) => d, QN(1, 2) => d)
  is = IndexSet(n -> settags(i, "i$n"), order ÷ 2)
  A = randomITensor(is'..., dag(is)...)
  B = randomITensor(is'..., dag(is)...)

  ITensors.disable_threaded_blocksparse()

  println("Serial contract:")
  @disable_warn_order begin
    C_contract = @btime $A' * $B samples = 5
  end
  println()

  println("Threaded contract:")
  @disable_warn_order begin
    ITensors.enable_threaded_blocksparse()
    C_threaded_contract = @btime $A' * $B samples = 5
    ITensors.disable_threaded_blocksparse()
  end
  println()
  @show C_contract ≈ C_threaded_contract
  return nothing
end

