using BenchmarkTools
using ITensors
using LinearAlgebra

function main(; d=20, order=4)
  BLAS.set_num_threads(1)
  ITensors.Strided.disable_threads()

  println("#################################################")
  println("# order = ", order)
  println("# d = ", d)
  println("#################################################")
  println()

  @show Threads.nthreads()
  @show Sys.CPU_THREADS
  @show BLAS.get_num_threads()
  @show ITensors.Strided.get_num_threads()
  println()

  i(n) = Index(QN(0) => d, QN(1) => d; tags="i$n")
  is = IndexSet(i, order ÷ 2)
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
