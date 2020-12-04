using BenchmarkTools
using ITensors

function main(; Nmax = 8)
  # Don't warn about large tensor orders
  disable_warn_order!()

  for N in 1:Nmax
    println("#################################################")
    println("# order = ", 2 * N)
    println("#################################################")
    println()

    d = 1
    i = Index(QN(0) => d, QN(1) => d)
    is = IndexSet(ntuple(n -> settags(i, "i$n"), Val(N)))
    A = randomITensor(is'..., dag(is)...)
    B = randomITensor(is'..., dag(is)...)

    # Use standard algorithm, without combining first
    println("Contract:")
    C_contract = @btime $A' * $B samples = 5
    println()

    # Reshape the ITensors into matrices before contracting
    println("Combine then contract:")
    enable_combine_contract!()
    C_combine_contract = @btime $A' * $B samples = 5
    disable_combine_contract!()
    println()

    @show C_contract ≈ C_combine_contract
    println()
  end
end

