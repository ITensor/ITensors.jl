using ITensors, Printf

function main()
  i = Index(100,"i");

  A = randomITensor(i);
  B = randomITensor(i);

  # Dry run (JIT)
  time_dry = @elapsed begin
    R = A*B
  end
  reset!(timer)
  # End dry run

  time = @elapsed begin
    for n =1:10
    R = A*B
  end
  end

  @printf "time = %.12f\n" time
  @printf "contract_t = %.12f\n" timer.contract_t
  @printf "  gemm_t = %.12f (%d)\n" timer.gemm_t timer.gemm_c
  @printf "  permute_t = %.12f\n" timer.permute_t


end; main()
