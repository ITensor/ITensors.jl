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
  printTimes(timer)


end; main()
