using ITensors, Printf

function main()
  s1 = Index(2,"s1,Site");
  s2 = Index(2,"s2,Site");
  h1 = Index(10,"h1,Link,H");
  h2 = Index(10,"h2,Link,H");
  h3 = Index(10,"h3,Link,H");
  a1 = Index(100,"a1,Link");
  a3 = Index(100,"a3,Link");

  Ntrial = 100;

  L = randomITensor(h1,prime(a1),a1);
  R = randomITensor(h3,prime(a3),a3);
  H1 = randomITensor(h1,prime(s1),s1,h2);
  H2 = randomITensor(h2,prime(s2),s2,h3);
  phi = randomITensor(a1,s1,s2,a3);

  # Dry run (JIT)
  time_dry = @elapsed begin
  for n=1:5
    phip = L*phi;
    phip *= H1;
    phip *= H2;
    phip *= R;
  end
  end
  reset!(timer)
  # End dry run

  time = @elapsed begin
  for n=1:Ntrial
    phip = L*phi;
    phip *= H1;
    phip *= H2;
    phip *= R;
  end
  end

  @printf "time = %.12f\n" time
  printTimes(timer)

end; main()
