using ITensors,
      Test,
      Random


include("util.jl")


function isingMPO(sites)::MPO
  H = MPO(sites)
  N = length(H)
  link = fill(Index(),N+1)
  for n=1:N+1
    link[n] = Index(3,"Link,Ising,l=$(n-1)")
  end
  for n=1:N
    s = sites[n]
    ll = link[n]
    rl = link[n+1]
    H[n] = ITensor(dag(ll),dag(s),s',rl)
    H[n] += setelt(ll[1])*setelt(rl[1])*op(sites,"Id",n)
    H[n] += setelt(ll[3])*setelt(rl[3])*op(sites,"Id",n)
    H[n] += setelt(ll[2])*setelt(rl[1])*op(sites,"Sz",n)
    H[n] += setelt(ll[3])*setelt(rl[2])*op(sites,"Sz",n)
  end
  LE = ITensor(link[1]); LE[3] = 1.0;
  RE = ITensor(dag(link[N+1])); RE[1] = 1.0;
  H[1] *= LE
  H[N] *= RE
  return H
end

function heisenbergMPO(sites,
                       h::Vector{Float64},
                       onsite::String="Sz")::MPO
  H = MPO(sites)
  N = length(H)
  link = fill(Index(),N+1)
  for n=1:N+1
    link[n] = Index(5,"Link,Heis,l=$(n-1)")
  end
  for n=1:N
    s = sites[n]
    ll = link[n]
    rl = link[n+1]
    H[n] = ITensor(ll,s,s',rl)
    H[n] += setelt(ll[1])*setelt(rl[1])*op(sites,"Id",n)
    H[n] += setelt(ll[5])*setelt(rl[5])*op(sites,"Id",n)
    H[n] += setelt(ll[2])*setelt(rl[1])*op(sites,"S+",n)
    H[n] += setelt(ll[3])*setelt(rl[1])*op(sites,"S-",n)
    H[n] += setelt(ll[4])*setelt(rl[1])*op(sites,"Sz",n)
    H[n] += setelt(ll[5])*setelt(rl[2])*op(sites,"S-",n)*0.5
    H[n] += setelt(ll[5])*setelt(rl[3])*op(sites,"S+",n)*0.5
    H[n] += setelt(ll[5])*setelt(rl[4])*op(sites,"Sz",n)
    H[n] += setelt(ll[5])*setelt(rl[1])*op(sites,onsite,n)*h[n]
  end
  H[1] *= setelt(link[1][5])
  H[N] *= setelt(link[N+1][1])
  return H
end

function NNheisenbergMPO(sites,
                         J1::Float64,
                         J2::Float64)::MPO
  H = MPO(sites)
  N = length(H)
  link = fill(Index(),N+1)
  for n=1:N+1
    link[n] = Index(8,"Link,H,l=$(n-1)")
  end
  for n=1:N
    s = sites[n]
    ll = link[n]
    rl = link[n+1]
    H[n] = ITensor(ll,s,s',rl)
    H[n] += setelt(ll[1])*setelt(rl[1])*op(sites,"Id",n)
    H[n] += setelt(ll[8])*setelt(rl[8])*op(sites,"Id",n)

    H[n] += setelt(ll[2])*setelt(rl[1])*op(sites,"S-",n)
    H[n] += setelt(ll[5])*setelt(rl[2])*op(sites,"Id",n)
    H[n] += setelt(ll[8])*setelt(rl[2])*op(sites,"S+",n)*J1/2
    H[n] += setelt(ll[8])*setelt(rl[5])*op(sites,"S+",n)*J2/2

    H[n] += setelt(ll[3])*setelt(rl[1])*op(sites,"S+",n)
    H[n] += setelt(ll[6])*setelt(rl[3])*op(sites,"Id",n)
    H[n] += setelt(ll[8])*setelt(rl[3])*op(sites,"S-",n)*J1/2
    H[n] += setelt(ll[8])*setelt(rl[6])*op(sites,"S-",n)*J2/2

    H[n] += setelt(ll[4])*setelt(rl[1])*op(sites,"Sz",n)
    H[n] += setelt(ll[7])*setelt(rl[4])*op(sites,"Id",n)
    H[n] += setelt(ll[8])*setelt(rl[4])*op(sites,"Sz",n)*J1
    H[n] += setelt(ll[8])*setelt(rl[7])*op(sites,"Sz",n)*J2
  end
  H[1] *= setelt(link[1][8])
  H[N] *= setelt(link[N+1][1])
  return H
end

function threeSiteIsingMPO(sites,
                           h::Vector{Float64})::MPO
  H = MPO(sites)
  N = length(H)
  link = fill(Index(),N+1)
  for n=1:N+1
    link[n] = Index(4,"Link,l=$(n-1)")
  end
  for n=1:N
    s = sites[n]
    ll = link[n]
    rl = link[n+1]
    H[n] = ITensor(ll,s,s',rl)
    H[n] += setelt(ll[1])*setelt(rl[1])*op(sites,"Id",n)
    H[n] += setelt(ll[4])*setelt(rl[4])*op(sites,"Id",n)
    H[n] += setelt(ll[2])*setelt(rl[1])*op(sites,"Sz",n)
    H[n] += setelt(ll[3])*setelt(rl[2])*op(sites,"Sz",n)
    H[n] += setelt(ll[4])*setelt(rl[3])*op(sites,"Sz",n)
    H[n] += setelt(ll[4])*setelt(rl[1])*op(sites,"Sx",n)*h[n]
  end
  H[1] *= setelt(link[1][4])
  H[N] *= setelt(link[N+1][1])
  return H
end

function fourSiteIsingMPO(sites)::MPO
  H = MPO(sites)
  N = length(H)
  link = fill(Index(),N+1)
  for n=1:N+1
    link[n] = Index(5,"Link,l=$(n-1)")
  end
  for n=1:N
    s = sites[n]
    ll = link[n]
    rl = link[n+1]
    H[n] = ITensor(ll,s,s',rl)
    H[n] += setelt(ll[1])*setelt(rl[1])*op(sites,"Id",n)
    H[n] += setelt(ll[5])*setelt(rl[5])*op(sites,"Id",n)
    H[n] += setelt(ll[2])*setelt(rl[1])*op(sites,"Sz",n)
    H[n] += setelt(ll[3])*setelt(rl[2])*op(sites,"Sz",n)
    H[n] += setelt(ll[4])*setelt(rl[3])*op(sites,"Sz",n)
    H[n] += setelt(ll[5])*setelt(rl[4])*op(sites,"Sz",n)
  end
  H[1] *= setelt(link[1][5])
  H[N] *= setelt(link[N+1][1])
  return H
end

@testset "AutoMPO" begin

  N = 10

  @testset "Show MPOTerm" begin
    ampo = AutoMPO()
    add!(ampo,"Sz",1,"Sz",2)
    @test sprint(show,terms(ampo)[1]) == "\"Sz\"(1)\"Sz\"(2)"
  end

  @testset "Show AutoMPO" begin
    ampo = AutoMPO()
    add!(ampo,"Sz",1,"Sz",2)
    add!(ampo,"Sz",2,"Sz",3)
    expected_string = "AutoMPO:\n  \"Sz\"(1)\"Sz\"(2)\n  \"Sz\"(2)\"Sz\"(3)\n"
    @test sprint(show,ampo) == expected_string
  end

  @testset "Single creation op" begin
    ampo = AutoMPO()
    add!(ampo,"Cdagup",3)
    sites = siteinds("Electron",N)
    W = toMPO(ampo,sites)
    psi = makeRandomMPS(sites)
    cdu_psi = copy(psi)
    cdu_psi[3] = noprime(cdu_psi[3]*op(sites,"Cdagup",3))
    @test inner(psi,W,psi) ≈ inner(cdu_psi,psi)
  end

  @testset "Ising" begin
    ampo = AutoMPO()
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
    end
    sites = siteinds("S=1/2",N)
    Ha = toMPO(ampo,sites)
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

  @testset "Ising-Different Order" begin
    ampo = AutoMPO()
    for j=1:N-1
      add!(ampo,"Sz",j+1,"Sz",j)
    end
    sites = siteinds("S=1/2",N)
    Ha = toMPO(ampo,sites)
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

  @testset "Heisenberg" begin
    ampo = AutoMPO()
    h = rand(N) #random magnetic fields
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
      add!(ampo,0.5,"S+",j,"S-",j+1)
      add!(ampo,0.5,"S-",j,"S+",j+1)
    end
    for j=1:N
      add!(ampo,h[j],"Sz",j)
    end

    sites = siteinds("S=1/2",N)
    Ha = toMPO(ampo,sites)
    He = heisenbergMPO(sites,h)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end


  @testset "Multiple Onsite Ops" begin
    sites = siteinds("S=1",N)
    ampo1 = AutoMPO()
    for j=1:N-1
      add!(ampo1,"Sz",j,"Sz",j+1)
      add!(ampo1,0.5,"S+",j,"S-",j+1)
      add!(ampo1,0.5,"S-",j,"S+",j+1)
    end
    for j=1:N
      add!(ampo1,"Sz*Sz",j)
    end
    Ha1 = toMPO(ampo1,sites)

    ampo2 = AutoMPO()
    for j=1:N-1
      add!(ampo2,"Sz",j,"Sz",j+1)
      add!(ampo2,0.5,"S+",j,"S-",j+1)
      add!(ampo2,0.5,"S-",j,"S+",j+1)
    end
    for j=1:N
      add!(ampo2,"Sz",j,"Sz",j)
    end
    Ha2 = toMPO(ampo2,sites)

    He = heisenbergMPO(sites,ones(N),"Sz*Sz")
    psi = makeRandomMPS(sites)
    Oe = inner(psi,He,psi)
    Oa1 = inner(psi,Ha1,psi)
    @test Oa1 ≈ Oe
    Oa2 = inner(psi,Ha2,psi)
    @test Oa2 ≈ Oe
  end

  @testset "Three-site ops" begin
    ampo = AutoMPO()
    # To test version of add! taking a coefficient
    add!(ampo,1.0,"Sz",1,"Sz",2,"Sz",3)
    @test length(terms(ampo)) == 1
    for j=2:N-2
      add!(ampo,"Sz",j,"Sz",j+1,"Sz",j+2)
    end
    h = ones(N)
    for j=1:N
      add!(ampo,h[j],"Sx",j)
    end
    sites = siteinds("S=1/2",N)
    Ha = toMPO(ampo,sites)
    He = threeSiteIsingMPO(sites,h)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

  @testset "Four-site ops" begin
    ampo = AutoMPO()
    for j=1:N-3
      add!(ampo,"Sz",j,"Sz",j+1,"Sz",j+2,"Sz",j+3)
    end
    sites = siteinds("S=1/2",N)
    Ha = toMPO(ampo,sites)
    He = fourSiteIsingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
  end

  @testset "Next-neighbor Heisenberg" begin
    ampo = AutoMPO()
    J1 = 1.0
    J2 = 0.5
    for j=1:N-1
      add!(ampo,J1,  "Sz",j,"Sz",j+1)
      add!(ampo,J1*0.5,"S+",j,"S-",j+1)
      add!(ampo,J1*0.5,"S-",j,"S+",j+1)
    end
    for j=1:N-2
      add!(ampo,J2,  "Sz",j,"Sz",j+2)
      add!(ampo,J2*0.5,"S+",j,"S-",j+2)
      add!(ampo,J2*0.5,"S-",j,"S+",j+2)
    end
    sites = siteinds("S=1/2",N)
    Ha = toMPO(ampo,sites)

    He = NNheisenbergMPO(sites,J1,J2)
    psi = makeRandomMPS(sites)
    Oa = inner(psi,Ha,psi)
    Oe = inner(psi,He,psi)
    @test Oa ≈ Oe
    #@test maxlinkdim(Ha) == 8
  end

  @testset "Onsite Regression Test" begin
    sites = siteinds("S=1",4)
    ampo = AutoMPO()
    add!(ampo, 0.5, "Sx",1)
    add!(ampo, 0.5, "Sy",1)
    H = toMPO(ampo, sites)
    l = commonind(H[1],H[2])
    T = setelt(l[1])*H[1]
    O = op(sites[1],"Sx")+op(sites[1],"Sy")
    @test norm(T-0.5*O) < 1E-8

      
    sites = siteinds("S=1",2)
    ampo = AutoMPO()
    add!(ampo, 0.5im, "Sx",1)
    add!(ampo, 0.5, "Sy",1)
    H = toMPO(ampo, sites)
    T = H[1]*H[2]
    O = im*op(sites[1],"Sx")*op(sites[2],"Id")+op(sites[1],"Sy")*op(sites[2],"Id")
    @test norm(T-0.5*O) < 1E-8
  end

  @testset "+ syntax" begin

    @testset "Show MPOTerm" begin
      ampo = AutoMPO()
      ampo += ("Sz",1,"Sz",2)
      @test sprint(show,terms(ampo)[1]) == "\"Sz\"(1)\"Sz\"(2)"
    end

    @testset "Show AutoMPO" begin
      ampo = AutoMPO()
      ampo += ("Sz",1,"Sz",2)
      ampo += ("Sz",2,"Sz",3)
      expected_string = "AutoMPO:\n  \"Sz\"(1)\"Sz\"(2)\n  \"Sz\"(2)\"Sz\"(3)\n"
      @test sprint(show,ampo) == expected_string
    end

    @testset "Single creation op" begin
      ampo = AutoMPO()
      ampo += ("Cdagup",3)
      sites = siteinds("Electron",N)
      W = toMPO(ampo,sites)
      psi = makeRandomMPS(sites)
      cdu_psi = copy(psi)
      cdu_psi[3] = noprime(cdu_psi[3]*op(sites,"Cdagup",3))
      @test inner(psi,W,psi) ≈ inner(cdu_psi,psi)
    end

    @testset "Ising" begin
      ampo = AutoMPO()
      for j=1:N-1
        ampo += ("Sz",j,"Sz",j+1)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Ising-Different Order" begin
      ampo = AutoMPO()
      for j=1:N-1
        ampo += ("Sz",j+1,"Sz",j)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Heisenberg" begin
      ampo = AutoMPO()
      h = rand(N) #random magnetic fields
      for j=1:N-1
        ampo += ("Sz",j,"Sz",j+1)
        ampo += (0.5,"S+",j,"S-",j+1)
        ampo += (0.5,"S-",j,"S+",j+1)
      end
      for j=1:N
        ampo += (h[j],"Sz",j)
      end

      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = heisenbergMPO(sites,h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end


    @testset "Multiple Onsite Ops" begin
      sites = siteinds("S=1",N)
      ampo1 = AutoMPO()
      for j=1:N-1
        ampo1 += ("Sz",j,"Sz",j+1)
        ampo1 += (0.5,"S+",j,"S-",j+1)
        ampo1 += (0.5,"S-",j,"S+",j+1)
      end
      for j=1:N
        ampo1 += ("Sz*Sz",j)
      end
      Ha1 = toMPO(ampo1,sites)

      ampo2 = AutoMPO()
      for j=1:N-1
        ampo2 += ("Sz",j,"Sz",j+1)
        ampo2 += (0.5,"S+",j,"S-",j+1)
        ampo2 += (0.5,"S-",j,"S+",j+1)
      end
      for j=1:N
        ampo2 += ("Sz",j,"Sz",j)
      end
      Ha2 = toMPO(ampo2,sites)

      He = heisenbergMPO(sites,ones(N),"Sz*Sz")
      psi = makeRandomMPS(sites)
      Oe = inner(psi,He,psi)
      Oa1 = inner(psi,Ha1,psi)
      @test Oa1 ≈ Oe
      Oa2 = inner(psi,Ha2,psi)
      @test Oa2 ≈ Oe
    end

    @testset "Three-site ops" begin
      ampo = AutoMPO()
      # To test version of add! taking a coefficient
      ampo += (1.0,"Sz",1,"Sz",2,"Sz",3)
      @test length(terms(ampo)) == 1
      for j=2:N-2
        ampo += ("Sz",j,"Sz",j+1,"Sz",j+2)
      end
      h = ones(N)
      for j=1:N
        ampo += (h[j],"Sx",j)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = threeSiteIsingMPO(sites,h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Four-site ops" begin
      ampo = AutoMPO()
      for j=1:N-3
        ampo += ("Sz",j,"Sz",j+1,"Sz",j+2,"Sz",j+3)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = fourSiteIsingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Next-neighbor Heisenberg" begin
      ampo = AutoMPO()
      J1 = 1.0
      J2 = 0.5
      for j=1:N-1
        ampo += (J1,  "Sz",j,"Sz",j+1)
        ampo += (J1*0.5,"S+",j,"S-",j+1)
        ampo += (J1*0.5,"S-",j,"S+",j+1)
      end
      for j=1:N-2
        ampo += (J2,  "Sz",j,"Sz",j+2)
        ampo += (J2*0.5,"S+",j,"S-",j+2)
        ampo += (J2*0.5,"S-",j,"S+",j+2)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)

      He = NNheisenbergMPO(sites,J1,J2)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
      #@test maxlinkdim(Ha) == 8
    end

    @testset "Onsite Regression Test" begin
      sites = siteinds("S=1",4)
      ampo = AutoMPO()
      ampo += (0.5, "Sx",1)
      ampo += (0.5, "Sy",1)
      H = toMPO(ampo, sites)
      l = commonind(H[1],H[2])
      T = setelt(l[1])*H[1]
      O = op(sites[1],"Sx")+op(sites[1],"Sy")
      @test norm(T-0.5*O) < 1E-8

        
      sites = siteinds("S=1",2)
      ampo = AutoMPO()
      ampo += (0.5im, "Sx",1)
      ampo += (0.5, "Sy",1)
      H = toMPO(ampo, sites)
      T = H[1]*H[2]
      O = im*op(sites[1],"Sx")*op(sites[2],"Id")+op(sites[1],"Sy")*op(sites[2],"Id")
      @test norm(T-0.5*O) < 1E-8
    end

  end

  @testset ".+= syntax" begin

    @testset "Show MPOTerm" begin
      ampo = AutoMPO()
      ampo .+= ("Sz",1,"Sz",2)
      @test sprint(show,terms(ampo)[1]) == "\"Sz\"(1)\"Sz\"(2)"
    end

    @testset "Show AutoMPO" begin
      ampo = AutoMPO()
      ampo .+= ("Sz",1,"Sz",2)
      ampo .+= ("Sz",2,"Sz",3)
      expected_string = "AutoMPO:\n  \"Sz\"(1)\"Sz\"(2)\n  \"Sz\"(2)\"Sz\"(3)\n"
      @test sprint(show,ampo) == expected_string
    end

    @testset "Single creation op" begin
      ampo = AutoMPO()
      ampo .+= ("Cdagup",3)
      sites = siteinds("Electron",N)
      W = toMPO(ampo,sites)
      psi = makeRandomMPS(sites)
      cdu_psi = copy(psi)
      cdu_psi[3] = noprime(cdu_psi[3]*op(sites,"Cdagup",3))
      @test inner(psi,W,psi) ≈ inner(cdu_psi,psi)
    end

    @testset "Ising" begin
      ampo = AutoMPO()
      for j=1:N-1
        ampo .+= ("Sz",j,"Sz",j+1)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Ising-Different Order" begin
      ampo = AutoMPO()
      for j=1:N-1
        ampo .+= ("Sz",j+1,"Sz",j)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Heisenberg" begin
      ampo = AutoMPO()
      h = rand(N) #random magnetic fields
      for j=1:N-1
        ampo .+= ("Sz",j,"Sz",j+1)
        ampo .+= (0.5,"S+",j,"S-",j+1)
        ampo .+= (0.5,"S-",j,"S+",j+1)
      end
      for j=1:N
        ampo .+= (h[j],"Sz",j)
      end

      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = heisenbergMPO(sites,h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end


    @testset "Multiple Onsite Ops" begin
      sites = siteinds("S=1",N)
      ampo1 = AutoMPO()
      for j=1:N-1
        ampo1 .+= ("Sz",j,"Sz",j+1)
        ampo1 .+= (0.5,"S+",j,"S-",j+1)
        ampo1 .+= (0.5,"S-",j,"S+",j+1)
      end
      for j=1:N
        ampo1 .+= ("Sz*Sz",j)
      end
      Ha1 = toMPO(ampo1,sites)

      ampo2 = AutoMPO()
      for j=1:N-1
        ampo2 .+= ("Sz",j,"Sz",j+1)
        ampo2 .+= (0.5,"S+",j,"S-",j+1)
        ampo2 .+= (0.5,"S-",j,"S+",j+1)
      end
      for j=1:N
        ampo2 .+= ("Sz",j,"Sz",j)
      end
      Ha2 = toMPO(ampo2,sites)

      He = heisenbergMPO(sites,ones(N),"Sz*Sz")
      psi = makeRandomMPS(sites)
      Oe = inner(psi,He,psi)
      Oa1 = inner(psi,Ha1,psi)
      @test Oa1 ≈ Oe
      Oa2 = inner(psi,Ha2,psi)
      @test Oa2 ≈ Oe
    end

    @testset "Three-site ops" begin
      ampo = AutoMPO()
      # To test version of add! taking a coefficient
      ampo .+= (1.0,"Sz",1,"Sz",2,"Sz",3)
      @test length(terms(ampo)) == 1
      for j=2:N-2
        ampo .+= ("Sz",j,"Sz",j+1,"Sz",j+2)
      end
      h = ones(N)
      for j=1:N
        ampo .+= (h[j],"Sx",j)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = threeSiteIsingMPO(sites,h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Four-site ops" begin
      ampo = AutoMPO()
      for j=1:N-3
        ampo .+= ("Sz",j,"Sz",j+1,"Sz",j+2,"Sz",j+3)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)
      He = fourSiteIsingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
    end

    @testset "Next-neighbor Heisenberg" begin
      ampo = AutoMPO()
      J1 = 1.0
      J2 = 0.5
      for j=1:N-1
        ampo .+= (J1,  "Sz",j,"Sz",j+1)
        ampo .+= (J1*0.5,"S+",j,"S-",j+1)
        ampo .+= (J1*0.5,"S-",j,"S+",j+1)
      end
      for j=1:N-2
        ampo .+= (J2,  "Sz",j,"Sz",j+2)
        ampo .+= (J2*0.5,"S+",j,"S-",j+2)
        ampo .+= (J2*0.5,"S-",j,"S+",j+2)
      end
      sites = siteinds("S=1/2",N)
      Ha = toMPO(ampo,sites)

      He = NNheisenbergMPO(sites,J1,J2)
      psi = makeRandomMPS(sites)
      Oa = inner(psi,Ha,psi)
      Oe = inner(psi,He,psi)
      @test Oa ≈ Oe
      #@test maxlinkdim(Ha) == 8
    end

    @testset "Onsite Regression Test" begin
      sites = siteinds("S=1",4)
      ampo = AutoMPO()
      ampo .+= (0.5, "Sx",1)
      ampo .+= (0.5, "Sy",1)
      H = toMPO(ampo, sites)
      l = commonind(H[1],H[2])
      T = setelt(l[1])*H[1]
      O = op(sites[1],"Sx")+op(sites[1],"Sy")
      @test norm(T-0.5*O) < 1E-8

        
      sites = siteinds("S=1",2)
      ampo = AutoMPO()
      ampo .+= (0.5im, "Sx",1)
      ampo .+= (0.5, "Sy",1)
      H = toMPO(ampo, sites)
      T = H[1]*H[2]
      O = im*op(sites[1],"Sx")*op(sites[2],"Id")+op(sites[1],"Sy")*op(sites[2],"Id")
      @test norm(T-0.5*O) < 1E-8
    end

  end
end
