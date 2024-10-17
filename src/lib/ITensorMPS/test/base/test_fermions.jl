using ITensors, Test
import ITensors: Out, In

@testset "AutoFermion MPS, MPO, and OpSum" begin
  ITensors.enable_auto_fermion()

  @testset "MPS Tests" begin
    @testset "Product MPS consistency checks" begin
      s = siteinds("Fermion", 3; conserve_qns=true)

      pA = MPS(s, [2, 1, 2])
      TA = ITensor(s[1], s[2], s[3])
      TA[s[1] => 2, s[2] => 1, s[3] => 2] = 1.0
      A = pA[1] * pA[2] * pA[3]
      @test norm(A - TA) < 1E-8

      pB = MPS(s, [1, 2, 2])
      TB = ITensor(s[1], s[2], s[3])
      TB[s[1] => 1, s[2] => 2, s[3] => 2] = 1.0
      B = pB[1] * pB[2] * pB[3]
      @test norm(B - TB) < 1E-8
    end
    @testset "MPS inner regression test" begin
      sites = siteinds("Fermion", 3; conserve_qns=true)
      psi = MPS(sites, [2, 2, 1])
      @test inner(psi, psi) ≈ 1.0
    end
    @testset "Orthogonalize of Product MPS" begin
      N = 3

      sites = siteinds("Fermion", N; conserve_qns=true)

      state = [1 for n in 1:N]
      state[1] = 2
      state[2] = 2
      psi = MPS(sites, state)
      psi_fluxes = [flux(psi[n]) for n in 1:N]

      psi_orig = copy(psi)
      orthogonalize!(psi, 1)
      @test inner(psi_orig, psi) ≈ 1.0
      @test inner(psi, psi_orig) ≈ 1.0
    end
  end

  @testset "Fermionic OpSum Tests" begin
    @testset "Spinless Fermion Hamiltonian" begin
      N = 2
      sites = siteinds("Fermion", N; conserve_qns=true)
      t1 = 1.0
      os = OpSum()
      for b in 1:(N-1)
        os -= t1, "Cdag", b, "C", b + 1
        os -= t1, "Cdag", b + 1, "C", b
      end
      H = MPO(os, sites)

      HH = H[1]
      for n in 2:N
        HH *= H[n]
      end
      HHc = dag(swapprime(HH, 0, 1))
      @test norm(HHc - HH) < 1E-8
    end

    @testset "Fermion Hamiltonian Matrix Elements" begin
      N = 10
      t1 = 0.654
      V1 = 1.23

      sites = siteinds("Fermion", N; conserve_qns=true)

      os = OpSum()
      for b in 1:(N-1)
        os -= t1, "Cdag", b, "C", b + 1
        os -= t1, "Cdag", b + 1, "C", b
        os += V1, "N", b, "N", b + 1
      end
      H = MPO(os, sites)

      for j in 1:(N-2)
        stateA = [1 for n in 1:N]
        stateA[j] = 2
        stateA[N] = 2 # to make MPS bosonic

        stateB = [1 for n in 1:N]
        stateB[j + 1] = 2
        stateB[N] = 2 # to make MPS bosonic

        psiA = MPS(sites, stateA)
        psiB = MPS(sites, stateB)

        @test inner(psiA', H, psiB) ≈ -t1
        @test inner(psiB', H, psiA) ≈ -t1
      end

      for j in 1:(N-1)
        state = [1 for n in 1:N]
        state[j] = 2
        state[j + 1] = 2
        psi = MPS(sites, state)
        @test inner(psi', H, psi) ≈ V1
      end
    end

    @testset "Fermion Second Neighbor Hopping" begin
      N = 4
      t1 = 1.79
      t2 = 0.427
      s = siteinds("Fermion", N; conserve_qns=true)
      os = OpSum()
      for n in 1:(N-1)
        os -= t1, "Cdag", n, "C", n + 1
        os -= t1, "Cdag", n + 1, "C", n
      end
      for n in 1:(N-2)
        os -= t2, "Cdag", n, "C", n + 2
        os -= t2, "Cdag", n + 2, "C", n
      end
      H = MPO(os, s)

      state1 = [1 for n in 1:N]
      state1[1] = 2
      state1[4] = 2
      psi1 = MPS(s, state1)

      state2 = [1 for n in 1:N]
      state2[2] = 2
      state2[4] = 2
      psi2 = MPS(s, state2)

      state3 = [1 for n in 1:N]
      state3[3] = 2
      state3[4] = 2
      psi3 = MPS(s, state3)

      @test inner(psi1', H, psi2) ≈ -t1
      @test inner(psi2', H, psi1) ≈ -t1
      @test inner(psi2', H, psi3) ≈ -t1
      @test inner(psi3', H, psi2) ≈ -t1

      @test inner(psi1', H, psi3) ≈ -t2
      @test inner(psi3', H, psi1) ≈ -t2

      # Add stationary particle to site 2,
      # hopping over should change sign:
      state1[2] = 2
      psi1 = MPS(s, state1)
      state3[2] = 2
      psi3 = MPS(s, state3)
      @test inner(psi1', H, psi3) ≈ +t2
      @test inner(psi3', H, psi1) ≈ +t2
    end

    @testset "OpSum Regression Test" begin
      N = 3
      s = siteinds("Fermion", N; conserve_qns=true)

      os = OpSum()
      os += "Cdag", 1, "C", 3
      @test_nowarn H = MPO(os, s)
    end
  end

  @testset "DMRG Tests" begin
    @testset "Nearest Neighbor Fermions" begin
      N = 8
      t1 = 1.0
      V1 = 4.0

      s = siteinds("Fermion", N; conserve_qns=true)

      ost = OpSum()
      osV = OpSum()
      for b in 1:(N-1)
        ost -= t1, "Cdag", b, "C", b + 1
        ost -= t1, "Cdag", b + 1, "C", b
        osV += V1, "N", b, "N", b + 1
      end
      Ht = MPO(ost, s)
      HV = MPO(osV, s)

      state = ["Emp" for n in 1:N]
      for i in 1:2:N
        state[i] = "Occ"
      end
      psi0 = MPS(s, state)

      sweeps = Sweeps(3)
      maxdim!(sweeps, 20, 20, 40, 80, 200)
      cutoff!(sweeps, 1E-6)

      correct_energy = -2.859778

      energy, psi = dmrg([Ht, HV], psi0, sweeps; outputlevel=0)
      @test abs(energy - correct_energy) < 1E-4

      # Test using SVD within DMRG too:
      energy, psi = dmrg([Ht, HV], psi0, sweeps; outputlevel=0, which_decomp="svd")
      @test abs(energy - correct_energy) < 1E-4

      # Test using only eigen decomp:
      energy, psi = dmrg([Ht, HV], psi0, sweeps; outputlevel=0, which_decomp="eigen")
      @test abs(energy - correct_energy) < 1E-4
    end

    @testset "Further Neighbor and Correlations" begin
      N = 8
      t1 = 1.0
      t2 = 0.2

      s = siteinds("Fermion", N; conserve_qns=true)

      ost = OpSum()
      for b in 1:(N-1)
        ost -= t1, "Cdag", b, "C", b + 1
        ost -= t1, "Cdag", b + 1, "C", b
      end
      for b in 1:(N-2)
        ost -= t2, "Cdag", b, "C", b + 2
        ost -= t2, "Cdag", b + 2, "C", b
      end
      Ht = MPO(ost, s)

      state = ["Emp" for n in 1:N]
      for i in 1:2:N
        state[i] = "Occ"
      end
      psi0 = MPS(s, state)

      sweeps = Sweeps(3)
      maxdim!(sweeps, 20, 20, 40, 80, 200)
      cutoff!(sweeps, 1E-6)

      energy, psi = dmrg(Ht, psi0, sweeps; outputlevel=0)

      energy_inner = inner(psi', Ht, psi)

      C = correlation_matrix(psi, "Cdag", "C")
      C_energy =
        sum(j -> -2t1 * C[j, j + 1], 1:(N-1)) + sum(j -> -2t2 * C[j, j + 2], 1:(N-2))

      @test energy_inner ≈ energy
      @test C_energy ≈ energy
    end
  end

  @testset "MPS gate system" begin
    @testset "Fermion sites" begin
      N = 3

      s = siteinds("Fermion", N; conserve_qns=true)

      # Ground state |000⟩
      ψ000 = MPS(s, "0")

      # Start state |011⟩
      ψ011 = MPS(s, n -> n == 2 || n == 3 ? "1" : "0")

      # Reference state |110⟩
      ψ110 = MPS(s, n -> n == 1 || n == 2 ? "1" : "0")

      function ITensors.op(::OpName"CdagC3", ::SiteType, s1::Index, s2::Index)
        return op("Cdag", s1) * op("C", s2)
      end

      os = [("CdagC3", 1, 3)]
      Os = ops(os, s)

      # Results in -|110⟩
      ψ1 = product(Os, ψ011; cutoff=1e-15)

      @test inner(ψ1, ψ110) == -1

      os = OpSum()
      os += "Cdag", 1, "C", 3
      H = MPO(os, s)

      # Results in -|110⟩
      ψ2 = noprime(contract(H, ψ011; cutoff=1e-15))

      @test inner(ψ2, ψ110) == -1
    end
  end

  ITensors.disable_auto_fermion()
end
