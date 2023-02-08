module BenchContract

using BenchmarkTools
using ITensors
using ITensors.HDF5
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

suite["matmul"] = BenchmarkGroup()
suite["matmul"]["allocating"] = BenchmarkGroup()
suite["matmul"]["inplace"] = BenchmarkGroup()
for d in 20:20:100
  i = Index(d)
  A = randomITensor(i, i')
  B = randomITensor(i', i'')
  C = randomITensor(i, i'')

  suite["matmul"]["allocating"]["$d"] = @benchmarkable $A * $B
  suite["matmul"]["inplace"]["$d"] = @benchmarkable $C .= $A .* $B
end

suite["heff_2site"] = BenchmarkGroup()
let
  s1 = Index(2, "s1,Site")
  s2 = Index(2, "s2,Site")
  h1 = Index(10, "h1,Link,H")
  h2 = Index(10, "h2,Link,H")
  h3 = Index(10, "h3,Link,H")
  a1 = Index(100, "a1,Link")
  a3 = Index(100, "a3,Link")
  phi = randomITensor(a1, s1, s2, a3)
  H1 = randomITensor(h1, s1', s1, h2)
  H2 = randomITensor(h2, s2', s2, h3)
  L = randomITensor(h1, a1', a1)
  R = randomITensor(h3, a3', a3)

  suite["heff_2site"]["dense"] = @benchmarkable $phi * $L * $H1 * $H2 * $R
end

suite["heff_2site"]["blocksparse"] = BenchmarkGroup()
let
  """
  Load the ground state energy `energy`, Hamiltonian MPO `H`, and ground state MPS `psi` resulting from running:
  ```julia
  using Pkg
  Pkg.add(; name="ITensors", version="0.3.25")

  using ITensors

  example_dir = joinpath(pkgdir(ITensors), "examples", "dmrg")
  include(joinpath(example_dir, "2d_hubbard_conserve_momentum.jl"))
  kwargs = (; Nx=8, Ny=4, U=4.0, t=1.0, nsweeps=10,
    maxdim=3000, random_init=false, threaded_blocksparse=true)
  energy, H, psi = main(; kwargs...);

  function get_effective_hamiltonian(H, psi, center=(length(psi) ÷ 2))
    psi = orthogonalize(psi, center)
    PH = ProjMPO(H)
    PH = position!(PH, psi, center)
    v = psi[center] * psi[center + 1]
    L = lproj(PH)
    R = rproj(PH)
    H1 = H[center]
    H2 = H[center + 1]
    return [v, L, H1, H2, R]
  end

  tn = get_effective_hamiltonian(H, psi);

  using ITensors.HDF5
  file_dir = joinpath(pkgdir(ITensors), "benchmark", "artifacts")
  h5open(joinpath(file_dir, "2d_hubbard_conserve_momentum.h5"), "w") do fid
    fid["energy"] = energy
    fid["ntensors"] = length(tn)
    for j in eachindex(tn)
      fid["tn[" * string(j) * "]"] = tn[j]
    end
  end
  ```
  """
  file_path = joinpath(pkgdir(ITensors), "benchmark", "artifacts")
  file_name = "2d_hubbard_conserve_momentum.h5"
  energy, tn = h5open(joinpath(file_path, file_name)) do fid
    energy = read(fid, "energy")
    ntensors = read(fid, "ntensors")
    tn = Vector{ITensor}(undef, ntensors)
    for j in eachindex(tn)
      tn[j] = read(fid, "tn[" * string(j) * "]", ITensor)
    end
    return energy, tn
  end

  # Correctness check
  @assert contract([tn; dag(first(tn)')])[] ≈ energy

  suite["heff_2site"]["blocksparse"]["sequential"] = @benchmarkable begin
    ITensors.enable_threaded_blocksparse(false)
    contract($tn)
  end
  suite["heff_2site"]["blocksparse"]["threaded"] = @benchmarkable begin
    ITensors.enable_threaded_blocksparse(true)
    contract($tn)
    ITensors.enable_threaded_blocksparse(false)
  end
end

end

BenchContract.suite
