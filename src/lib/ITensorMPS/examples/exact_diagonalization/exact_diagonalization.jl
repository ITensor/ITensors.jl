using ITensors
using KrylovKit
using LinearAlgebra
using MKL

include("fuse_inds.jl")

ITensors.Strided.disable_threads()
ITensors.disable_threaded_blocksparse()

function heisenberg(n)
  os = OpSum()
  for j in 1:(n - 1)
    os += 1 / 2, "S+", j, "S-", j + 1
    os += 1 / 2, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  return os
end

function main(n; blas_num_threads=Sys.CPU_THREADS, fuse=true, binary=true)
  if n > 16
    @warn "System size of $n is likely too large for exact diagonalization."
  end

  BLAS.set_num_threads(blas_num_threads)

  # Hilbert space
  s = siteinds("S=1/2", n; conserve_qns=true)
  H = MPO(heisenberg(n), s)
  initstate(j) = isodd(j) ? "↑" : "↓"
  ψ0 = randomMPS(s, initstate; linkdims=10)

  edmrg, ψdmrg = dmrg(H, ψ0; nsweeps=10, cutoff=1e-6)

  if fuse
    if binary
      println("Fuse the indices using a binary tree")
      T = fusion_tree_binary(s)
      H_full = @time fuse_inds_binary(H, T)
      ψ0_full = @time fuse_inds_binary(ψ0, T)
    else
      println("Fuse the indices using an unbalances tree")
      T = fusion_tree(s)
      H_full = @time fuse_inds(H, T)
      ψ0_full = @time fuse_inds(ψ0, T)
    end
  else
    println("Don't fuse the indices")
    @disable_warn_order begin
      H_full = @time contract(H)
      ψ0_full = @time contract(ψ0)
    end
  end

  vals, vecs, info = @time eigsolve(
    H_full, ψ0_full, 1, :SR; ishermitian=true, tol=1e-6, krylovdim=30, eager=true
  )

  @show edmrg, vals[1]
end

main(14)
