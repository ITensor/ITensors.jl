using ITensors: Algorithm, @Algorithm_str

# Select solver function
solver_function(solver_backend::String) = solver_function(Algorithm(solver_backend))
solver_function(::Algorithm"exponentiate") = exponentiate
function solver_function(solver_backend::Algorithm)
  return error(
    "solver_backend=$(String(solver_backend)) not recognized (only \"exponentiate\" is supported)",
  )
end

# Kept for backwards compatibility
function solver_function(::Algorithm"applyexp")
  println(
    "Warning: the `solver_backend` option `\"applyexp\"` in `tdvp` has been removed. `\"exponentiate\"` will be used instead. To remove this warning, don't specify the `solver_backend` keyword argument.",
  )
  return solver_function(Algorithm"exponentiate"())
end

function tdvp_solver(
  f::typeof(exponentiate);
  ishermitian,
  issymmetric,
  solver_tol,
  solver_krylovdim,
  solver_maxiter,
  solver_outputlevel,
)
  function solver(H, t, psi0; current_time, outputlevel)
    psi, info = f(
      H,
      t,
      psi0;
      ishermitian,
      issymmetric,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_outputlevel,
      eager=true,
    )
    return psi, info
  end
  return solver
end

function itensortdvp_tdvp(
  H,
  t::Number,
  psi0::MPS;
  ishermitian=default_ishermitian(),
  issymmetric=default_issymmetric(),
  solver_backend=default_tdvp_solver_backend(),
  solver_function=solver_function(solver_backend),
  solver_tol=default_solver_tol(solver_function),
  solver_krylovdim=default_solver_krylovdim(solver_function),
  solver_maxiter=default_solver_maxiter(solver_function),
  solver_outputlevel=default_solver_outputlevel(solver_function),
  kwargs...,
)
  return itensortdvp_tdvp(
    tdvp_solver(
      solver_function;
      ishermitian,
      issymmetric,
      solver_tol,
      solver_krylovdim,
      solver_maxiter,
      solver_outputlevel,
    ),
    H,
    t,
    psi0;
    kwargs...,
  )
end

function itensortdvp_tdvp(t::Number, H, psi0::MPS; kwargs...)
  return itensortdvp_tdvp(H, t, psi0; kwargs...)
end

function itensortdvp_tdvp(H, psi0::MPS, t::Number; kwargs...)
  return itensortdvp_tdvp(H, t, psi0; kwargs...)
end

"""
    tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)
    tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*psi0` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of H.

Returns:
* `psi::MPS` - time-evolved MPS

Optional keyword arguments:
* `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
* `observer` - object implementing the [Observer](@ref observer) interface which can perform measurements and stop early
* `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
"""
function itensortdvp_tdvp(solver, H::MPO, t::Number, psi0::MPS; kwargs...)
  return alternating_update(solver, H, t, psi0; kwargs...)
end

function itensortdvp_tdvp(solver, t::Number, H, psi0::MPS; kwargs...)
  return itensortdvp_tdvp(solver, H, t, psi0; kwargs...)
end

function itensortdvp_tdvp(solver, H, psi0::MPS, t::Number; kwargs...)
  return itensortdvp_tdvp(solver, H, t, psi0; kwargs...)
end

"""
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number; kwargs...)
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number, sweeps::Sweeps; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*psi0` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of H.

This version of `tdvp` accepts a representation of H as a
Vector of MPOs, Hs = [H1,H2,H3,...] such that H is defined
as H = H1+H2+H3+...
Note that this sum of MPOs is not actually computed; rather
the set of MPOs [H1,H2,H3,..] is efficiently looped over at
each step of the algorithm when optimizing the MPS.

Returns:
* `psi::MPS` - time-evolved MPS
"""
function itensortdvp_tdvp(solver, Hs::Vector{MPO}, t::Number, psi0::MPS; kwargs...)
  return alternating_update(solver, Hs, t, psi0; kwargs...)
end
