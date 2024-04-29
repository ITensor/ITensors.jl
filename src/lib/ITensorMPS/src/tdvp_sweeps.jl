function process_sweeps(s::Sweeps)
  return (;
    nsweeps=s.nsweep, maxdim=s.maxdim, mindim=s.mindim, cutoff=s.cutoff, noise=s.noise
  )
end

function tdvp(H, t::Number, psi0::MPS, sweeps::Sweeps; kwargs...)
  return tdvp(H, t, psi0; process_sweeps(sweeps)..., kwargs...)
end

function tdvp(solver, H, t::Number, psi0::MPS, sweeps::Sweeps; kwargs...)
  return tdvp(solver, H, t, psi0; process_sweeps(sweeps)..., kwargs...)
end

function dmrg(H, psi0::MPS, sweeps::Sweeps; kwargs...)
  return dmrg(H, psi0; process_sweeps(sweeps)..., kwargs...)
end
