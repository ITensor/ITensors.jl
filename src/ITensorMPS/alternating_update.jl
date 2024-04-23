using ITensors: permute
using NDTensors: scalartype
using Printf: @printf

function _compute_nsweeps(t; time_step=default_time_step(t), nsweeps=default_nsweeps())
  if isinf(t) && isnothing(nsweeps)
    nsweeps = 1
  elseif !isnothing(nsweeps) && time_step != t
    error("Cannot specify both time_step and nsweeps in alternating_update")
  elseif isfinite(time_step) && abs(time_step) > 0 && isnothing(nsweeps)
    nsweeps = convert(Int, ceil(abs(t / time_step)))
    if !(nsweeps * time_step ≈ t)
      error("Time step $time_step not commensurate with total time t=$t")
    end
  end
  return nsweeps
end

function _extend_sweeps_param(param, nsweeps)
  if param isa Number
    eparam = fill(param, nsweeps)
  else
    length(param) == nsweeps && return param
    eparam = Vector(undef, nsweeps)
    eparam[1:length(param)] = param
    eparam[(length(param) + 1):end] .= param[end]
  end
  return eparam
end

function process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)
  return (; maxdim, mindim, cutoff, noise)
end

function alternating_update(
  solver,
  PH,
  t::Number,
  psi0::MPS;
  nsweeps=default_nsweeps(),
  checkdone=default_checkdone(),
  write_when_maxdim_exceeds=default_write_when_maxdim_exceeds(),
  nsite=default_nsite(),
  reverse_step=default_reverse_step(),
  time_start=default_time_start(),
  time_step=default_time_step(t),
  order=default_order(),
  (observer!)=default_observer!(),
  (step_observer!)=default_step_observer!(),
  outputlevel=default_outputlevel(),
  normalize=default_normalize(),
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
)
  nsweeps = _compute_nsweeps(t; time_step, nsweeps)
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
  forward_order = TDVPOrder(order, Base.Forward)
  psi = copy(psi0)
  # Keep track of the start of the current time step.
  # Helpful for tracking the total time, for example
  # when using time-dependent solvers.
  # This will be passed as a keyword argument to the
  # `solver`.
  current_time = time_start
  info = nothing
  for sweep in 1:nsweeps
    if !isnothing(write_when_maxdim_exceeds) && maxdim[sweep] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sweep)), writing environment tensors to disk",
        )
      end
      PH = disk(PH)
    end
    sweep_time = @elapsed begin
      psi, PH, info = sweep_update(
        forward_order,
        solver,
        PH,
        time_step,
        psi;
        nsite,
        current_time,
        reverse_step,
        sweep,
        observer!,
        normalize,
        maxdim=maxdim[sweep],
        mindim=mindim[sweep],
        cutoff=cutoff[sweep],
        noise=noise[sweep],
      )
    end
    current_time += time_step
    update_observer!(step_observer!; psi, sweep, outputlevel, current_time)
    if outputlevel >= 1
      print("After sweep ", sweep, ":")
      print(" maxlinkdim=", maxlinkdim(psi))
      @printf(" maxerr=%.2E", info.maxtruncerr)
      print(" current_time=", round(current_time; digits=3))
      print(" time=", round(sweep_time; digits=3))
      println()
      flush(stdout)
    end
    isdone = false
    if !isnothing(checkdone)
      isdone = checkdone(; psi, sweep, outputlevel)
    elseif observer! isa AbstractObserver
      isdone = checkdone!(observer!; psi, sweep, outputlevel)
    end
    isdone && break
  end
  return psi
end

# Convenience wrapper to not have to specify time step.
# Use a time step of `Inf` as a convention, since TDVP
# with an infinite time step corresponds to DMRG.
function alternating_update(solver, H, psi0::MPS; kwargs...)
  return alternating_update(solver, H, scalartype(psi0)(Inf), psi0; kwargs...)
end

function alternating_update(solver, H::MPO, t::Number, psi0::MPS; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return alternating_update(solver, PH, t, psi0; kwargs...)
end

function alternating_update(solver, Hs::Vector{MPO}, t::Number, psi0::MPS; kwargs...)
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHs = ProjMPOSum(Hs)
  return alternating_update(solver, PHs, t, psi0; kwargs...)
end
