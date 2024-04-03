function update_observer!(observer; kwargs...)
  return error("Not implemented")
end

function update_observer!(observer::AbstractObserver; kwargs...)
  return measure!(observer; kwargs...)
end
