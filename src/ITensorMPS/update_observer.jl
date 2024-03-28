using Observers: Observers

function Observers.update!(observer::AbstractObserver; kwargs...)
  return measure!(observer; kwargs...)
end
