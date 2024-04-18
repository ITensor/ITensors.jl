module ITensorsObserversExt

using Observers: Observers
using Observers.DataFrames: AbstractDataFrame
using ITensors.ITensorMPS: ITensorMPS

function ITensorMPS.update_observer!(observer::AbstractDataFrame; kwargs...)
  return Observers.update!(observer; kwargs...)
end

end
