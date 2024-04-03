module ITensorsObserversExt

using ITensors.ITensorMPS: ITensorMPS
using Observers: Observers
using Observers.DataFrames: AbstractDataFrame

function ITensorMPS.update_observer!(observer::AbstractDataFrame; kwargs...)
  return Observers.update!(observer; kwargs...)
end

end
