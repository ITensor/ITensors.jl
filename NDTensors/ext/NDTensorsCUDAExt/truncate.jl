# import NDTensors:truncate!
# function NDTensors.truncate!(P::CuArray{ElT}; kwargs...)::Tuple{ElT,ElT} where {ElT}
#     p_cpu = NDTensors.cpu(P)
#     truncerr, docut = NDTensors.truncate!(p_cpu)
#     P = typeof(P)(p_cpu)
#     return truncerr, docut
# end