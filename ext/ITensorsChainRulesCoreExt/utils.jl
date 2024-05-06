using ChainRulesCore: AbstractZero, NoTangent
using Compat: Returns
using ITensors: ITensors

ITensors.dag(z::AbstractZero) = z

map_notangent(a) = map(Returns(NoTangent()), a)
