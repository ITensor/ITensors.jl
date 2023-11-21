using Adapt: Adapt, adapt
using NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, named, unname
Adapt.adapt_structure(to, na::AbstractNamedDimsArray) = named(adapt(to, unname(na)), dimnames(na))
