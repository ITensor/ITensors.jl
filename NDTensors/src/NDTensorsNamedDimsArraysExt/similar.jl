# I tried putting this inside the `NamedDimsArrays` module
# but for some reason it kept overloading `Base.similar`.
# NDTensors.similar
similar(a::NamedDimsArrays.AbstractNamedDimsArray) = Base.similar(a)
similar(a::NamedDimsArrays.AbstractNamedDimsArray, elt::Type) = Base.similar(a, elt)
