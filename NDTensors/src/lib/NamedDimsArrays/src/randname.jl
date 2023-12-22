using Random: randstring

randname(::Any) = error("Not implemented")

randname(::String) = randstring()
