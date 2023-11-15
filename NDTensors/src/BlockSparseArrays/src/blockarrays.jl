# Extensions to BlockArrays.jl
blocktuple(b::Block) = Block.(b.n)
inttuple(b::Block) = b.n
