using .Expose: Exposed, unexpose

# TODO: Move to `Expose` module.
# Create the Exposed version of Base.permutedims
function permutedims(E::Exposed{<:Array}, perm)
    ## Creating Mperm here to evaluate the permutation and
    ## avoid returning a Stridedview
    @strided Mperm = permutedims(unexpose(E), perm)
    return Mperm
end

function permutedims!(Edest::Exposed{<:Array}, Esrc::Exposed{<:Array}, perm)
    a_dest = unexpose(Edest)
    a_src = unexpose(Esrc)
    @strided a_dest .= permutedims(a_src, perm)
    return a_dest
end

function permutedims!(Edest::Exposed{<:Array}, Esrc::Exposed{<:Array}, perm, f)
    a_dest = unexpose(Edest)
    a_src = unexpose(Esrc)
    @strided a_dest .= f.(a_dest, permutedims(a_src, perm))
    return a_dest
end
