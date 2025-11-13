# Based on https://github.com/JuliaTesting/ExplicitImports.jl/tree/v1.13.2/vendor

using PackageAnalyzer: PackageAnalyzer

deps = [("TypeParameterAccessors", v"0.3.11")]

for (name, version) in deps
    pkg = PackageAnalyzer.find_package(name; version)
    local_path, reachable, _ = PackageAnalyzer.obtain_code(pkg)
    @assert reachable
    p = mkpath(joinpath(@__DIR__, "..", "src", "vendored", name))
    # remove any existing files
    if isdir(p)
        rm(p; recursive = true, force = true)
    end
    mkpath(joinpath(p, "src"))
    cp(joinpath(local_path, "src"), joinpath(p, "src"); force = true)
    mkpath(joinpath(p, "ext"))
    cp(joinpath(local_path, "ext"), joinpath(p, "ext"); force = true)

    for filename in readdir(joinpath(p, "ext"); join = true)
        txt = read(filename, String)
        open(filename, "w") do f
            replacement = "using $name" => "using NDTensors.Vendored.$name"
            return write(f, replace(txt, replacement))
        end
    end
end
