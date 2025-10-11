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
end
