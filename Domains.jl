module Domains

using ClimaCore
import ClimaCore: Meshes, Spaces, Topologies, Geometry

abstract type AbstractVerticalDomain{FT <: AbstractFloat} end

struct Column{FT} <: AbstractVerticalDomain{FT}
    zlim::Tuple{FT, FT}
    nelements::Int32
    boundary_tags::Tuple{Symbol, Symbol}
end

function Column(FT::DataType = Float64; zlim, nelements)
    @assert zlim[1] < zlim[2]
    boundary_tags = (:bottom, :top)
    return Column{FT}(zlim, nelements, boundary_tags)
end


function make_function_space(domain::Column{FT}) where {FT}
    column = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.ZPoint{FT}(domain.zlim[1]),
        ClimaCore.Geometry.ZPoint{FT}(domain.zlim[2]);
        boundary_tags = domain.boundary_tags,
    )
    mesh = Meshes.IntervalMesh(column; nelems = domain.nelements)
    center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

export AbstractVerticalDomain
export Column
export make_function_space

end # module
