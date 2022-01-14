module Domains

using ClimaCore
using IntervalSets
import ClimaCore: Meshes, Spaces, Topologies, Geometry
abstract type AbstractDomain{FT <: AbstractFloat} end

struct Rectangle{FT} <: AbstractDomain{FT}
    xlim::Tuple{FT, FT}
    zlim::Tuple{FT, FT}
    nelements::Tuple{Int32,Int32}
    xboundary_tags::Tuple{Symbol, Symbol}
    zboundary_tags::Tuple{Symbol, Symbol}
end

function Rectangle(FT::DataType = Float64; xlim, zlim, nelements)
    @assert zlim[1] < zlim[2]
    @assert xlim[1] < xlim[2]
    zboundary_tags = (:bottom, :top)
    xboundary_tags = (:xmin, :xmax)
    return Rectangle{FT}(xlim, zlim, nelements, xboundary_tags, zboundary_tags)
end

function make_function_space(domain::Rectangle{FT}) where {FT}
    rectangle = ClimaCore.Domains.RectangleDomain(
        Geometry.XPoint{FT}(domain.xlim[1])..Geometry.XPoint{FT}(domain.xlim[2]),
        Geometry.ZPoint{FT}(domain.zlim[1])..Geometry.ZPoint{FT}(domain.zlim[2]),
        x1periodic = false,
        x2periodic = false,
        x1boundary = domain.xboundary_tags,
        x2boundary = domain.zboundary_tags,
    )
    mesh = Meshes.RectilinearMesh(rectangle, domain.nelements[1], domain.nelements[2])
    center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    return center_space, face_space
en


struct Column{FT} <: AbstractDomain{FT}
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
