module Richards
using ClimaCore: Fields, Operators, Geometry
using UnPack
export rhs_flux!

function volumetric_liquid_fraction(ϑ_l::FT, ν_eff::FT) where {FT}
    if ϑ_l < ν_eff
        θ_l = ϑ_l
    else
        θ_l = ν_eff
    end
    return θ_l
end

function matric_potential(α::FT, n::FT, m::FT, S::FT) where {FT}
    ψ_m = -((S^(-FT(1) / m) - FT(1)) * α^(-n))^(FT(1) / n)
    return ψ_m
end

function effective_saturation(porosity::FT, ϑ_l::FT, θr::FT) where {FT}
    ϑ_l_safe = max(ϑ_l, θr + eps(FT))
    S_l = (ϑ_l_safe - θr) / (porosity - θr)
    return S_l
end

function pressure_head(
    α::FT,
    n::FT,
    m::FT,
    θ_r::FT,
    ϑ_l::FT,
    ν_eff::FT,
    S_s::FT,
) where {FT}
    S_l_eff = effective_saturation(ν_eff, ϑ_l, θ_r)
    if S_l_eff <= FT(1.0)
        ψ = matric_potential(α,n,m, S_l_eff)
    else
        ψ = (ϑ_l - ν_eff) / S_s
    end
    return ψ
end

function hydraulic_conductivity(
    Ksat::FT,
    m::FT,
    S::FT
) where {FT}
    if S < FT(1)
        K = sqrt(S) * (FT(1) - (FT(1) - S^(FT(1) / m))^m)^FT(2)
    else
        K = FT(1)
    end
    return K* Ksat
end

function rhs_flux!(dY, Y, p, t)
    @unpack ν, vg_α, vg_n, vg_m, Ksat, S_s, θ_r, top_flux_bc, bot_flux_bc, z = p
    interpc2f = Operators.InterpolateC2F()
    gradc2f_water = Operators.GradientC2F()
    divf2c_water = Operators.DivergenceF2C(
        top = Operators.SetValue(
            Geometry.WVector(top_flux_bc),
        ),
        bottom = Operators.SetValue(
            Geometry.WVector(bot_flux_bc),
        ),
    )
    @. dY.ϑ_l = -(
        divf2c_water(
            -interpc2f(
                hydraulic_conductivity(Ksat,
                                       vg_m,
                                       effective_saturation(ν, Y.ϑ_l, θ_r)
                                       )
            )
            * gradc2f_water(
                pressure_head(vg_α,vg_n,vg_m, θ_r,Y.ϑ_l, ν, S_s) + z
            )
        )
    )
end


end # module
