using DifferentialEquations
using OrdinaryDiffEq:
    ODEProblem,
    solve,
    ImplicitEuler,
    ImplicitMidpoint,
    Midpoint,
    Euler,
    SSPRK33,
    NLNewton,
    NLAnderson
using LinearSolve: KrylovJL_GMRES
using ModelingToolkit
using BenchmarkTools
using SparsityDetection, SparseArrays

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
    S_l::FT = (ϑ_l_safe - θr) / (porosity - θr)
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

const ft = Float64;

ν = ft(0.495);
Ksat = ft(0.0443 / 3600 / 100); # m/s
S_s = ft(1e-3); #inverse meters
vg_n = ft(1.4);
vg_α = ft(2.6); # inverse meters
vg_m = ft(1) - ft(1) / vg_n;
θ_r = ft(0.124);

zmax = ft(0);
zmin = ft(-1.5);
const n = 150;
const Δz = (zmax-zmin)/n
const z = Array((zmin+Δz/ft(2.0)):Δz:(zmax-Δz/ft(2.0)))


const p = [ν,vg_α, vg_n, vg_m, Ksat, S_s, θ_r];
const dt = 1.0
Yprev =  ft(0.24) .+ zeros(n)# in this case, the initial condition


#dY/dt = f(Y)
#Y^{n+1}-Y^n = f(Y^{n+1})*dt = dY^{n+1}
# F = Y-dY-yprev = 0


function f!(F, Y)
    ν,
    vg_α,
    vg_n,
    vg_m,
    Ksat,
    S_s,
    θ_r = p
    K =  hydraulic_conductivity.(Ksat,vg_m,
                                 effective_saturation.(ν, Y, θ_r))
    ψ = pressure_head.(vg_α,vg_n,vg_m, θ_r,Y, ν, S_s) .+ z
    
    top = -0.5*(Ksat+K[end])*(0.0-ψ[end])/Δz
    @inbounds for i in 1:1:n
        ip1, im1 = i+1, i-1
        Fi_ph::ft = ip1 == n+1 ? ft.(top) : ft(-2.0)/(ft(1.0)/K[ip1]+ft(1.0)/K[i])*(ψ[ip1]-ψ[i])/Δz
        Fi_mh::ft = im1  == 0 ? ft.(-K[1]) : ft(-2.0)/(ft(1.0)/K[i]+ft(1.0)/K[im1])*(ψ[i]-ψ[im1])/Δz
        # dY = -ft(1.0)/Δz*(Fi_ph-Fi_mh)
        F[i] = Y[i] -(-ft(1.0)/Δz*(Fi_ph-Fi_mh) * dt) -Yprev[i]
    end

end


# then I need the Jacobian for Newton, MP, P

function j_newton!(J,Y)
    J .= 0.0
    ν,
    vg_α,
    vg_n,
    vg_m,
    Ksat,
    S_s,
    θ_r = p
    K =  hydraulic_conductivity.(Ksat,vg_m,
                                 effective_saturation.(ν, Y, θ_r))
    ψ = pressure_head.(vg_α,vg_n,vg_m, θ_r,Y, ν, S_s) .+ z
    
    top = -0.5*(Ksat+K[end])*(0.0-ψ[end])/Δz
    
    @inbounds for i in 1:1:n
        ip1, im1 = i+1, i-1
        J[i,i] = X
        J[i,ip1] = Y
        J[i,im1] = Z
    end
end

            
            
