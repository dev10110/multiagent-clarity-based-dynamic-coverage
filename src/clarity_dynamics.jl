module ClarityDynamics

using SpatiotemporalGPs

const AbstractKernel = SpatiotemporalGPs.STGPKF.AbstractKernel
const Matern12 = SpatiotemporalGPs.STGPKF.Matern12

"""
    create_sensitivity_function(ks, dt, σm)

Consider a spatiotemporal field with spatial kernel ks and temporal kernel kt. Assume the spatial kernel is isotropic.
Suppose measurements are taken every dt, with measurement standard deviation σm, and from a distance d away from a point of interest. 

This function returns the sensing function S(p, x) such that the clarity dynamics are 
```math
dq/dt = S(p, x) (1 - q)^ + W(p, q)
```
"""
function create_sensitivity_function(ks::KS, dt, σm) where {KS <: AbstractKernel}

    function sfunc(p, x)
    
        kxx = ks(x, x)
        kpp = ks(p, p)
        kxp = ks(x, p)

        return -(kxp^2/(dt*(kxp^2 - kpp*(kxx + σm^2))))

    end

    return sfunc

end

"""
    create_decay_function(kt)

Consider a spatiotemporal field with spatial kernel ks and temporal kernel kt. Assume the spatial kernel is isotropic.
Suppose measurements are taken every dt, with measurement standard deviation σm, and from a distance d away from a point of interest. 
This function returns the decay function W(p, q) such that the clarity dynamics are 
```math
dq/dt = S(p, x) (1 - q)^ + W(p, q)
```
"""
function create_decay_function(kt::KM) where {KM <: Matern12 }

    λt = kt.λ
    σt_sq = kt.σsq

    function wfunc(p, q)
        return 2*q*λt*(-1 + q + q*σt_sq)
    end
    return wfunc

end


"""
    create_clarity_dynamics_function(kt)

Consider a spatiotemporal field with spatial kernel ks and temporal kernel kt. Assume the spatial kernel is isotropic.
Suppose measurements are taken every dt, with measurement standard deviation σm, and from a distance d away from a point of interest. 
This function returns the decay function qdot(q) such that the clarity dynamics are 
```math
dq/dt = qdot(q) = S(p, x) (1 - q)^ + W(p, q)
```
""" 
function create_clarity_dynamics_function(ks::KS, kt::KM, dt, σm, d) where {KS <:SpatiotemporalGPs.STGPKF.AbstractKernel, KM <: SpatiotemporalGPs.STGPKF.Matern12}

    p = 0.
    x = d

    sfunc = create_sensitivity_function(ks, dt, σm)
    s0 = sfunc(p, x)

    wfunc = create_decay_function(kt)

    function qdot(q, params, time)
        return s0 * (1-q)^2 - wfunc(p, q)
    end

    return qdot
        
end

"""
    get_ricatti_coeffs(ks, kt, dt, σm, d)

Consider a spatiotemporal field with spatial kernel ks and temporal kernel kt. Assume the spatial kernel is isotropic.
Suppose measurements are taken every dt, with measurement standard deviation σm, and from a distance d away from a point of interest. 
This function returns the coefficients α, β, γ such that the clarity dynamics are 
```math
dq/dt + α q^2 + β q + γ = 0
```
"""
function get_ricatti_coeffs(ks::KS, kt::KT, dt, σm, d) where {KS <: AbstractKernel, KT <: Matern12}

    p = 0
    x = d

    kxx = ks(x, x)
    kpp = ks(p, p)
    kxp = ks(x, p)

    λt = kt.λ
    σt = sqrt(kt.σsq)
    
    α = -(-2*λt+kxp^2/(dt*kpp*(-(kxp^2/kpp)+kxx+σm^2))-2*λt*σt^2)
    β = -(2*λt-(2*kxp^2)/(dt*kpp*(-(kxp^2/kpp)+kxx+σm^2)))
    γ = -(kxp^2/(dt*kpp*(-(kxp^2/kpp)+kxx+σm^2)))

    return α, β, γ

end

"""
    ricatti_solution(t, q0, α, β, γ)

given a ricatti problem
```math
    dq/dt + α q^2 + β q + γ = 0
    q(0) = q_0
```

returns the solution at time t.
"""
function ricatti_solution(t, q0, α, β, γ)

    @assert α != 0
    
    δsq = β^2 - 4*α*γ
    δ = sqrt(δsq)
    ρ0 = β - δ + 2*α*q0

    return (1 / (2*α) ) * ( -β + δ + (2*δ*ρ0) / ( (2*δ + ρ0)*exp(δ*t) - ρ0) )

end

"""
    ricatti_limit(α, β, γ)
given a ricatti problem
```math
    dq/dt + α q^2 + β q + γ = 0
    q(0) = q_0
```

returns the limit 
```math
q_\\infty = \\lim_{t \\to \\infty} q(t)
```
"""

function ricatti_limit(α, β, γ)

    @assert α != 0
    δsq = β^2 - 4*α*γ
    δ = sqrt(δsq)

    return yinfty = (δ - β)/(2α)
end

"""
    inverse_ricatti_solution(q0, qf, α, β, γ)

given a ricatti problem
```math
    dq/dt + α q^2 + β q + γ = 0
    q(0) = q_0
    q(t) = q_f
```

returns the time t such that q(t) = qf. 
"""
function inverse_ricatti_solution(q0, qf, α, β, γ)

    @assert (0 <= q0 <= 1)
    @assert (0 <= qf <= 1)
    @assert α != 0
    
    δ = sqrt(β^2 - 4*α*γ)

    # check if too large
    qinfty = ricatti_limit(α, β, γ)

    if q0 < qinfty <= qf
        return Inf
    end

    if qf < qinfty <= q0
        return Inf
    end        

    ρ0 = β - δ + 2*α*q0
    ρf = β - δ + 2*α*qf

    return (1/δ) * log( (ρ0 * (2δ + ρf) ) / ( ρf * (2δ + ρ0) ) )
end

function clarity_solution(t, q0, ks::KS, kt::KT, dt, σm, d) where {KS <: AbstractKernel, KT <: Matern12}
    α, β, γ = get_ricatti_coeffs(ks, kt, dt, σm, d)
    return ricatti_solution(t, q0, α, β, γ)
end

function clarity_limit(ks::KS, kt::KT, dt, σm, d) where {KS <: AbstractKernel, KT <: Matern12}
    α, β, γ = get_ricatti_coeffs(ks, kt, dt, σm, d)
    return ricatti_limit(α, β, γ)
end

function time_to_reach_clarity(q0, ks::KS, kt::KT, dt, σm, d) where {KS <: AbstractKernel, KT <: Matern12}

    α, β, γ = get_ricatti_coeffs(ks, kt, dt, σm, d)
    return inverse_ricatti_solution(q0, qf, α, β, γ)

end


end
