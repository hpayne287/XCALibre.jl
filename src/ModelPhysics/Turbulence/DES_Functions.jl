"""
    menterF1!(des::HybridModel, model::Physics)

Set the values of the `blendWeight` field according to the Menter F1 equation

### Input
- `des` -- turbulence model.
- `model` -- Physics model defined by user.
"""
function menterF1!(des::HybridModel, model::Physics)
    (; rho) = model.fluid
    (; k, omega, nut, blendWeight, CDkw, y) = model.turbulence
    (; βstar, σω2) = model.turbulence.coeffs
    (; ω_eqn) = des

    dkdomegadx = get_source(ω_eqn, 2)

    @. CDkw.values = max((2 * rho.values * σω2 * (1 / omega.values) * dkdomegadx.values), 10e-20)
    @. blendWeight.values = tanh(min(max(sqrt(k.values) / (βstar * y.values * omega.values),
            (500 * nut.values) / (y.values^2 * omega.values)),
        (4 * rho.values * σω2 * k.values) / (CDkw.values * y.values^2))^4)

end

function cross_diffusion!(des, model, config)

    (; ω_eqn, ∇k, ∇ω) = des
    (; σd) = model.turbulence.coeffs
    (; k, omega, kf, omegaf, nutf) = model.turbulence

    dkdomegadx = get_source(ω_eqn, 2)

    interpolate!(kf, k, config)
    correct_boundaries!(nutf, k, k.BCs, time, config)
    interpolate!(omegaf, omega, config)
    correct_boundaries!(nutf, omega, omega.BCs, time, config)
    grad!(∇ω, omegaf, omega, omega.BCs, time, config)
    grad!(∇k, kf, k, k.BCs, time, config)
    inner_product!(dkdomegadx, ∇k, ∇ω, config)
    @. dkdomegadx.values = max((σd / omega.values) * dkdomegadx.values, 0.0)
end

function blend_nut!(nut,blend,nutRANS,nutLES)
    @. nut.values = (blend.values * nutRANS.values) + ((1 - blend.values) * nutLES.values)
end