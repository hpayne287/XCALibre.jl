struct DESKOmega{S1,S2,S3,F1,F2,F3,C,C2,Y} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
    Tu::C2
    y::Y
end
Adapt.@adapt_structure DESKOmega

struct DESKOmegaModel{E1,E2,D,S}
    k_eqn::E1 
    ω_eqn::E2
    Δ::D 
    magS::S
end 
Adapt.@adapt_structure DESKOmegaModel

#Model API Constructor 
DES{KωSmagorinsky}(; β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5, C=0.15) = begin
    coeffs = (β⁺=β⁺, α1=α1, β1=β1, σk=σk, σω=σω, C=C)
    ARG = typeof(coeffs)
    DES{KωSmagorinsky,ARG}(coeffs)
end

# Function as constructor
(rans::DES{KωSmagorinsky,ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = rans.args
    KωSmagorinsky(k,omega,nut,kf,omegaf,nutf,coeffs)
end

