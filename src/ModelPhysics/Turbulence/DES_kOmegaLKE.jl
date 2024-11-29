struct DESKOmegaLKE{}<:AbstractDESModel
    k::S1
    omega::S2
    kl::S3
    nut::S4
    kf::F1
    omegaf::F2
    klf::F3
    nutf::F4
    coeffs::C1
    Tu::C2
    y::Y
end
Adapt.@adapt_structure DESKOmegaLKE

struct DESKOmegaLKEModel{
    E1,E2,E3,F1,F2,F3,S1,S2,S3,S4,S5,S6,S7,V1,V2}
    k_eqn::E1
    ω_eqn::E2
    kl_eqn::E3
    nueffkLS::F1
    nueffkS::F2
    nueffωS::F3
    nuL::S1
    nuts::S2
    Ω::S3
    γ::S4
    fv::S5
    normU::S6
    Reυ::S7
    ∇k::V1
    ∇ω::V2
end 
Adapt.@adapt_structure DESKOmegaLKEModel

function initialise(
    
)