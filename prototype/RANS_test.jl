# using Plots
using XCALibre
using Alert

using Accessors


# grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid1 = "flatplate_2D_lowRe.unv"
# # grid1 = "bfs_unv_tet_10mm.unv"
# mesh_file = joinpath(grids_dir, grid1)

mesh_file = "C:/Users/Hudson/OneDrive - The University of Nottingham/Year 3/Individual Project/Code/Meshes/2025-04-26-Case31Mesh3D.unv"

mesh = UNV3D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=32)

mesh_dev = mesh

U_mag = 5.4
L = 1.7
velocity = [U_mag, 0.0, 0.0]
noSlip = [0.0,0.0,0.0]
nu = 1.5e-5
Tu = 0.03

# Re = 23000
k_inlet = 3/2*(Tu*U_mag)^2
ω_inlet = (k_inlet^0.5)/(0.09^0.25)*L

model = Physics(
    time=Transient(),
    fluid=Fluid{Incompressible}(nu=nu),
    turbulence=RANS{KOmega}(),
    energy=Energy{Isothermal}(),
    domain=mesh_dev
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Neumann(:top,0.0),
    Neumann(:left,0.0),
    Neumann(:right,0.0),
)

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top,0.0),
    Neumann(:left,0.0),
    Neumann(:right,0.0),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top,0.0),
    Neumann(:left,0.0),
    Neumann(:right,0.0),
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Neumann(:top,0.0),
    Neumann(:left,0.0),
    Neumann(:right,0.0),
)

@assign! model turbulence omega (
    Dirichlet(:inlet, 0.000737),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 1e20),
    Neumann(:top,0.0),
    Neumann(:left,0.0),
    Neumann(:right,0.0),
)


schemes = (
    U=set_schemes(divergence=Linear,time=Euler),
    p=set_schemes(time=Euler),
    k=set_schemes(gradient=Midpoint,time=Euler),
    omega=set_schemes(gradient=Midpoint,time=Euler)
    
)

solvers = (
    U=set_solver(
        model.momentum.U;
        solver=BicgstabSolver,
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=0.7,
        rtol=1e-4,
        atol=1e-10
    ),
    p=set_solver(
        model.momentum.p;
        solver=CgSolver,
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=0.7,
        rtol=1e-4,
        atol=1e-10
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = set_runtime(iterations=4000, time_step=0.000001, write_interval=10) #Adjust timestep to get a decent courant value

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


zeroV = [0.0,0.0,0.0]
initialise!(model.momentum.U, velocity) 
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config);
