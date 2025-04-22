# using Plots
using XCALibre
using Alert


# grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid1 = "flatplate_2D_lowRe.unv"
# grid1 = "bfs_unv_tet_10mm.unv"
# mesh_file = joinpath(grids_dir, grid1)

mesh_file = "C:/Users/Hudson/OneDrive - The University of Nottingham/Year 3/Individual Project/Code/Meshes/2025-04-20-Case20Mesh2D.unv"

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=32)

mesh_dev = mesh

velocity = [5.4, 0.0, 0.0]
noSlip = [0.0,0.0,0.0]
nu = 1.5e-5
# Re = 23000
k_inlet = 2.916e-5 
ω_inlet = 3.1764 

kBC = (Dirichlet(:inlet, k_inlet),
Neumann(:outlet, 0.0),
KWallFunction(:plate1),
KWallFunction(:plate2),
Neumann(:top1,0.0),
Neumann(:top2,0.0))

ωBC = (Dirichlet(:inlet, ω_inlet),
Neumann(:outlet, 0.0),
OmegaWallFunction(:plate1),
OmegaWallFunctionWallFunction(:plate2),
Neumann(:top1,0.0),
Neumann(:top2,0.0))

nutBC = (Dirichlet(:inlet, k_inlet/ω_inlet),
Neumann(:outlet, 0.0),
NutWallFunction(:plate1),
NutWallFunction(:plate2),
Neumann(:top1,0.0),
Neumann(:top2,0.0))

model = Physics(
    time=Transient(),
    fluid=Fluid{Incompressible}(nu=nu),
    turbulence=DES{Hybrid}(walls=(:plate1,:plate2),blendType=MenterF1(),kBC=kBC,ωBC=ωBC,nutBC=nutBC),
    energy=Energy{Isothermal}(),
    domain=mesh_dev
)


#region Define BC
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:plate1, [0.0, 0.0, 0.0]),
    Wall(:plate2, [0.0, 0.0, 0.0]),
    Neumann(:top1,0.0),
    Neumann(:top2,0.0),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Wall(:plate1, 0.0),
    Wall(:plate2, 0.0),
    Neumann(:top1,0.0),
    Neumann(:top2,0.0),
)
@assign! model turbulence k (kBC)

@assign! model turbulence omega (ωBC)

@assign! model turbulence nut (nutBC)
#endregion

schemes = (
    U=set_schemes(divergence=Linear,time=Euler),
    p=set_schemes(time=Euler),
    y = set_schemes(gradient=Midpoint,time=Euler),
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
    y = set_solver(
        model.turbulence.y;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.9,
        itmax = 2000
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

runtime = set_runtime(iterations=1, time_step=0.0000001, write_interval=1) #Adjust timestep to get a decent courant value

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


zeroV = [0.0,0.0,0.0]
initialise!(model.momentum.U, velocity) 
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, 0.0)

residuals = run!(model, config);

# alert("Simulation Done!!");