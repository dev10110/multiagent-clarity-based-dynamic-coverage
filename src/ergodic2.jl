module Ergodic2



using FFTW, LinearAlgebra, StaticArrays
using RecipesBase, ForwardDiff

struct Grid{T, F}
    o::T
    dx::T # spacing of each cell 
    N::Tuple{Int, Int} # total extent of the simulation domain
    dct_plan::F
end

function Grid(o::T, dx::T, L::T) where {T}
    Ns = ntuple(i->Int(ceil(1 + L[i] / (dx[i]))), 2)
    return Grid(o, dx, Ns)
end

function Grid(o, dx, N)
        M = zeros(N)
        dct_plan = FFTW.plan_r2r(M, FFTW.REDFT10)

    os = @SVector [o[1], o[2]]
    dxs = @SVector [dx[1], dx[2]]
    Grid(os, dxs, N, dct_plan)
end

function lengths(grid::G) where {G <: Grid}
    return @SVector [ (grid.N[i]-1) * grid.dx[i] for i=1:2]
end

function center(grid::G) where {G <: Grid}
    return grid.o + 0.5 * lengths(grid)
end

function xs(grid::G) where {G <: Grid}
    return grid.o[1] .+ (0:(grid.N[1]-1)) * grid.dx[1]
end

function ys(grid::G) where {G <: Grid}
    return grid.o[2] .+ (0:(grid.N[2]-1)) * grid.dx[2]
end

@inline function pos2ind(grid::G, pos) where {G <: Grid}
    return CartesianIndex( (1 + Int( floor( (pos[i] - grid.o[i]) / grid.dx[i] )) for i=1:2 )...)
        # CartesianIndex(1, 1) + CartesianIndex(Int.(round.((pos - grid.o) ./ grid.dx))...)
end

@inline function ind2pos(grid, ind)
    return @SVector [grid.o[i] + (ind[i] - 1) * grid.dx[i] for i=1:2]
end

function fill!(f, grid, M)
    for ind in CartesianIndices(M)
        pos = ind2pos(grid, ind)
        M[ind] = f(pos)
    end
end

@inline normsq(x) = mapreduce(abs2, sum, x)

@inline hk(k) = k==0 ? 1 : 1/sqrt(2)
@inline hk(k1, k2) = hk(k1) * hk(k2)

@inline Λ(k; d=length(k), s=(1+d)/2)  = (1 + normsq(k))^(-s)

@inline Λ2D(k1, k2)  = (1 + k1^2 + k2^2)^(-1.5)


function dct_map(grid, M)

    @assert size(M) == grid.N
    
    N1, N2 = grid.N
    L1, L2 = lengths(grid)

    # normalize!(M)

    # do the un-normalized DCT
    Y = grid.dct_plan * M
    
    # do the normalization
    δ = (L1/N1) * (L2/N2) / (2^2)
    for k1=0:(N1-1), k2=0:(N2-1)
        Y[k1+1, k2+1] *= δ
        if k1==0 || k2==0
            Y[k1+1,k2+1] = Y[k1+1,k2+1]  / hk(k1,k2)
        end
    end
    
    return Y

end



function grad_fk(grid, p, k)
    
  L1, L2 = lengths(grid)
  p1, p2 = (p - grid.o)
  k1, k2 = k
  
  return (1/(hk(k1, k2) * prod(grid.dx)))  * (@SVector [
      -k1 * sin(π * k1 * p1 / L1) * cos(π * k2 * p2 / L2),
      -k2 * cos(π * k1 * p1 / L1) * sin(π * k2 * p2 / L2),
  ]) 
end


# this is a direction vector for single-integrator control
function descent_direction(grid, x, target_clarity, current_clarity, Cfun, Rfun)


  size(target_clarity) == grid.N || throw(DimensionMismatch())
  size(current_clarity) == grid.N || throw(DimensionMismatch())
  
  # L1, L2 = lengths(grid)
  N1, N2 = grid.N

  S(p, x)  = Cfun(p, x)^2 / Rfun(p, x)
  DxS(p, x) = ForwardDiff.gradient(xx-> S(p, xx), x)

  Bx = zeros(N1, N2)
  By = zeros(N1, N2)
  for i=1:N1, j=1:N2
    p = ind2pos(grid, CartesianIndex(i, j))
    q = current_clarity[i, j]
    Dx1S, Dx2S = DxS(p, x)
    Bx[i, j] = (1 - q)^2 * Dx1S
    By[i, j] = (1 - q)^2 * Dx2S
  end

  # compute Bhat
  Bxhat = dct_map(grid, Bx)
  Byhat = dct_map(grid, By)

  # compute the clarity dct
  dct_target_minus_current = dct_map(grid, target_clarity - current_clarity)

  # compute the Lk
  Lx = 0
  Ly = 0

  for i=1:N1, j=1:N2
    k1 = (i-1)
    k2 = (j-1)
    s = Λ2D(k1, k2) * dct_target_minus_current[i, j]
    Lx += s * Bxhat[i,j]
    Ly += s * Byhat[i,j]
  end
  
  return @SVector [Lx, Ly]
end


function boundary_correction(grid, p, u; α=1)

    # decompose the control input
    u1, u2 = u

    lower = grid.o
    upper = grid.o + lengths(grid)
        
    # h1 constraint
    # h1dot(x, u) ≥ -α(h1)
    # ∴ u ≥ - α * h1
    h1 = p[1] - lower[1]
    ux_min = -α * h1

    # h2 constraint
    # h2dot(x, u) ≥ -α(h2)
    # -u ≥  -α * h2
    # u ≤ α * h2
    h2 = upper[1] - p[1]
    ux_max = α * h2

    # h3 constraint
    # h3dot(x, u) ≥ -α(h3)
    # ∴ u ≥ - α * h3
    h3 = p[2] - lower[2]
    uy_min = -α * h3

    # h2 constraint
    # h2dot(x, u) ≥ -α(h2)
    # -u ≥  -α * h2
    # u ≤ α * h2
    h4 = upper[2] - p[2]
    uy_max = α * h4

    # do the corrections

    ux = clamp(u1, ux_min, ux_max)
    uy = clamp(u2, uy_min, uy_max)
    
    return @SVector [ux, uy]
end


function controller_single_integrator(grid, p, target_clarity, current_clarity, Cfun, Rfun; umax=1.0, do_boundary_correction=true)
    
  b_ergo = descent_direction(grid, p, target_clarity, current_clarity, Cfun, Rfun)

  u_ergo = umax * normalize(b_ergo)

  if do_boundary_correction
      return boundary_correction(grid, p, u_ergo)
  else
      return u_ergo
  end
   
end




end