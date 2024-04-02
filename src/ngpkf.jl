module NGPKF

# Numerical Gaussian Process Kalman Filter


using LinearAlgebra, Distributions, RecipesBase, StaticArrays

import ..KF

abstract type AbstractKernel end;

mutable struct MaternKernel{F} <: AbstractKernel
  σ::F
  l::F
end

mutable struct SEKernel{F} <: AbstractKernel
  σ::F
  l::F
end

function (k::MaternKernel)(x1, x2)

  d = norm(x1-x2)
  if d > 20 * k.l
    return 0.0
  else
    return k.σ^2 * exp( - d / k.l )
  end
end

function (k::SEKernel)(x1, x2)
  d = norm(x1-x2)
  # if d > 10 * k.l
  #   return 0.0
  # else
    return k.σ^2 * exp( - 1/2 * d^2 / k.l^2 )
  # end
end


struct NGPKFGrid{P, G, M, K}
  xs::P
  ys::P
  grid_pts::G
  kernel::K
  invK::M
end

function NGPKFGrid(xs, ys, kern)
  grid_pts = vec([@SVector [x, y] for x in xs, y in ys])
  K_grid = kernel_matrix(kern, grid_pts)
  iK = inv(K_grid)
  return NGPKFGrid(xs, ys, grid_pts, kern, iK)
end


function confint(σ, β=0.95)
  d = Normal(0, σ)
  return quantile(d, 1 / 2 + β / 2)
end

# function plot_cov!(mygrid, m, P; kwargs...)
#   σs = sqrt.(diag(P))
#   rs = confint.(σs)
#   plot!(mygrid, m, ribbon=rs; kwargs...)
# end

function kernel_matrix(kern, Xa, Xb)
  N = length(Xa)
  M = length(Xb)
  K = zeros(N, M)
  for i = 1:N, j = 1:M
    K[i, j] = kern(Xa[i], Xb[j])
  end
  return K
end

function kernel_matrix(kern, X)
  N = length(X)
  K = zeros(N, N)
  for i = 1:N, j = i:N
    K[i, j] = kern(X[i], X[j])
  end
  return Hermitian(K)
end

function initialize(mygrid::G) where {G <: NGPKFGrid}

  Σ = inv(mygrid.invK)
  N = length(mygrid.grid_pts)

  s = KF.KFState(;μ = zeros(N), Σ=Σ)

  return s
end

function predict(mygrid::G, s::S; σ_process=0, Q_process = σ_process^2 * I) where {G <: NGPKFGrid, S <: KF.KFState}

  A = I

  s_pred = KF.predict(s, A, Q_process)

  return s_pred

end


function correct(mygrid::G, s::S, ps, ys; σ_meas=0, R_measure = σ_meas^2 * I(length(ys))) where {G <: NGPKFGrid, S <: KF.KFState}

  length(ps) == length(ys) || throw(DimensionMismatch("len(ps) = $(length(ps)) != len(ys) = $(length(ys))"))

  # calculate kernels
  iK_f_f_t_t = mygrid.invK
  K_fy_f_t_t = kernel_matrix(mygrid.kernel, ps, mygrid.grid_pts)
  K_fy_fy_t_t = kernel_matrix(mygrid.kernel, ps)
  K_f_fy_t_t = K_fy_f_t_t'

  # calculate matrices
  Ct = K_fy_f_t_t  *  iK_f_f_t_t
  Rt  = Hermitian( K_fy_fy_t_t - Ct * K_f_fy_t_t  + R_measure )

  # return the KF correction
  s_new = KF.correct(s, ys, Ct, Rt)

  return s_new 

end

"""
add a fast method for diagonal of a cholesky matrix
"""
function LinearAlgebra.diag(M::Cholesky{T}) where {T}
  N = size(M, 1)
  z = zeros(T, N)
  U = M.U
  for i=1:N
      for j=1:i
      z[i] += U[j, i]^2
      end
  end
  return z
end


@recipe function f(g::G, s::S; plotstd = false, plot_max=false, plot_min=false, plot_safe=false, β=0.95) where {G <: NGPKFGrid, S <: KF.KFState}

  Nx, Ny = length(g.xs), length(g.ys)

  if plotstd
    σ = KF.σ(s)
    M = reshape(σ, Nx, Ny)
  elseif plot_max
    μ = KF.μ(s)
    σ = KF.σ(s)
    m = confint.(σ, β)
    M = reshape(μ + m, Nx, Ny)
  elseif plot_min
    μ = KF.μ(s)
    σ = KF.σ(s)
    m = confint.(σ, β)
    M = reshape(μ - m, Nx, Ny)
  elseif plot_safe
    μ = KF.μ(s)
    σ = KF.σ(s)
    m = confint.(σ, β)
    M = reshape(-5 .<= μ - m .<= μ + m .<= 5, Nx, Ny) 
  else
    M = reshape(KF.μ(s), Nx, Ny)
  end

  @series begin

    seriestype --> :heatmap
    xlabel --> "x [km]" 
    ylabel --> "y [km]" 

    g.xs, g.ys, M'

  end



end


function clarity_map(g::G, wx::S, wy::S) where {G <: NGPKFGrid, S <: KF.KFState}

  Nx, Ny = length(g.xs), length(g.ys)

  σx = KF.σ(wx)
  σy = KF.σ(wy)

  q = similar(σx)
  for i=eachindex(q)
    q[i] = clarity(σx[i], σy[i])
  end

  return reshape(q, Nx, Ny)

end

function clarity(σ)
  return 1 / (1 + σ^2)
end

function clarity(σx, σy) 
  # assume a diagonal matrix P = [σx^2; σy^2]
  # q(P) = 1 / (1 + det(P))
  detP = (σx * σy)^2
  return 1 / (1 + detP)
end








end


