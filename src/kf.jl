module KF

## Devansh Agrawal Mar 2024

# implementation of Kevin Tracy A SQUARE-ROOT KALMAN FILTER USING ONLY QR DECOMPOSITIONS
#https://arxiv.org/pdf/2208.06452.pdf

# also should look at https://ntrs.nasa.gov/api/citations/19770005172/downloads/19770005172.pdf


using LinearAlgebra


##################
# Getters/Setters
##################

struct KFState{V, M}
  μ::V # mean estimate of the kalman filter
  F::M # upper triangular cholesky component of the Kalman State
end

function KFState(; μ, Σ)
  return KFState(μ, chol_sqrt(Σ))
end

function μ(s::S) where {S <: KFState}
  return s.μ
end

function Σ(s::S) where {S <: KFState}
  return Cholesky(s.F) # avoids explicitly computing the full matrix, which could save computation
end

function σ(s::S) where {S <: KFState}
  return sqrt.(diag(Σ(s)))
end

##################
# MAIN METHODS 
##################


function kalman_filter(s_k::S, y_kp1, u_k, A, B, C, V, W) where {S <: KFState}

  s_pred = predict(s_k, A, B, u_k, W)
  s_corr = correct(s_pred, y_kp1, C, V)
  return s_corr

end

"""
  s_{k+1} = predict(s_k, A, B, u, W)

implements the prediction rule based on the update dynamics
```math
 x_{k+1} = A x_k + B * u + w, where w ∼ N(0, W)
```
"""
function predict(s::S, A, B, u, W) where {S <: KFState}

  N = length(s.μ)
  Γw = chol_sqrt(0*I(N) + W)

  μ_new = A * S.μ + B * u
  F_new = qrr(s.F * A', Γw)

  return KFState(μ_new, F_new)

end

"""
  s_{k+1} = predict(s_k, A, W)

implements the prediction rule based on the update dynamics
```math
 x_{k+1} = A x_k + w, where w ∼ N(0, W)
```
"""
function predict(s::S, A, W) where {S <: KFState}

  N = length(s.μ)
  Γw = chol_sqrt(0*I(N) + W)

  μ_new = A * s.μ
  F_new = qrr(s.F * A', Γw)

  return KFState(μ_new, F_new)

end

"""
  s_{k+1} = correct(s_k, y, C, V)

implements the correction rule based on the measurement
```math
y_k = C x_k + v, v ∼ N(0, V)
```
"""
function correct(s::S, y, C, V) where {S <: KFState}

  M = length(y)
  Γv = chol_sqrt(0*I(M) + V)

  # innovation
  z = y - C * s.μ

  # kalman gain
  L = kalman_gain(s, C, Γv)

  # update
  μ_new = s.μ + L * z

  # @time sqrtA_ = s.F * (I - L * C)'
  sqrtA_ = s.F + (s.F * C') * (-L')
  sqrtB_ = Γv * L'

  F_new = qrr( sqrtA_ , sqrtB_)

  return KFState(μ_new, F_new)

end


##############
# UTILITIES
##############

"""
  U = chol_sqrt(A)
returns an upper-triangular matrix U such that A = U' * U
"""
function chol_sqrt(A)
  return cholesky(A).U
end

"""
  R = qrr(sqrtA, sqrtB)

returns R = sqrt(A + B)  where A = sqrtA' * sqrtA, and B = sqrtB' * sqrtB. The result is an UpperTriangular matrix.
"""
function qrr(sqrtA, sqrtB)

  
  M = [sqrtA; sqrtB]

  return qrr!(M)
  # C = qr!(M)
  # T = UpperTriangular(C.R)

  # return T 
end

function qrr!(A)
  N = minimum(size(A))
  LinearAlgebra.LAPACK.geqrf!(A)
  return UpperTriangular(A[1:N, 1:N])
end



function kalman_gain(s::S, C, Γv) where {S <: KFState}

  G = qrr(s.F * C', Γv)
  L = ((s.F' * s.F * C') / G) / (G')

  return L

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


end