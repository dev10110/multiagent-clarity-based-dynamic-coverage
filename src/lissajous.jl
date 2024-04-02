module LissajousController

using StaticArrays, RecipesBase, ForwardDiff, LinearAlgebra



struct Lissajous{V}
  A::V
  ω::V
  ϕ::V
  o::V
end

const default_lissajous = Lissajous(
  @SVector[12.0, 8.5], # A
  @SVector[1, 7/4]/12, # ω
  @SVector[0.0, π/2], # ϕ
  @SVector[470.5, 5267.], # o
)

# evaluate method
function (l::Lissajous)(t)

  p = @SVector [l.o[i] + l.A[i] * sin(l.ω[i] * t + l.ϕ[i]) for i=1:2]

  return p

end

function controller(t, x; lissa, k=1.0, vmax=30 * 60 / 1000, kwargs...) 

  # desired location and velocity
  p = lissa(t)
  v = ForwardDiff.derivative(lissa, t)  

  # commanded velocity 
  u = v + k * (p - x)

  if norm(u) > vmax
    u = vmax * normalize(u)
  end

  return u

end



@recipe function f(l::L, tmin=0, tmax=2*π*prod(l.ω); length=1000) where {L <: Lissajous}
  
  ps = [l(t) for t in range(tmin, tmax, length=length)]

  @series begin
    [p[1] for p in ps], [p[2] for p in ps]
  end

end

end