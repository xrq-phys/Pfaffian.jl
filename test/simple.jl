using LinearAlgebra
using DelimitedFiles

A = LowerTriangular(rand(6, 6))
A = A - A'
A_= copy(A)
n₁, n₂ = size(A)

for i = 1:n₁-2
    # [ 0  .  .  . ; s: same
    #  (α) 0  .  . ; u: updated.
    #   |  s  0  . ;
    #   |  s  u  0 ]
    #
    αₖ = view(A, :, i)
    # A[:, i+1] doesn't change after each update.
    aₖ = view(A, :, i+1)
    αₖ[i+2:end] ./= αₖ[i+1]
    # Should zeroize αₖ[1:i+1] .= 0.0 , but we use the same space to store T.

    # L[i+2:end, i+1] = αₖ[i+2:end]
    # is saved to A[i+2:end, i] <<< NB: i not i+1 here.
    #
    A[i+2:end, i+2:end] += αₖ[i+2:end] * aₖ[i+2:end]' - aₖ[i+2:end] * αₖ[i+2:end]';
    @info "A="
    writedlm(stdout, A)
end 

vT = [A[i+1, i] for i=1:n₁-1]
pfA = prod(vT[1:2:end])
@show pfA
@show det(A_'A_)^0.25

Lcore = LowerTriangular(A[2:end, 1:n₂-1])
for i = 1:n₂-1
    Lcore[i, i] = 1.0
end
L = [ 1            zeros(n₂-1)';
      zeros(n₂-1)  Lcore ]
writedlm(stdout, (x -> if abs(x) > 1e-8 x else 0 end).(A_ - L * diagm(-1 => vT, 1 => -vT) * L'))

