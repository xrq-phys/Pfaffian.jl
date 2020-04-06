# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"Module for Parlett-Reid transformation."
module ParlettReid

using LinearAlgebra
using ..SkewSymmetric: AntiSymmetric, r2k!, r2!, ᵀ

"""
    simple(A::AntiSymmetric{T})

Finds Pfaffian and inverse without pivoting.
Not safe. Only use it for debugging.
"""
function simple(A::AntiSymmetric{T}; calc_inv::Bool=false) where {T}
    n₁, n₂ = size(A)

    local M = zero(A) + I # which is actually Mᵀ
    local aₖ = A[:, 1]
    for i = 1:n₁-2
        αₖ = aₖ # A[:, i], this is subscripted k+1 in Wimmer's paper.
        aₖ = A[:, i+1]
        αₖ /= αₖ[i+1]
        αₖ[1:i+1] .= 0.0
        r2!(A, αₖ, aₖ, 1.0, 1.0)
        if calc_inv
            # mₖ = M[i+1, :]
            mₖ = M[:, i+1]
            # M -= αₖ*mₖ'
            M -= mₖ*αₖ'
        end
    end; 
    PfA = prod([A[i, i+1] for i=1:2:7])
    if calc_inv
        PfA, M*inv(copy(A))* ᵀ(M)
    else
        (PfA, )
    end
end

end
