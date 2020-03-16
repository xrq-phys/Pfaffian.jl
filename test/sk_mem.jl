using Profile

include("../src/SkewSymmetric.jl")
At = SkewSymmetric.AntiSymmetric(randn(400, 400));
Ar = SkewSymmetric.AntiSymmetric(copy(At));
vA = randn(400, 100);
vB = randn(400, 100);
SkewSymmetric.r2k!(Ar, vA, vB, 1.0, 1.0);
Profile.clear_malloc_data()
SkewSymmetric.r2k!(Ar, vA, vB, 1.0, 1.0);
