//! Reviewed lane/decode serving bytes and calls; changing them needs a reviewed implementation and receipt.
const std = @import("std");
fn hash(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}
test "dense serving: owned CUDA images retain their reviewed lane and decode bytes" {
    try std.testing.expectEqualStrings("aa13e9f274fe6535a5fa54c3d3a68273435e2a716cb4065c073c55f26f01ec1b", &hash(@embedFile("kernels/cuda/experts.cu")));
    try std.testing.expectEqualStrings("f195f5678edf7d3267ec30e442b1e0048543006e595ad21801afd905df621baf", &hash(@embedFile("kernels/cuda/experts_pack.cu")));
    try std.testing.expectEqualStrings("db560d705729142c66df8dee18fc635f20f09e4f09b8431de265bd6fac68e892", &hash(@embedFile("kernels/cuda/experts_prefill.cu")));
    try std.testing.expectEqualStrings("a6bf09198ffd685f7392d4d433af9739c91d67f7039309d3fd5fe6785f810ab9", &hash(@embedFile("kernels/cuda/gdn.cu")));
    try std.testing.expectEqualStrings("895afd1d40d3f9a0485dcf4a21a7b0605e8f663c5b895b4d4c32573d05780211", &hash(@embedFile("kernels/cuda/lane_gemv.cu")));
    try std.testing.expectEqualStrings("a398a9aa4d257b7a2a9a4301196c369d46bfc636abff9b4f69009fa93fe119bb", &hash(@embedFile("kernels/cuda/prefill_attention.cu")));
    try std.testing.expectEqualStrings("646835a352a3292d8c4e36344c55590994b545f5286432097745674670702a4d", &hash(@embedFile("kernels/cuda/qwen35_head_norm.cuh")));
    try std.testing.expectEqualStrings("ce78a982a0094517c4a528412df1921417098c26926b79e57f6b5884cad802ed", &hash(@embedFile("kernels/cuda/qwen35_head_rope.cuh")));
    try std.testing.expectEqualStrings("bf5b88a89721f2ff6a1573bbb4e2ba70882a6defbbe4ba0611827f1e43dcca2b", &hash(@embedFile("kernels/cuda/gdn_prefill.cu")));
    try std.testing.expectEqualStrings("1abee1683403bedd4f034eada7e772178f1780f2eac5f373ad576cc4309ca99e", &hash(@embedFile("kernels/cuda/qwen35_slots.cuh")));
    try std.testing.expectEqualStrings("af80792588566e8a1ebc58029ab90d54e534b4eacf73449f0d80bdd35615e322", &hash(@embedFile("kernels/cuda/qmm_mlp_layout.cuh")));
    try std.testing.expectEqualStrings("0d38352a6e57cf7268787576357dfe281e8596a46042caf07639e8262025527a", &hash(@embedFile("kernels/cuda/qmm_mlp_single.cuh")));
    try std.testing.expectEqualStrings("af3c488dc9feef73821feae0d5d22559e45ade5d73a59b2f9436f8a878abf481", &hash(@embedFile("kernels/cuda/qwen35_slots16.cuh")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("3b27b8e964f6e7453e962cd7a259253ae7e1940297439b4b518f05da04dd1041", &hash(group[0..16132]));
    try std.testing.expectEqualStrings("e2f8e41a089604de5de1e1a7ab799fe5fe6e88621c34cb423378efc59a947398", &hash(group));
    try std.testing.expectEqualStrings("82b19ebcf8f1828cde8c70fff8b9375fa264d630544401d5c59ece4967cb4b95", &hash(@embedFile("kernels/cuda/qmm_output_stream.cuh")));
    try std.testing.expectEqualStrings("b571f01c22a6e033c19241807d1ee28a1c4d9bb21271255ad8e8099508a2806a", &hash(@embedFile("kernels/cuda/qmm_gdn_stream.cuh")));
    try std.testing.expectEqualStrings("fb1101f0b75e07be8452e79da886930f44c27774f5f431d906edbd933c6dd9bd", &hash(@embedFile("kernels/cuda/qmm_wide_kmajor.cuh")));
    try std.testing.expectEqualStrings("f5d4386f2fc954c813d9f0536432f36cebe6c6ebd05c074f10776d2aa3b545ee", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    try std.testing.expectEqualStrings("98d7a19a2a005f4d7767c6d12ae13ee29bf525b307449b07d6c95c3ec5be27db", &hash(@embedFile("kernels/cuda/qmm_prompt_rows2.cuh")));
    try std.testing.expectEqualStrings("bf29ad254c8e1692ce860312d48b01232c0b7cc737e5b43212068c0784f20b27", &hash(@embedFile("kernels/cuda/qmm_prompt_rows3.cuh")));
    try std.testing.expectEqualStrings("aa0c6e88f0091ab0d01c19d8ad85a07664031e2e93049db56260fc4790459ede", &hash(@embedFile("kernels/cuda/qmm_prompt_persist.cuh")));
    try std.testing.expectEqualStrings("96323e03a11ef111b5c039ca418f99e4d5c00b43171c15be5c1958c41cf2274c", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows2.cuh")));
    try std.testing.expectEqualStrings("e9c3e754302e5cacd24529d775368c2e14eb485f6ef59fbd100bdeea085ad589", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows3.cuh")));
    try std.testing.expectEqualStrings("89f24972c0c9e38cf376aed85db492ce149c1e46f9cec91ae75539317f719465", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    try std.testing.expectEqualStrings("3d01151641d7cb710d8b7a8e2f7a5ac6b6a6b07c2a95b448122fb7d7a7564de8", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
    try std.testing.expectEqualStrings("60ffcb12aaf8155d8b82274299a0e21cb007d06e540c47cfeaeb6de2e69a6317", &hash(@embedFile("kernels/cuda/qwen35_draft.cu")));
    try std.testing.expectEqualStrings("4f14d7ce48fec892a4bb6d5724619ae6d495a6bb50f81d4a3bccc7a3d578dd9a", &hash(@embedFile("kernels/cuda/qwen35_keep.cu")));
    try std.testing.expectEqualStrings("d1a379a98cf7504a7ed3b3f2d24c8ab6a39fe5511bcded6449e598ce5c3c79e7", &hash(@embedFile("kernels/cuda/scan_rows.cu")));
}
test "dense serving: owned launches and projections retain their reviewed argument sequence" {
    const source = @embedFile("src/families/qwen3_5/cuda_engine.zig") ++ @embedFile("src/families/qwen3_5/cuda_forward.zig");
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, source, at, "try e.")) |start| {
        const rest = source[start..];
        const end = std.mem.indexOfScalar(u8, rest, ';') orelse return error.BadSource;
        const call = rest[0 .. end + 1];
        at = start + 6;
        if (!(std.mem.startsWith(u8, call, "try e.run(e.fns.") or std.mem.startsWith(u8, call, "try e.proj(") or
            std.mem.startsWith(u8, call, "try e.norm("))) continue;
        at = start + end + 1;
        for (call) |c| if (!std.ascii.isWhitespace(c)) {
            try out.append(std.testing.allocator, c);
        };
    }
    try std.testing.expectEqualStrings("b2ed0c08dd1c3c01cbd9eabdc85a977dd26871b0ee64a0d484c40fc8d2941baf", &hash(out.items));
}
