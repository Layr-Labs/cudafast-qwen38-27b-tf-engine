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
    try std.testing.expectEqualStrings("e300390b015008c8e0b4260c368248f641e8db71674a8728c4be02eacbfe6245", &hash(@embedFile("kernels/cuda/lane_gemv.cu")));
    try std.testing.expectEqualStrings("a398a9aa4d257b7a2a9a4301196c369d46bfc636abff9b4f69009fa93fe119bb", &hash(@embedFile("kernels/cuda/prefill_attention.cu")));
    try std.testing.expectEqualStrings("fa0891d60566ef77ab789204d62c34aae1edf93d8443328879d8d1833947bda8", &hash(@embedFile("kernels/cuda/qwen35_head_norm.cuh")));
    try std.testing.expectEqualStrings("23235a2ba2e7c3faf67f9456db67e4379c97799f049fba09c131fdf1f2175e4d", &hash(@embedFile("kernels/cuda/qwen35_head_rope.cuh")));
    try std.testing.expectEqualStrings("72422d92d61a9b00211128a7a7f6fe9622ea0e32cf6893568e07e5f7eeccaf08", &hash(@embedFile("kernels/cuda/gdn_prefill.cu")));
    try std.testing.expectEqualStrings("31b7a69d172a17a415a9851eb77ffe3a934ee92e5a7e61a6d5951dcd7f999414", &hash(@embedFile("kernels/cuda/qwen35_slots.cuh")));
    try std.testing.expectEqualStrings("af80792588566e8a1ebc58029ab90d54e534b4eacf73449f0d80bdd35615e322", &hash(@embedFile("kernels/cuda/qmm_mlp_layout.cuh")));
    try std.testing.expectEqualStrings("0d38352a6e57cf7268787576357dfe281e8596a46042caf07639e8262025527a", &hash(@embedFile("kernels/cuda/qmm_mlp_single.cuh")));
    try std.testing.expectEqualStrings("04003b19f0de148f06cc7d1b780cf4cb18da9434de007b59e469083595e8bd88", &hash(@embedFile("kernels/cuda/qwen35_slots16.cuh")));
    try std.testing.expectEqualStrings("dd6bc83e19eb98f2a332e5a2c71db797ddb89f05ae3790a833ba1688287273b1", &hash(@embedFile("kernels/cuda/qmm_mlp_compact_pair.cuh")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("3b27b8e964f6e7453e962cd7a259253ae7e1940297439b4b518f05da04dd1041", &hash(group[0..16132]));
    try std.testing.expectEqualStrings("2886eaf132828a0becccf3b45b993fb2b7d44cf15535ca95089c4fe051862df2", &hash(group));
    try std.testing.expectEqualStrings("b050ef8c8605af2868d991fae171fe6cea295c1607589da836829c6882de1307", &hash(@embedFile("kernels/cuda/qmm_output_stream.cuh")));
    try std.testing.expectEqualStrings("22769156856a1e69bb6ff94a6a1d5e56eb962a2a74b23d35a1fce699127b2e16", &hash(@embedFile("kernels/cuda/qmm_gdn_stream.cuh")));
    try std.testing.expectEqualStrings("fb1101f0b75e07be8452e79da886930f44c27774f5f431d906edbd933c6dd9bd", &hash(@embedFile("kernels/cuda/qmm_wide_kmajor.cuh")));
    try std.testing.expectEqualStrings("f5d4386f2fc954c813d9f0536432f36cebe6c6ebd05c074f10776d2aa3b545ee", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    try std.testing.expectEqualStrings("98d7a19a2a005f4d7767c6d12ae13ee29bf525b307449b07d6c95c3ec5be27db", &hash(@embedFile("kernels/cuda/qmm_prompt_rows2.cuh")));
    try std.testing.expectEqualStrings("bf29ad254c8e1692ce860312d48b01232c0b7cc737e5b43212068c0784f20b27", &hash(@embedFile("kernels/cuda/qmm_prompt_rows3.cuh")));
    try std.testing.expectEqualStrings("aa0c6e88f0091ab0d01c19d8ad85a07664031e2e93049db56260fc4790459ede", &hash(@embedFile("kernels/cuda/qmm_prompt_persist.cuh")));
    try std.testing.expectEqualStrings("96323e03a11ef111b5c039ca418f99e4d5c00b43171c15be5c1958c41cf2274c", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows2.cuh")));
    try std.testing.expectEqualStrings("e9c3e754302e5cacd24529d775368c2e14eb485f6ef59fbd100bdeea085ad589", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows3.cuh")));
    try std.testing.expectEqualStrings("b62a44e1728b1ebe3c879e05103a6a88fb4858b302b97a9bc0936d34f4b72724", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    try std.testing.expectEqualStrings("73c078eb4bcd26c3483f24ba2ab99b3d8f6064ea4079f510dca86ec4ac4ed420", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
    try std.testing.expectEqualStrings("5c51529e499f3947f7d799d6c2a9e0f6c1522f0f6f8a50518f63d385191a84a8", &hash(@embedFile("kernels/cuda/qwen35_draft.cu")));
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
    try std.testing.expectEqualStrings("6e901d013217a8ba3bc4894db6aac42d2829a923262114f09c6145fe17c71e4e", &hash(out.items));
}
