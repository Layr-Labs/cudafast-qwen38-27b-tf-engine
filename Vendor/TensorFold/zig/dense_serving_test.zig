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
    try std.testing.expectEqualStrings("bf5b88a89721f2ff6a1573bbb4e2ba70882a6defbbe4ba0611827f1e43dcca2b", &hash(@embedFile("kernels/cuda/gdn_prefill.cu")));
    try std.testing.expectEqualStrings("9047e42ef52d26632eb50610f1b736ca40f9f066330aa5fb15029fdde301b23d", &hash(@embedFile("kernels/cuda/qwen35_slots.cuh")));
    try std.testing.expectEqualStrings("af80792588566e8a1ebc58029ab90d54e534b4eacf73449f0d80bdd35615e322", &hash(@embedFile("kernels/cuda/qmm_mlp_layout.cuh")));
    try std.testing.expectEqualStrings("0d38352a6e57cf7268787576357dfe281e8596a46042caf07639e8262025527a", &hash(@embedFile("kernels/cuda/qmm_mlp_single.cuh")));
    try std.testing.expectEqualStrings("b0d8ddf48fcf4083aaf30c81ad770f41b1a8742806464e172c2d26b6e98dacc9", &hash(@embedFile("kernels/cuda/qwen35_slots16.cuh")));
    try std.testing.expectEqualStrings("da7a4ffc2fa00dac5bde38280d77bffcf87d9020ed2094f8fcfc3d659d715717", &hash(@embedFile("kernels/cuda/qmm_mlp_compact_pair.cuh")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("3b27b8e964f6e7453e962cd7a259253ae7e1940297439b4b518f05da04dd1041", &hash(group[0..16132]));
    try std.testing.expectEqualStrings("e24e2d386cb63c76c6a9957976440d9712d10c18caf793516dc0ec3c29ebdaf4", &hash(group));
    try std.testing.expectEqualStrings("b050ef8c8605af2868d991fae171fe6cea295c1607589da836829c6882de1307", &hash(@embedFile("kernels/cuda/qmm_output_stream.cuh")));
    try std.testing.expectEqualStrings("db1f8ab74a654ea0d9858a4eb20f62346f8c016fe17435cb5b3ea292661262a0", &hash(@embedFile("kernels/cuda/qmm_gdn_stream.cuh")));
    try std.testing.expectEqualStrings("fb1101f0b75e07be8452e79da886930f44c27774f5f431d906edbd933c6dd9bd", &hash(@embedFile("kernels/cuda/qmm_wide_kmajor.cuh")));
    try std.testing.expectEqualStrings("f5d4386f2fc954c813d9f0536432f36cebe6c6ebd05c074f10776d2aa3b545ee", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    try std.testing.expectEqualStrings("98d7a19a2a005f4d7767c6d12ae13ee29bf525b307449b07d6c95c3ec5be27db", &hash(@embedFile("kernels/cuda/qmm_prompt_rows2.cuh")));
    try std.testing.expectEqualStrings("bf29ad254c8e1692ce860312d48b01232c0b7cc737e5b43212068c0784f20b27", &hash(@embedFile("kernels/cuda/qmm_prompt_rows3.cuh")));
    try std.testing.expectEqualStrings("aa0c6e88f0091ab0d01c19d8ad85a07664031e2e93049db56260fc4790459ede", &hash(@embedFile("kernels/cuda/qmm_prompt_persist.cuh")));
    try std.testing.expectEqualStrings("96323e03a11ef111b5c039ca418f99e4d5c00b43171c15be5c1958c41cf2274c", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows2.cuh")));
    try std.testing.expectEqualStrings("e9c3e754302e5cacd24529d775368c2e14eb485f6ef59fbd100bdeea085ad589", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows3.cuh")));
    try std.testing.expectEqualStrings("9e85707c0673d4a27d1ea03f602532528af8d7b9784ca31550f837db3d62ec63", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    try std.testing.expectEqualStrings("e6a57b7545249e0bcdda36c677792456ed76d0a35aa3cd88d43601b38061fd3d", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
    try std.testing.expectEqualStrings("d653c8357a6f20ae2aa15880d581932d186ed2667d1347b688b1be3a12b4a421", &hash(@embedFile("kernels/cuda/qwen35_draft.cu")));
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
