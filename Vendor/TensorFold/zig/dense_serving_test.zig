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
    try std.testing.expectEqualStrings("a1657e44be63043d0a18241ded21ccce3f2372b9246de9a748b8b8306c980d44", &hash(@embedFile("kernels/cuda/qwen35_head_norm.cuh")));
    try std.testing.expectEqualStrings("ecbb79734ea82c6670cafda577265618aa30610a583f499cb35c5175554e14a2", &hash(@embedFile("kernels/cuda/qwen35_head_rope.cuh")));
    try std.testing.expectEqualStrings("2763bdceea63ff6afbbe094acaf72cfb96e62528ea368231fea5601a5f5ab0e2", &hash(@embedFile("kernels/cuda/gdn_prefill.cu")));
    try std.testing.expectEqualStrings("9f12eaf351184a1ba307d7e6236581a12ee7be566e2af78127cb62eac5c3063c", &hash(@embedFile("kernels/cuda/qwen35_slots.cuh")));
    try std.testing.expectEqualStrings("af80792588566e8a1ebc58029ab90d54e534b4eacf73449f0d80bdd35615e322", &hash(@embedFile("kernels/cuda/qmm_mlp_layout.cuh")));
    try std.testing.expectEqualStrings("0d38352a6e57cf7268787576357dfe281e8596a46042caf07639e8262025527a", &hash(@embedFile("kernels/cuda/qmm_mlp_single.cuh")));
    try std.testing.expectEqualStrings("d134cbe4caa1276dac9cedad9da910b0bcfe19cd4b20ba7588273b75556d8525", &hash(@embedFile("kernels/cuda/qwen35_slots16.cuh")));
    try std.testing.expectEqualStrings("c6881a073d0c6f301449501d63630dd6ba7ca1c8b85bd5ea7da4dd2b49245611", &hash(@embedFile("kernels/cuda/qmm_mlp_compact_pair.cuh")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("3b27b8e964f6e7453e962cd7a259253ae7e1940297439b4b518f05da04dd1041", &hash(group[0..16132]));
    try std.testing.expectEqualStrings("318274c83bf1112b42607c89e1603f5bb056c3c4b60dbf45a46d070c3955298e", &hash(group));
    try std.testing.expectEqualStrings("93909d39aa18fea062320cebbb1acc6d27d60ec3109b617d69c5ac4a22b0a053", &hash(@embedFile("kernels/cuda/qmm_output_stream.cuh")));
    try std.testing.expectEqualStrings("6a5eaebeecca0b9511dc21370368f519f6d93577d90dad74ea4ef3db599f25b5", &hash(@embedFile("kernels/cuda/qmm_gdn_stream.cuh")));
    try std.testing.expectEqualStrings("fb1101f0b75e07be8452e79da886930f44c27774f5f431d906edbd933c6dd9bd", &hash(@embedFile("kernels/cuda/qmm_wide_kmajor.cuh")));
    try std.testing.expectEqualStrings("98046f76117f421d18b56416b07b82a21253db5769172e13fc38dfbb506503d0", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    try std.testing.expectEqualStrings("98d7a19a2a005f4d7767c6d12ae13ee29bf525b307449b07d6c95c3ec5be27db", &hash(@embedFile("kernels/cuda/qmm_prompt_rows2.cuh")));
    try std.testing.expectEqualStrings("bf29ad254c8e1692ce860312d48b01232c0b7cc737e5b43212068c0784f20b27", &hash(@embedFile("kernels/cuda/qmm_prompt_rows3.cuh")));
    try std.testing.expectEqualStrings("474b0a0054f31d60841b02a25b72ed3b0cc1ad4d32fc83a2ef8109caedfdcac5", &hash(@embedFile("kernels/cuda/qmm_prompt_persist.cuh")));
    try std.testing.expectEqualStrings("f92150d13431454a8f30b2136340a9c8b8bc7c057724037a5cb10d833e72d4f0", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows2.cuh")));
    try std.testing.expectEqualStrings("8b5f26364c6b7cac7beb7e935ae7b6e0210c4e281226f5f786137c46766c0c06", &hash(@embedFile("kernels/cuda/qmm_prompt_persist_rows3.cuh")));
    try std.testing.expectEqualStrings("f2e33b5a29d0bc17cb67dd5e5c35d5d1a6793a08ac3621f5fca068fc7754bb3c", &hash(@embedFile("kernels/cuda/qmm_prompt_pipe.cuh")));
    try std.testing.expectEqualStrings("6a68a35f350990dc0c4f44203d2a70ea574dd896d26028657df808ee921ddf3b", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    try std.testing.expectEqualStrings("d45373c32730df17d1269fd62bcb21048af0b374acb8c433de4c1b13a2c460a7", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
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
