//! qmm_prompt_persist_bal.cuh's slot lists for one 5120-column projection in four 128-row bands on 96 slots.
//!
//! Slot s starts with the 128-column tile of panel s / 4 (columns 0 .. 3071) in band s % 4. The last 2048 columns of
//! each band are cut into 64- and 32-column tiles and dealt so that slots s and s + 48 (the two CTAs of one SM) carry
//! about 25.8 of the 1240 (16-row, 128-column) units each: no slot past 14 units, no SM past 26.25, where the 160 whole
//! tiles and their split leave a second wave a third full. An entry is band (bits 0-1), width (bits 2-3: 128, 64, 32
//! columns) and first column / 32 (bits 4-11); every (band, 32-column block) appears exactly once (tested below).
const std = @import("std");

pub const slots = 96;
pub const max_tiles = 320;
pub const cols = 5120;
pub const bands = 4;

/// The kernel's Sched, passed by value.
pub const Sched = extern struct { start: [slots + 1]u16, desc: [max_tiles]u16 };

const start_list = [_]u16{
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
    64, 66, 68, 70, 73, 75, 77, 79, 82, 84, 86, 88, 91, 93, 95, 97,
    100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145,
    148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193,
    196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241,
    244,
};
const desc_list = [_]u16{
    0, 1540, 1, 1542, 2, 1573, 3, 2277, 64, 1604, 65, 1606, 66, 1637, 67, 2278,
    128, 1668, 129, 1670, 130, 1701, 131, 2308, 192, 1732, 193, 1734, 194, 1765, 195, 2309,
    256, 1796, 257, 1798, 258, 1829, 259, 2310, 320, 1860, 321, 1862, 322, 1893, 323, 2340,
    384, 1924, 385, 1926, 386, 1957, 387, 2341, 448, 1988, 449, 1990, 450, 2021, 451, 2342,
    512, 2052, 513, 2054, 514, 2085, 515, 2119, 2507, 576, 2116, 577, 2118, 578, 2149, 579,
    2215, 2523, 640, 2180, 641, 2182, 642, 2213, 643, 2311, 2539, 704, 2244, 705, 2246, 706,
    2276, 707, 2407, 2555, 768, 1541, 2376, 769, 1572, 2377, 770, 1574, 2378, 771, 1607, 1639,
    832, 1605, 2392, 833, 1636, 2393, 834, 1638, 2394, 835, 1671, 1703, 896, 1669, 2408, 897,
    1700, 2409, 898, 1702, 2410, 899, 1735, 1767, 960, 1733, 2424, 961, 1764, 2425, 962, 1766,
    2426, 963, 1799, 1831, 1024, 1797, 2440, 1025, 1828, 2441, 1026, 1830, 2442, 1027, 1863, 1895,
    1088, 1861, 2456, 1089, 1892, 2457, 1090, 1894, 2458, 1091, 1927, 1959, 1152, 1925, 2472, 1153,
    1956, 2473, 1154, 1958, 2474, 1155, 1991, 2023, 1216, 1989, 2488, 1217, 2020, 2489, 1218, 2022,
    2490, 1219, 2055, 2087, 1280, 2053, 2504, 1281, 2084, 2505, 1282, 2086, 2506, 1283, 2151, 2183,
    1344, 2117, 2520, 1345, 2148, 2521, 1346, 2150, 2522, 1347, 2247, 2279, 1408, 2181, 2536, 1409,
    2212, 2537, 1410, 2214, 2538, 1411, 2343, 2375, 1472, 2245, 2552, 1473, 1543, 2553, 1474, 1575,
    2554, 1475, 2439, 2471,
};

pub const sched: Sched = blk: {
    var s: Sched = .{ .start = start_list, .desc = @splat(0) };
    for (desc_list, 0..) |d, i| s.desc[i] = d;
    break :blk s;
};

comptime {
    std.debug.assert(@sizeOf(Sched) == (slots + 1 + max_tiles) * 2);
    std.debug.assert(start_list.len == slots + 1 and start_list[slots] == desc_list.len and desc_list.len <= max_tiles);
}

test "the balanced lists cover every band's 5120 columns once" {
    var seen: [bands][cols / 32]u8 = @splat(@splat(0));
    for (0..slots) |s| {
        try std.testing.expect(sched.start[s] <= sched.start[s + 1]);
        for (sched.start[s]..sched.start[s + 1]) |i| {
            const d = sched.desc[i];
            const band = d & 3;
            const width: usize = @as(usize, 128) >> @intCast((d >> 2) & 3);
            const n0: usize = (d >> 4) * 32;
            try std.testing.expect(n0 % width == 0 and n0 + width <= cols);
            for (n0 / 32..(n0 + width) / 32) |c| seen[band][c] += 1;
        }
    }
    for (seen) |row| for (row) |n| try std.testing.expectEqual(@as(u8, 1), n);
}
