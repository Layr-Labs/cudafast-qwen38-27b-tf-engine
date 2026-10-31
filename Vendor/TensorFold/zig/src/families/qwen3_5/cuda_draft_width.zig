//! Stream-local width selection from accepted draft rows, without extra target work.
/// The offered own-node prefix excludes any context-copy graft following it.
pub fn extraNodes(path: []const u32, offered: usize) usize {
    var count: usize = 0;
    for (path) |row| count += @intFromBool(row >= 16 and row <= offered);
    return count;
}

pub const State = struct {
    mode: enum { pilot, wide, narrow, retry } = .pilot,
    rounds: u8 = 0,
    extra: u16 = 0,

    pub fn nodes(s: State) usize {
        return if (s.mode == .narrow) 15 else 31;
    }

    /// `extra` counts accepted proposal nodes outside the ancestry-closed first fifteen.
    pub fn observe(s: *State, extra: usize) void {
        if (s.mode == .wide) return;
        if (s.mode == .pilot or s.mode == .retry) s.extra += @intCast(@min(extra, 31));
        s.rounds += 1;
        const span: u8 = switch (s.mode) { .pilot => 12, .narrow => 16, .retry => 2, .wide => unreachable };
        if (s.rounds < span) return;
        s.mode = switch (s.mode) {
            .pilot => if (s.extra >= 2) .wide else .narrow,
            .narrow => .retry,
            .retry => if (s.extra >= 1) .wide else .narrow,
            .wide => unreachable,
        };
        s.rounds = 0;
        s.extra = 0;
    }
};
