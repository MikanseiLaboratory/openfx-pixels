/// Fast FNV-1a style hash used to skip duplicate live frames.
pub fn packed_frame_hash(width: u32, height: u32, data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64
        ^ (u64::from(width) << 32)
        ^ u64::from(height)
        ^ (data.len() as u64);
    let mut chunks = data.chunks_exact(8);
    for chunk in chunks.by_ref() {
        hash ^= u64::from_le_bytes(chunk.try_into().unwrap());
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    for &b in chunks.remainder() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_payload_hashes_equal() {
        let a = packed_frame_hash(16, 16, &[1, 2, 3, 4]);
        let b = packed_frame_hash(16, 16, &[1, 2, 3, 4]);
        assert_eq!(a, b);
        assert_ne!(a, packed_frame_hash(16, 16, &[1, 2, 3, 5]));
    }
}
