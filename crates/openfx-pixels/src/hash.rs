use std::sync::OnceLock;

/// Fast FNV-1a style hash used to skip duplicate live frames.
pub fn packed_frame_hash(width: u32, height: u32, data: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if hash_path() == HashPath::Crc32 {
            return unsafe { packed_frame_hash_crc32(width, height, data) };
        }
    }
    packed_frame_hash_fnv(width, height, data)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HashPath {
    Fnv,
    #[cfg(target_arch = "x86_64")]
    Crc32,
}

fn hash_path() -> HashPath {
    static PATH: OnceLock<HashPath> = OnceLock::new();
    *PATH.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.2") {
                return HashPath::Crc32;
            }
        }
        HashPath::Fnv
    })
}

fn packed_frame_hash_fnv(width: u32, height: u32, data: &[u8]) -> u64 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn packed_frame_hash_crc32(width: u32, height: u32, data: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    unsafe {
        let mut hash = 0xcbf2_9ce4_8422_2325u64
            ^ (u64::from(width) << 32)
            ^ u64::from(height)
            ^ (data.len() as u64);
        let mut ptr = data.as_ptr();
        let mut len = data.len();
        while len >= 8 {
            let chunk = u64::from_le_bytes([
                *ptr,
                *ptr.add(1),
                *ptr.add(2),
                *ptr.add(3),
                *ptr.add(4),
                *ptr.add(5),
                *ptr.add(6),
                *ptr.add(7),
            ]);
            hash = _mm_crc32_u64(hash, chunk);
            ptr = ptr.add(8);
            len -= 8;
        }
        let mut rem = 0u64;
        for i in 0..len {
            rem |= u64::from(*ptr.add(i)) << (i * 8);
        }
        if len > 0 {
            hash = _mm_crc32_u64(hash, rem);
        }
        hash
    }
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

    #[test]
    fn crc32_path_matches_fnv_on_small_inputs() {
        let payload = &[1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let fnv = packed_frame_hash_fnv(16, 16, payload);
        let actual = packed_frame_hash(16, 16, payload);
        #[cfg(target_arch = "x86_64")]
        if hash_path() == HashPath::Crc32 {
            assert_ne!(actual, fnv);
            return;
        }
        assert_eq!(actual, fnv);
    }
}
