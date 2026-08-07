use super::{BackendErrorCode, Error, PhysicalSize, Result};
use std::ops::Range;

const RGBA8_BYTES_PER_PIXEL: u64 = 4;
const COPY_BYTES_PER_ROW_ALIGNMENT: u64 = 256;

pub(super) struct ReadbackLayout {
    width: u32,
    height: u32,
    row_bytes: usize,
    padded_bytes_per_row: u32,
    buffer_size: u64,
    decoded_len: usize,
    mapped_range: ValidatedMappedRange,
}

impl ReadbackLayout {
    pub(super) fn try_new(size: PhysicalSize) -> Result<Self> {
        let width = size.width();
        let height = size.height();
        let row_bytes_u64 = u64::from(width)
            .checked_mul(RGBA8_BYTES_PER_PIXEL)
            .ok_or_else(|| readback_failed("readback row byte count overflowed"))?;
        let padded_bytes_per_row_u64 = row_bytes_u64
            .checked_add(COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            .map(|bytes| bytes / COPY_BYTES_PER_ROW_ALIGNMENT * COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| readback_failed("aligned readback row byte count overflowed"))?;
        let padded_bytes_per_row = u32::try_from(padded_bytes_per_row_u64)
            .map_err(|_| readback_failed("aligned readback row byte count exceeds WGPU limits"))?;
        let buffer_size = padded_bytes_per_row_u64
            .checked_mul(u64::from(height))
            .ok_or_else(|| readback_failed("readback staging buffer size overflowed"))?;
        let row_bytes = usize::try_from(row_bytes_u64)
            .map_err(|_| readback_failed("readback row byte count exceeds addressable memory"))?;
        let decoded_len = row_bytes
            .checked_mul(
                usize::try_from(height)
                    .map_err(|_| readback_failed("readback height exceeds addressable memory"))?,
            )
            .ok_or_else(|| readback_failed("decoded readback byte count overflowed"))?;
        let mapped_range = ValidatedMappedRange::try_new(buffer_size)?;
        Ok(Self {
            width,
            height,
            row_bytes,
            padded_bytes_per_row,
            buffer_size,
            decoded_len,
            mapped_range,
        })
    }

    pub(super) const fn width(&self) -> u32 {
        self.width
    }

    pub(super) const fn height(&self) -> u32 {
        self.height
    }

    pub(super) const fn padded_bytes_per_row(&self) -> u32 {
        self.padded_bytes_per_row
    }

    pub(super) const fn buffer_size(&self) -> u64 {
        self.buffer_size
    }

    pub(super) fn mapped_range(&self) -> Range<wgpu::BufferAddress> {
        self.mapped_range.bytes()
    }
}

#[derive(Clone)]
struct ValidatedMappedRange {
    bytes: Range<wgpu::BufferAddress>,
}

impl ValidatedMappedRange {
    fn try_new(buffer_size: u64) -> Result<Self> {
        let bytes = 0..buffer_size;
        let length = bytes
            .end
            .checked_sub(bytes.start)
            .ok_or_else(|| readback_failed("readback mapped range was reversed"))?;
        if length == 0 {
            return Err(readback_failed("readback mapped range must be nonempty"));
        }
        if bytes.start % wgpu::MAP_ALIGNMENT != 0 {
            return Err(readback_failed(
                "readback mapped range offset was not map-aligned",
            ));
        }
        if length % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(readback_failed(
                "readback mapped range length was not four-byte aligned",
            ));
        }
        Ok(Self { bytes })
    }

    fn bytes(&self) -> Range<wgpu::BufferAddress> {
        self.bytes.clone()
    }
}

pub(super) fn decode_padded_rows(layout: &ReadbackLayout, mapped: &[u8]) -> Result<Vec<u8>> {
    let mut rgba = Vec::with_capacity(layout.decoded_len);
    for row in 0..layout.height {
        let start = u64::from(row)
            .checked_mul(u64::from(layout.padded_bytes_per_row))
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| readback_failed("mapped readback row offset overflowed"))?;
        let end = start
            .checked_add(layout.row_bytes)
            .ok_or_else(|| readback_failed("mapped readback row end overflowed"))?;
        let row = mapped
            .get(start..end)
            .ok_or_else(|| readback_failed("mapped readback row was incomplete"))?;
        rgba.extend_from_slice(row);
    }
    if rgba.len() != layout.decoded_len {
        return Err(readback_failed(
            "decoded readback byte count did not match the validated layout",
        ));
    }
    Ok(rgba)
}

fn readback_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::ReadbackFailed, message)
}
