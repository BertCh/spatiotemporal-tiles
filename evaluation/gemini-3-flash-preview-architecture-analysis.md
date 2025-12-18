# STT Architecture Evaluation: Spatiotemporal Tiling for 120fps Rendering

**Model:** gemini-3-flash-preview
**Date:** December 18, 2025
**Subject:** Full technical evaluation of the `.stt` format and `deck.gl` rendering architecture.

---

## 1. Executive Summary

The current architecture is **highly optimized** and specifically designed to meet 120fps rendering requirements for large-scale spatiotemporal datasets. While the system appears complex, this complexity is a direct result of solving the "4D problem" (X, Y, Z, Time) while maintaining the seamless interaction experience of modern map engines like Mapbox and the smooth playback of video platforms like YouTube.

The choice of **deck.gl 9.x**, **luma.gl 9.x**, and a custom **binary columnar format** using **Protocol Buffers** provides a solid foundation for both WebGL2 and future WebGPU performance.

---

## 2. The `.stt` File Format Analysis

### Columnar Layout & Proto3

The `.stt` format uses a **columnar features layout** (`ColumnarFeatures`) which is the industry standard for high-performance analytics (similar to Apache Parquet but for spatiotemporal data).

- **Pros:** Data is stored in typed arrays (geometry, times, properties) that can be uploaded directly to the GPU with zero transformation.
- **Dequantization:** Using quantized coordinates (MVT-style) reduces tile size significantly while maintaining precision through double-precision dequantization in the loader.

### Spatial-Temporal Indexing

- **Hilbert Curve:** Maximizes spatial locality, making range requests more efficient as nearby tiles are often nearby in the file.
- **Temporal Index:** Allows $O(1)$ spatial lookup and $O(\log n)$ temporal lookup, which is critical for smooth scrubbing.
- **HTTP Range Requests:** The archive structure allows fetching headers, index, and data chunks in separate parallel requests, enabling a "streaming" experience without a specialized backend.

---

## 3. The Rendering Pipeline (120fps Strategy)

The 120fps requirement is met through several key architectural patterns:

### GPU-Side Time Filtering

Instead of filtering data in JavaScript, the system uses a custom `TimeFilterExtension`.

- **Vertex Shader filtering:** Features are "discarded" or faded in the vertex shader based on `instanceStartTime` and `instanceEndTime` attributes.
- **Low Overhead:** This moves the $O(n)$ work from the CPU to the GPU, allowing millions of points to be processed every frame.

### React Bypass

The `SpatioTemporalLayer` implements a sophisticated update pattern:

- **Tick Throttling:** The `TimeController` fires a "tick" event. The layer updates its `currentTime` uniform and calls `setNeedsRedraw()`.
- **Memoization:** In the application (e.g., `App.tsx`), the `layers` array is memoized _without_ a dependency on `currentTime`. This prevents the entire deck.gl layer tree from being destroyed and recreated 120 times per second, which would be a fatal performance bottleneck.

### Layer Instance Re-use

Subclasses like `AnimatedPointLayer` use a `layerCache` and call `clone()` on deck.gl layers. This preserves the underlying GPU buffers (VBOs) while only updating the props (uniforms), which is the fastest possible way to update a layer in deck.gl.

---

## 4. Data Management: The "YouTube" Experience

The `SpatiotemporalTileset` is the "brain" of the loading system:

- **Priority Queueing:** It prioritizes current viewport/time tiles over prefetched tiles.
- **Predictive Prefetching:** It looks at the animation speed and direction to pre-load upcoming temporal tiles. This ensures that when the "video playhead" reaches a new time window, the data is already in the browser's LRU cache.
- **LOD (Level of Detail):** The `best-available` strategy loads parent tiles first. This provides a "blurred" but immediate visual while the high-resolution tiles are streaming in, exactly like Mapbox or YouTube's progressive loading.

---

## 5. Potential Bottlenecks & Recommendations

1. **Memory Ceiling:** The default 2GB cache is generous for desktops but will crash mobile browsers. I recommend implementing a device-aware `maxCacheByteSize`.
2. **Precision Jitter:** While `timeOffset` is used, the shader still uses 32-bit floats. For datasets spanning decades with millisecond precision, you may encounter "z-fighting" in time. Ensure `timeOffset` is updated when the viewport moves significantly in time.
3. **Worker Pool Contention:** The `STTLoader` uses a worker pool. Ensure the number of workers is tuned to `navigator.hardwareConcurrency` to avoid starving the main thread's UI performance.
4. **Binary-Only Path:** Ensure that all properties are delivered via `numeric_properties` or `categorical_properties`. Any fallback to the (now reserved) `Feature` message should be strictly forbidden as it forces an $O(n)$ object-to-binary conversion.

---

## 6. Conclusion

**The architecture is sound and meets the high-performance 120fps requirement.**

The complexity is well-encapsulated:

- **Frontend developers** just use the `Animated` layers.
- **Data engineers** just use `stt-build`.
- **The system** handles the complex orchestration of Range Requests, Worker-based decoding, and GPU-side temporal filtering.

This is a world-class implementation of spatiotemporal tiling.
