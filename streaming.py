from typing import Set, Tuple

import numpy as np
import pyrr

from settings import (
    CHUNK_WIDTH,
    CHUNK_DEPTH,
    RENDER_DISTANCE,
    UNLOAD_DISTANCE,
)

from world import Chunk, procedural_height, bake_splatmap_for_chunk


def _stitch_splatmap_borders(world, cx: int, cz: int) -> None:
    """Make splatmap edges continuous between a chunk and any existing neighbors.

    This averages the weights along shared edges so both chunks use identical
    border rows/columns, removing visible seams when sampling separate
    splatmap textures per chunk.
    """
    key = (cx, 0, cz)
    chunk = world.chunks.get(key)
    if chunk is None:
        return

    # Helper to average two 1D edges (shape: (N, 3)) and write back
    def avg_edges(edge_a, edge_b):
        avg = (edge_a + edge_b) * 0.5
        return avg

    # Left neighbor (cx-1, cz): neighbor right edge <-> chunk left edge
    nb = world.chunks.get((cx - 1, 0, cz))
    if nb is not None:
        # shape (CHUNK_DEPTH+1, 3)
        left_edge = chunk.splatmap[:, 0, :]
        right_edge_nb = nb.splatmap[:, CHUNK_WIDTH, :]
        avg = avg_edges(left_edge, right_edge_nb)
        chunk.splatmap[:, 0, :] = avg
        nb.splatmap[:, CHUNK_WIDTH, :] = avg
        chunk.splatmap_is_dirty = True
        nb.splatmap_is_dirty = True

    # Right neighbor (cx+1, cz): neighbor left edge <-> chunk right edge
    nb = world.chunks.get((cx + 1, 0, cz))
    if nb is not None:
        right_edge = chunk.splatmap[:, CHUNK_WIDTH, :]
        left_edge_nb = nb.splatmap[:, 0, :]
        avg = avg_edges(right_edge, left_edge_nb)
        chunk.splatmap[:, CHUNK_WIDTH, :] = avg
        nb.splatmap[:, 0, :] = avg
        chunk.splatmap_is_dirty = True
        nb.splatmap_is_dirty = True

    # Top neighbor (cx, cz-1): neighbor bottom edge <-> chunk top edge
    nb = world.chunks.get((cx, 0, cz - 1))
    if nb is not None:
        top_edge = chunk.splatmap[0, :, :]
        bottom_edge_nb = nb.splatmap[CHUNK_DEPTH, :, :]
        avg = avg_edges(top_edge, bottom_edge_nb)
        chunk.splatmap[0, :, :] = avg
        nb.splatmap[CHUNK_DEPTH, :, :] = avg
        chunk.splatmap_is_dirty = True
        nb.splatmap_is_dirty = True

    # Bottom neighbor (cx, cz+1): neighbor top edge <-> chunk bottom edge
    nb = world.chunks.get((cx, 0, cz + 1))
    if nb is not None:
        bottom_edge = chunk.splatmap[CHUNK_DEPTH, :, :]
        top_edge_nb = nb.splatmap[0, :, :]
        avg = avg_edges(bottom_edge, top_edge_nb)
        chunk.splatmap[CHUNK_DEPTH, :, :] = avg
        nb.splatmap[0, :, :] = avg
        chunk.splatmap_is_dirty = True
        nb.splatmap_is_dirty = True


def ensure_chunk_by_coord(world, cx: int, cz: int) -> Chunk:
    key = (cx, 0, cz)
    if key in world.chunks:
        return world.chunks[key]
    pos = pyrr.Vector3([cx, 0, cz], dtype=np.int32)
    chunk = Chunk(pos)
    for x in range(CHUNK_WIDTH + 1):
        for z in range(CHUNK_DEPTH + 1):
            wx = cx * CHUNK_WIDTH + x
            wz = cz * CHUNK_DEPTH + z
            chunk.heightmap[x, z] = procedural_height(wx, wz)
    bake_splatmap_for_chunk(chunk)
    # Persisted override (if any)
    try:
        world.try_load_chunk_from_disk(cx, cz, chunk)
    except Exception:
        pass
    world.chunks[key] = chunk
    # Stitch borders with already-present neighbors to avoid visual seams
    _stitch_splatmap_borders(world, cx, cz)
    chunk.is_meshing = True
    world.request_queue.put(chunk)
    return chunk


def ensure_chunk_at_world_pos(world, world_x: float, world_z: float) -> Chunk:
    cx = int(np.floor(world_x / CHUNK_WIDTH))
    cz = int(np.floor(world_z / CHUNK_DEPTH))
    return ensure_chunk_by_coord(world, cx, cz)


def update_streaming(world, center_world_pos) -> None:
    if center_world_pos is None:
        return
    cx = int(np.floor(center_world_pos.x / CHUNK_WIDTH))
    cz = int(np.floor(center_world_pos.z / CHUNK_DEPTH))

    desired: Set[Tuple[int, int, int]] = set()
    for dz in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
        for dx in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
            desired.add((cx + dx, 0, cz + dz))

    for key in desired:
        if key not in world.chunks:
            ensure_chunk_by_coord(world, key[0], key[2])

    to_remove = []
    for key in list(world.chunks.keys()):
        kx, _, kz = key
        dx = abs(kx - cx)
        dz = abs(kz - cz)
        if max(dx, dz) > UNLOAD_DISTANCE:
            to_remove.append(key)

    for key in to_remove:
        chunk = world.chunks.pop(key)
        try:
            # Save edits before freeing GPU memory
            world.save_chunk_to_disk(chunk)
            chunk.destroy()
        except Exception:
            pass
