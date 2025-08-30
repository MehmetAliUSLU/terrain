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
